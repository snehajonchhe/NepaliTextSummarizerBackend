# from IPython import get_ipython
# from IPython.display import display
import pandas as pd
import re
import unicodedata
from sentence_transformers import SentenceTransformer, util
from indicnlp.tokenize import sentence_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from functools import lru_cache
import os



# Model and stopwords loading
STOPWORDS_PATH = os.path.join(os.path.dirname(__file__), "stopwords.txt")
EXTRACTIVE_MODEL_NAME = 'all-mpnet-base-v2'
sent_model = None

def get_sent_model():
    global sent_model
    if sent_model is None:
        sent_model = SentenceTransformer(EXTRACTIVE_MODEL_NAME)
    return sent_model

@lru_cache(maxsize=1)
def get_stopwords():
    if not os.path.exists(STOPWORDS_PATH):
        return []
    with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def clean_nepali_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\u0900-\u097F\s।?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = unicodedata.normalize('NFC', text)
    nepali_numbers = '०१२३४५६७८९'
    arabic_numbers = '0123456789'
    translation_table = str.maketrans(nepali_numbers, arabic_numbers)
    text = text.translate(translation_table)
    return text

def remove_stopwords(text):
    nepali_stopwords = get_stopwords()
    words = text.split()
    filtered_words = [word for word in words if word not in nepali_stopwords]
    return ' '.join(filtered_words)

def calculate_embeddings(sentences):
    model = get_sent_model()
    return model.encode(sentences)

def score_sentences_tfidf(sentences):
    if not sentences:
        return np.array([])
    vectorizer = TfidfVectorizer()
    tfidf_vectors = vectorizer.fit_transform(sentences)
    return tfidf_vectors.sum(axis=1).A1

def score_sentences_similarity(embeddings):
    if len(embeddings) == 0:
        return np.array([])
    centroid_embedding = np.mean(embeddings, axis=0)
    return cosine_similarity(embeddings, [centroid_embedding])

def score_sentences_textrank(embeddings):
    if len(embeddings) == 0:
        return {}
    similarity_matrix = cosine_similarity(embeddings)
    nx_graph = nx.from_numpy_array(similarity_matrix)
    return nx.pagerank(nx_graph)

def safe_normalize(scores):
    if len(scores) == 0:
        return scores
    min_val = np.min(scores)
    max_val = np.max(scores)
    if max_val == min_val:
        return np.ones_like(scores)
    return (scores - min_val) / (max_val - min_val)

def generate_summary(text, num_sentences_in_summary=None):
    if not text.strip():
        return ""
        
    cleaned_text = clean_nepali_text(text)
    # Don't remove stopwords for sentence segmentation, only for scoring? 
    # Usually extractive summarization uses the original sentences.
    
    # Original sentences for final output
    raw_sentences = sentence_tokenize.sentence_split(text, lang='ne')
    if not raw_sentences:
        return ""
        
    # Processed sentences for scoring
    processed_sentences = [remove_stopwords(clean_nepali_text(s)) for s in raw_sentences]
    processed_sentences = [s for s in processed_sentences if s.strip()]
    
    if not processed_sentences:
        return text # Return original if processing fails

    # Sentence Embedding
    embeddings = calculate_embeddings(processed_sentences)

    # Sentence Scoring
    tfidf_scores = score_sentences_tfidf(processed_sentences)
    similarity_scores = score_sentences_similarity(embeddings).flatten()
    textrank_scores = list(score_sentences_textrank(embeddings).values())

    num_sentences = len(processed_sentences)
    tfidf_scores = tfidf_scores[:num_sentences]
    similarity_scores = similarity_scores[:num_sentences]
    textrank_scores = np.array(textrank_scores[:num_sentences])

    # Normalize Scores
    normalized_tfidf = safe_normalize(tfidf_scores)
    normalized_similarity = safe_normalize(similarity_scores)
    normalized_textrank = safe_normalize(textrank_scores)

    # Combine Scores
    tfidf_weight = 0.3
    similarity_weight = 0.4
    textrank_weight = 0.3
    combined_scores = tfidf_weight * normalized_tfidf + similarity_weight * normalized_similarity + textrank_weight * normalized_textrank

    # Sentence Selection
    if num_sentences_in_summary is None:
        num_sentences_in_summary = max(1, int(num_sentences * 0.3))
    elif isinstance(num_sentences_in_summary, float) and 0 < num_sentences_in_summary <= 1:
        num_sentences_in_summary = max(1, int(num_sentences * num_sentences_in_summary))
    elif num_sentences_in_summary > num_sentences:
        num_sentences_in_summary = num_sentences

    if num_sentences_in_summary <= 0:
        return ""

    top_sentence_indices = np.argpartition(combined_scores, -num_sentences_in_summary)[-num_sentences_in_summary:]
    top_sentence_indices = sorted(top_sentence_indices)
    
    # Map back to raw sentences (assuming they align)
    summary_sentences = [raw_sentences[index] for index in top_sentence_indices]

    return ' '.join(summary_sentences)