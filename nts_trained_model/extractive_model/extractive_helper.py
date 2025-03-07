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



def clean_nepali_text(text):
    # Remove HTML tags using a more robust regex
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation and special characters (except for Devanagari script)
    # Keep common Nepali punctuation: ।, ?
    text = re.sub(r'[^\u0900-\u097F\s।?]', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Unicode normalization
    text = unicodedata.normalize('NFC', text)

    # 6. Handle Nepali numbers (optional - convert to Arabic numerals)
    nepali_numbers = '०१२३४५६७८९'
    arabic_numbers = '0123456789'
    translation_table = str.maketrans(nepali_numbers, arabic_numbers)
    text = text.translate(translation_table)

    return text

def remove_stopwords(text):
    with open( os.path.join(os.path.dirname(__file__), "stopwords.txt"), 'r', encoding='utf-8') as f:  # Assuming UTF-8 encoding
        nepali_stopwords = [line.strip() for line in f] # Indented this line by 4 spaces

    words = text.split()
    filtered_words = [word for word in words if word not in nepali_stopwords]
    filtered_text = ' '.join(filtered_words)
    return filtered_text


from functools import lru_cache
import hashlib

@lru_cache(maxsize=128)
def calculate_embeddings(sentences):
    # Convert the list of sentences into a tuple to make it hashable
    sentences_tuple = tuple(sentences) # converting list to tuple to make it hashable for lru_cache

    model = SentenceTransformer('all-mpnet-base-v2')  # or a Nepali-specific model
    sentence_embeddings = model.encode(sentences_tuple)  # Use sentences_tuple here
    return sentence_embeddings


def score_sentences_tfidf(sentences):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(sentences)
    tfidf_vectors = vectorizer.transform(sentences)
    return tfidf_vectors.sum(axis=1).A1

def score_sentences_similarity(embeddings):
    centroid_embedding = np.mean(embeddings, axis=0)
    return cosine_similarity(embeddings, [centroid_embedding])

def score_sentences_textrank(embeddings):
    similarity_matrix = cosine_similarity(embeddings)
    nx_graph = nx.from_numpy_array(similarity_matrix)
    return nx.pagerank(nx_graph)

def rank_array(arr):
    # If arr is a dictionary, convert its values to a list before flattening
    if isinstance(arr, dict):
        arr = list(arr.values())

    # Now, flatten the array (if it's a NumPy array or a nested list)
    arr = np.array(arr).flatten()
    temp = arr.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(arr))
    return ranks

def generate_summary(text, num_sentences_in_summary=None):
    cleaned_text = clean_nepali_text(text)
    filtered_text =remove_stopwords(cleaned_text)

    # Sentence Segmentation
    global sentences # Declare sentences as global to modify it within the function
    sentences = sentence_tokenize.sentence_split(filtered_text, lang='ne')

    # Create a hash of the sentences list
    sentences_hash = hashlib.sha256(str(sentences).encode()).hexdigest()

    # Sentence Embedding
    embeddings = calculate_embeddings(sentences_hash)

    # Sentence Scoring
    tfidf_scores = score_sentences_tfidf(sentences)
    similarity_scores = score_sentences_similarity(embeddings).flatten()
    textrank_scores = list(score_sentences_textrank(embeddings).values())

    # Ensure all score arrays have the same length as the number of sentences
    num_sentences = len(sentences)
    tfidf_scores = tfidf_scores[:num_sentences]
    similarity_scores = similarity_scores[:num_sentences]
    textrank_scores = textrank_scores[:num_sentences]

    # Normalize Scores
    normalized_tfidf = (tfidf_scores - np.min(tfidf_scores)) / (np.max(tfidf_scores) - np.min(tfidf_scores))
    normalized_similarity = (similarity_scores - np.min(similarity_scores)) / (np.max(similarity_scores) - np.min(similarity_scores))
    normalized_textrank = (np.array(textrank_scores) - np.min(np.array(textrank_scores))) / (np.max(np.array(textrank_scores)) - np.min(np.array(textrank_scores)))

    # Combine Scores
    tfidf_weight = 0.3
    similarity_weight = 0.4
    textrank_weight = 0.3
    combined_scores = tfidf_weight * normalized_tfidf + similarity_weight * normalized_similarity + textrank_weight * normalized_textrank

    # Sentence Selection
    total_sentences = len(sentences)

    if num_sentences_in_summary is None:
        num_sentences_in_summary = int(total_sentences * 0.3)  # Default to 30% if not specified
    elif isinstance(num_sentences_in_summary, float) and 0 < num_sentences_in_summary <= 1:
        num_sentences_in_summary = int(total_sentences * num_sentences_in_summary)  # Treat as a percentage
    elif num_sentences_in_summary > total_sentences:
        num_sentences_in_summary = total_sentences  # Cap at the total number of sentences

    # --- Sentence Selection and Return ---
    top_sentence_indices = np.argpartition(combined_scores, -num_sentences_in_summary)[-num_sentences_in_summary:]
    top_sentence_indices = sorted(top_sentence_indices)
    summary_sentences = [sentences[index] for index in top_sentence_indices]

    # Return Summary
    return ' '.join(summary_sentences)