import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from peft import PeftModel
import os

class SummarizationModel:
    def __init__(self):
        mbart_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "nts_trained_model", "mbart-lora-nepali"))
        self.model = MBartForConditionalGeneration.from_pretrained(mbart_model_path)
        self.tokenizer = MBart50TokenizerFast.from_pretrained(mbart_model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def summarize_text(self, text: str, max_length: int = 150, num_beams: int = 5):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True) # Ensure input is on the correct device
        summary_ids = self.model.generate(**inputs, max_length=max_length, num_beams=num_beams, early_stopping=True,repetition_penalty=2.0,no_repeat_ngram_size=3)
        summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary_text