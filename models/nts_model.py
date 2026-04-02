import torch
from transformers import MBartForConditionalGeneration, MBartTokenizer
import os

class SummarizationModel:
    def __init__(self):
        # Path to the locally downloaded mbart model
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "nts_trained_model", "mbart"))
        
        # Load tokenizer and model from local directory
        self.tokenizer = MBartTokenizer.from_pretrained(model_path)
        self.model = MBartForConditionalGeneration.from_pretrained(model_path)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
    def summarize_text(self, text: str, max_length: int = 150, num_beams: int = 4):
        self.tokenizer.src_lang = "ne_NP"
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(self.device)
        
        # Calculate dynamic bounds to target ~20% - 45% of original input size
        input_token_count = inputs["input_ids"].shape[1]
        dynamic_max_length = max(50, int(input_token_count * 0.45))
        dynamic_min_length = max(20, int(input_token_count * 0.2))
        
        # Explicitly set decoding params to Nepali
        lang_id = self.tokenizer.lang_code_to_id.get("ne_NP")
        
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=dynamic_max_length,
                min_length=dynamic_min_length,
                length_penalty=1.5,         # Strong nudge for length
                repetition_penalty=1.2,     # Math penalty for repeated tokens
                num_beams=5,                # Better search beams
                early_stopping=True,
                no_repeat_ngram_size=3,
                forced_bos_token_id=lang_id,
                decoder_start_token_id=lang_id
            )
        
        summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # CLEANING: Strip hallucinated "filler" tokens common in MBart news fine-tuning
        billboard_fillers = [
            "( भिडियोसहित )", "( भिडियो सहित )", "( फोटो फिचर )", "( भिडियो )", 
            "( सूचनासहित )", "( सूचीसहित )", "( विवरणसहित )", "हेर्नुहोस्", 
            "विज्ञप्तिसहित", "नतिजासहित", "पढ्नुहोस्"
        ]
        for filler in billboard_fillers:
            summary_text = summary_text.replace(filler, "")
            
        # Remove trailing junk like lone dots or colon artifacts
        summary_text = summary_text.strip().rstrip(":.").strip()
        
        return summary_text