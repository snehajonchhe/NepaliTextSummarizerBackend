from flask import request, jsonify
from nts_trained_model.extractive_model import extractive_helper
 

class SummarizationController:
    def __init__(self, model):
        self.model = model

    def summarize(self):
        data = request.get_json()
        sentence = data.get("text", "")
        model_type = data.get("model-type", "")
        
        if not sentence:
            return jsonify({"error": "No text provided"}), 400
        try:
            if model_type == "extractive":
                summary = extractive_helper.generate_summary(sentence)
            elif model_type == "abstractive":
                summary = self.model.summarize_text(sentence)

            last_index = summary.rfind("।")

            return jsonify({"summary": summary[:last_index+1]})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        

        
