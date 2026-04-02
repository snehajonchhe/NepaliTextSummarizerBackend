from flask import request, jsonify
from nts_trained_model.extractive_model import extractive_helper
import os
import uuid
 

class SummarizationController:
    def __init__(self, summarization_model, stt_model=None):
        self.summarization_model = summarization_model
        self.stt_model = stt_model
        self.temp_dir = "temp_audio"
        os.makedirs(self.temp_dir, exist_ok=True)

    def transcribe(self):
        if not self.stt_model:
            return jsonify({"error": "STT model not initialized"}), 500
        
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Save file temporarily - use generic name to allow librosa/soundfile to guess
        file_id = str(uuid.uuid4())
        temp_path = os.path.join(self.temp_dir, f"{file_id}") 
        audio_file.save(temp_path)

        try:
            transcription = self.stt_model.transcribe(temp_path)
            return jsonify({"transcription": transcription})
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Transcription error: {str(e)}")
            return jsonify({"error": f"Transcription failed: {str(e)}"}), 500
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def summarize(self):
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON payload provided"}), 400
                
            sentence = data.get("text", "")
            model_type = str(data.get("model-type", "")).lower().strip()
            
            if not sentence:
                return jsonify({"error": "No text provided"}), 400
                
            if model_type == "extractive":
                summary = extractive_helper.generate_summary(sentence)
            elif model_type == "abstractive":
                summary = self.summarization_model.summarize_text(sentence)
            else:
                # Default to abstractive or return error? Let's default to abstractive for now
                summary = self.summarization_model.summarize_text(sentence)

            # Post-processing: ensure it ends with a Nepali full stop if there is one
            last_index = summary.rfind("।")
            if last_index != -1:
                summary = summary[:last_index+1]
            
            return jsonify({
                "summary": summary,
                "model_used": model_type if model_type in ["extractive", "abstractive"] else "abstractive"
            })
        except Exception as e:
            import traceback
            traceback.print_exc() # Log the error for debugging
            return jsonify({"error": str(e)}), 500

        

        
