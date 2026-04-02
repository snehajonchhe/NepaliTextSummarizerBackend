import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, GenerationConfig
import librosa
import os

class STTModel:
    def __init__(self):
        # Path to the locally Downloaded pre-trained models (too large for Git)
        # nts_trained_model/whisper-nepali/
        # nts_trained_model/whisper-small/
        # nts_trained_model/mbart/
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "nts_trained_model"))
        checkpoint_path = os.path.join(base_path, "whisper-nepali")
        base_config_path = os.path.join(base_path, "whisper-small")
        
        # Load processor and generation config from local base large-v3 directory (100% Offline)
        self.processor = WhisperProcessor.from_pretrained(base_config_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)

        # Fix for outdated/missing generation config - Load from local large-v3 base
        self.model.generation_config = GenerationConfig.from_pretrained(base_config_path)
        self.model.generation_config.language = "nepali"
        self.model.generation_config.task = "transcribe"
        self.model.generation_config.forced_decoder_ids = None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def transcribe(self, audio_path):
        # Load audio (Whisper requires 16kHz)
        try:
            import soundfile as sf
            audio_input, sr = sf.read(audio_path)
            # Resample to 16kHz if needed
            if sr != 16000:
                audio_input = librosa.resample(audio_input.T if len(audio_input.shape) > 1 else audio_input, orig_sr=sr, target_sr=16000)
            if len(audio_input.shape) > 1:
                audio_input = audio_input.mean(axis=0)  # Convert to mono
        except Exception as e:
            print(f"SoundFile fallback: {str(e)}")
            # Fallback to librosa if soundfile fails
            audio_input, _ = librosa.load(audio_path, sr=16000)
        
        # Pre-process the audio
        input_features = self.processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to(self.device)

        # Generate transcription - explicitly force Nepali language and transcription task
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features, 
                language="nepali", 
                task="transcribe"
            )
        
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        # Post-processing: Replace English . with Nepali ।
        transcription = transcription.replace(".", "।")
        
        return transcription
