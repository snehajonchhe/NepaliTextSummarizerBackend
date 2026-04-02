from flask import Blueprint
from models.nts_model import SummarizationModel
from models.stt_model import STTModel
from controllers.nts_controller import SummarizationController

summarize_bp = Blueprint('summarization', __name__)

# Instantiate models and controller
summarization_model = SummarizationModel()
stt_model = STTModel()
summarization_controller = SummarizationController(summarization_model, stt_model)

@summarize_bp.route('/summarize', methods=['POST'])
def summarize():
    return summarization_controller.summarize()

@summarize_bp.route('/transcribe', methods=['POST'])
def transcribe():
    return summarization_controller.transcribe()
