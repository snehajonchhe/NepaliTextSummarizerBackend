from flask import Blueprint
from models.nts_model import SummarizationModel
from controllers.nts_controller import SummarizationController

summarize_bp = Blueprint('summarization', __name__)

# Instantiate model and controller
summarization_model = SummarizationModel()
summarization_controller = SummarizationController(summarization_model)

@summarize_bp.route('/summarize', methods=['POST'])
def summarize():
    print("route")
    return summarization_controller.summarize()
    
