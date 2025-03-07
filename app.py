from flask import Flask
from flask_cors import CORS
from routes.nts_route import summarize_bp

app = Flask(__name__)
CORS(app)
# Register Blueprints
app.register_blueprint(summarize_bp, url_prefix='/api')

if __name__ == '__main__':
    app.run(debug=True)
