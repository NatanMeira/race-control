from flask import Flask
from flask_cors import CORS
from flasgger import Swagger
import logging
from controllers.prediction_controller import PredictionController

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """Factory para criar a aplicação Flask com configuração adequada."""
    app = Flask(__name__)
    CORS(app)
    
    # Configuração do Swagger
    swagger = Swagger(app, template={
        "swagger": "2.0",
        "info": {
            "title": "F1 Pit Stop Strategy API",
            "description": "API para predição de estratégia de pit stop em corridas de Fórmula 1 - Arquitetura MVC Simplificada",
            "version": "1.0.0"
        },
        "host": "127.0.0.1:8000",
        "basePath": "/",
        "schemes": ["http"]
    })
    
    # Registrar blueprint do prediction controller
    prediction_controller = PredictionController()
    app.register_blueprint(prediction_controller.blueprint)
    
    logger.info("✅ Aplicação Flask com MVC simplificada inicializada!")
    return app

app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

if __name__ == '__main__':
    app.run(debug=True, port=8000)