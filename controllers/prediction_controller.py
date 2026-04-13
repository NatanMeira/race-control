import logging
from flask import request, jsonify, Blueprint
from flasgger import swag_from
from services.prediction_service import PredictionService

logger = logging.getLogger(__name__)


class PredictionController:
    """
    Controller responsável pelas rotas de predição de pit stop.
    Versão simplificada mantendo estrutura MVC.
    """
    
    def __init__(self):
        self.prediction_service = PredictionService()
        self.blueprint = Blueprint('prediction', __name__)
        self._register_routes()
    
    def _register_routes(self):
        """
        Registra as rotas do controller.
        """
        self.blueprint.add_url_rule('/predict', 'predict_pit_stop', self.predict_pit_stop, methods=['POST'])
    
    @swag_from({
        'tags': ['Strategy Prediction'],
        'description': 'Prediz se o piloto deve parar nos boxes com base nos dados da telemetria',
        'parameters': [
            {
                'name': 'body',
                'in': 'body',
                'required': True,
                'schema': {
                    'type': 'object',
                    'properties': {
                        'TyreLife': {
                            'type': 'number',
                            'description': 'Voltas completadas com o pneu atual',
                            'example': 18
                        },
                        'LapTime_Delta': {
                            'type': 'number', 
                            'description': 'Diferença de tempo em relação à volta ideal (segundos)',
                            'example': 1.25
                        },
                        'Cumulative_Degradation': {
                            'type': 'number',
                            'description': 'Índice de degradação acumulada do pneu',
                            'example': 12.5
                        },
                        'Position': {
                            'type': 'integer',
                            'description': 'Posição atual na corrida',
                            'example': 4
                        },
                        'Compound': {
                            'type': 'string',
                            'description': 'Composto do pneu',
                            'enum': ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'],
                            'example': 'SOFT'
                        }
                    },
                    'required': ['TyreLife', 'LapTime_Delta', 'Cumulative_Degradation', 'Position', 'Compound']
                }
            }
        ],
        'responses': {
            '200': {
                'description': 'Predição realizada com sucesso',
                'examples': {
                    'application/json': {
                        'success': True,
                        'data': {
                            'prediction': 1,
                            'message': 'BOX BOX BOX! (Chamar para o Pit Stop)',
                            'confidence': 'HIGH',
                            'telemetry_summary': {
                                'tyre_state': 'CRITICAL',
                                'performance_state': 'DECLINING'
                            }
                        }
                    }
                }
            },
            '400': {
                'description': 'Dados de entrada inválidos',
                'examples': {
                    'application/json': {
                        'success': False,
                        'error': 'Campos obrigatórios ausentes: [TyreLife]'
                    }
                }
            },
            '503': {
                'description': 'Modelo não disponível',
                'examples': {
                    'application/json': {
                        'success': False,
                        'error': 'Modelo de ML não está disponível'
                    }
                }
            },
            '500': {
                'description': 'Erro interno do servidor'
            }
        }
    })
    def predict_pit_stop(self):
        """
        Endpoint para predição de estratégia de pit stop.
        Versão simplificada mantendo estrutura MVC.
        
        Returns:
            JSON: Resposta com predição e mensagem
        """
        try:
            # Obter dados do request
            data = request.get_json()
            
            if not data:
                logger.warning("⚠️ Requisição sem dados JSON")
                return jsonify({"error": "Dados JSON obrigatórios"}), 400
            
            logger.info(f"📲 Requisição de predição recebida: {data}")
            
            # Chamar serviço de predição
            result = self.prediction_service.predict_pit_stop(data)
            
            logger.info(f"✅ Predição concluída: {result['prediction']}")
            return jsonify(result), 200
            
        except Exception as e:
            logger.error(f"🔥 Erro inesperado: {str(e)}")
            return jsonify({"error": str(e)}), 500