import logging
import pandas as pd
from typing import Dict, Any
from utils.model_loader import get_model_instance

logger = logging.getLogger(__name__)


class PredictionService:
    """
    Serviço responsável pela lógica de negócio das predições de pit stop.
    Versão simplificada baseada no funciona.py.
    """
    
    def __init__(self):
        self._model_loader = get_model_instance()
    
    def predict_pit_stop(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Realiza predição de pit stop baseada nos dados de telemetria.
        
        Args:
            data: Dicionário com dados de telemetria
        
        Returns:
            Dicionário com resultado da predição
        """
        try:
            logger.info(f"📥 Dados recebidos: {data}")
            
            # Converter para DataFrame
            input_df = pd.DataFrame([data])
            
            # Garantir ordem das colunas (igual no funciona.py)
            expected_columns = ['TyreLife', 'LapTime_Delta', 'Cumulative_Degradation', 'Position', 'Compound']
            input_df = input_df[expected_columns]
            
            # Fazer predição
            prediction = int(self._model_loader.predict(input_df))
            logger.info(f"🧠 Predição do modelo: {prediction}")
            
            # Preparar resposta
            if prediction == 1:
                message = "BOX BOX BOX! (Chamar para o Pit Stop)"
            else:
                message = "STAY OUT! (Manter na pista)"
                
            return {
                "prediction": prediction,
                "message": message
            }
            
        except Exception as e:
            logger.error(f"🔥 Erro na predição: {str(e)}")
            raise