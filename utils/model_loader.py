import joblib
import os
import logging

# Configure logging
logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Responsável pelo carregamento do modelo de ML treinado.
    Implementa o padrão Singleton para garantir uma única instância do modelo.
    """
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._model is None:
            self._load_model()
    
    def _load_model(self):
        """
        Carrega o modelo treinado do arquivo pkl.
        """
        try:
            # Caminho do modelo
            model_path = 'ml/models/modelo_f1.pkl'
            
            if os.path.exists(model_path):
                self._model = joblib.load(model_path)
                print("✅ Modelo carregado com sucesso e pronto para uso!")
            else:
                raise FileNotFoundError(f"Modelo não encontrado em {model_path}")
                
        except Exception as e:
            print(f"❌ Erro crítico ao carregar o modelo: {e}")
            raise
    
    def predict(self, input_df):
        """
        Realiza predição usando o modelo carregado.
        """
        try:
            prediction = int(self._model.predict(input_df)[0])
            return prediction
        except Exception as e:
            logger.error(f"🔥 Erro na predição: {e}")
            raise
    
    def is_loaded(self) -> bool:
        """
        Verifica se o modelo está carregado.
        """
        return self._model is not None

# Função de conveniência para uso direto
def get_model_instance() -> ModelLoader:
    """
    Retorna a instância singleton do ModelLoader.
    """
    return ModelLoader()
