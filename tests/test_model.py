import sys
import os
import pandas as pd
from sklearn.metrics import recall_score

# Adicionar o diretório pai ao path para importar utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_loader import get_model_instance

def test_modelo_pipeline_f1():
    """
    Teste automatizado para validar o desempenho do modelo de F1 Strategy.
    Conforme requisitos do PDF: teste usando PyTest com métricas adequadas.
    """
    # 1. Verifica se os arquivos de validação existem
    test_data_dir = os.path.join(os.path.dirname(__file__), 'data')
    x_test_path = os.path.join(test_data_dir, 'X_test_validation.csv') 
    y_test_path = os.path.join(test_data_dir, 'y_test_validation.csv')
    
    assert os.path.exists(x_test_path), f"Features de teste não encontradas em {x_test_path}!"
    assert os.path.exists(y_test_path), f"Target de teste não encontrado em {y_test_path}!"

    # 2. Carrega o modelo usando o model_loader
    model_loader = get_model_instance()
    assert model_loader.is_loaded(), "Modelo não foi carregado corretamente!"

    # 3. Carrega dados de validação
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()

    # 4. Realiza predições usando o pipeline completo 
    y_pred = []
    for _, row in X_test.iterrows():
        prediction = model_loader.predict(pd.DataFrame([row]))
        y_pred.append(prediction)

    # 5. Avaliação pelo Recall (Capacidade de detectar situações de Pit Stop)
    recall = recall_score(y_test, y_pred, zero_division=0)

    # 6. Requisito de Negócio: modelo deve detectar ao menos 60% dos pit stops reais
    threshold = 0.60
    assert recall >= threshold, f"❌ Falha de Qualidade: Recall do modelo ({recall:.2f}) está abaixo do mínimo exigido de {threshold}."
    
    print(f"✅ Teste passou! Recall: {recall:.2f} (>= {threshold})")

def test_model_loader_singleton():
    """
    Testa se o ModelLoader implementa corretamente o padrão Singleton
    """
    loader1 = get_model_instance()
    loader2 = get_model_instance()
    
    # Devem ser a mesma instância
    assert loader1 is loader2, "ModelLoader não está implementando Singleton corretamente"
    
    print("✅ Singleton test passou!")

def test_prediction_format():
    """
    Testa se as predições retornam no formato correto
    """
    model_loader = get_model_instance()
    
    # Dados de teste simulados
    test_data = pd.DataFrame([{
        'TyreLife': 20,
        'LapTime_Delta': 1.5,
        'Cumulative_Degradation': 10.0,
        'Position': 3,
        'Compound': 'SOFT'
    }])
    
    prediction = model_loader.predict(test_data)
    
    # Deve retornar 0 ou 1
    assert prediction in [0, 1], f"Predição deve ser 0 ou 1, recebido: {prediction}"
    assert isinstance(prediction, int), f"Predição deve ser int, recebido: {type(prediction)}"
    
    print(f"✅ Format test passou! Predição: {prediction}")