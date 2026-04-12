import joblib
import pandas as pd
from sklearn.metrics import recall_score
import os

def test_modelo_pipeline_f1():
    # 1. Verifica se os arquivos exportados pelo Colab existem
    assert os.path.exists('modelo_f1.pkl'), "Modelo não encontrado!"
    assert os.path.exists('X_test_validation.csv'), "Features de teste não encontradas!"
    assert os.path.exists('y_test_validation.csv'), "Target de teste não encontrado!"

    # 2. Carrega artefatos
    model = joblib.load('modelo_f1.pkl')
    X_test = pd.read_csv('X_test_validation.csv')
    y_test = pd.read_csv('y_test_validation.csv').squeeze()

    # 3. Predição usando o Pipeline completo (aplica StandardScaler e OneHotEncoder implicitamente)
    y_pred = model.predict(X_test)

    # 4. Avaliação pelo Recall (Capacidade de acerto na classe minoritária "Pit Stop = 1")
    recall = recall_score(y_test, y_pred, zero_division=0)

    # 5. O Teste do Requisito de Negócio
    # Exigimos que o modelo consiga detectar ao menos 60% das situações reais de parada
    threshold = 0.60
    assert recall >= threshold, f"Falha de Qualidade: Recall do modelo ({recall:.2f}) está abaixo do mínimo exigido de {threshold}."