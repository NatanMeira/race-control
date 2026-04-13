import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# 1. Carga de Dados
print("Carregando o dataset...")
data_path = os.path.join('data', 'f1_strategy_dataset_v4.csv')
df = pd.read_csv(data_path)

# 2. Definição de Features (X) e Target (y)
# Escolhemos um mix perfeito de variáveis para o negócio e para o modelo
features_numericas = ['TyreLife', 'LapTime_Delta', 'Cumulative_Degradation', 'Position']
features_categoricas = ['Compound'] # Tipo de pneu (SOFT, MEDIUM, HARD, etc.)

X = df[features_numericas + features_categoricas]
y = df['PitNextLap']

# 3. Separação (Holdout) com estratificação
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Guardar lote de validação para o PyTest na Fase 4
# Criar diretório tests/data se não existir
test_data_dir = '../tests/data'
os.makedirs(test_data_dir, exist_ok=True)

X_test.to_csv(os.path.join(test_data_dir, 'X_test_validation.csv'), index=False)
y_test.to_csv(os.path.join(test_data_dir, 'y_test_validation.csv'), index=False)

# 4. Criação do Transformador (Normalização para números, Encoding para categorias)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features_numericas),
        ('cat', OneHotEncoder(handle_unknown='ignore'), features_categoricas)
    ])

# 5. Construção dos Pipelines com os 4 Algoritmos exigidos
pipelines = {
    'KNN': Pipeline([('preprocessor', preprocessor), ('classifier', KNeighborsClassifier())]),
    'DecisionTree': Pipeline([('preprocessor', preprocessor), ('classifier', DecisionTreeClassifier(class_weight='balanced', random_state=42))]),
    # Nota: GaussianNB puro não suporta matrizes esparsas do OneHotEncoder, por isso usamos uma conversão densa ou configuramos o OHE para dense.
    # Para simplificar e evitar erros no pipeline, passaremos o OneHotEncoder com sparse_output=False (se usar scikit-learn >= 1.2)
}

# Ajuste fino para o Naive Bayes funcionar com o preprocessor
preprocessor_nb = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features_numericas),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_categoricas)
    ])

pipelines['NaiveBayes'] = Pipeline([('preprocessor', preprocessor_nb), ('classifier', GaussianNB())])
pipelines['SVM'] = Pipeline([('preprocessor', preprocessor), ('classifier', SVC(class_weight='balanced', random_state=42))])

# 6. Treino e Avaliação
print("\n--- Avaliação Inicial dos Modelos ---")
for name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(f"\n{name}:")
    print(classification_report(y_test, y_pred))

# 7. Otimização de Hiperparâmetros (Árvore de Decisão)
print("\n--- Otimizando a Árvore de Decisão (GridSearchCV) ---")
param_grid = {
    'classifier__max_depth': [3, 5, 10, None],
    'classifier__min_samples_split': [2, 5, 10]
}

# Otimizando para 'recall' porque prever o Pit Stop (Classe 1) é crítico
grid_search = GridSearchCV(pipelines['DecisionTree'], param_grid, cv=5, scoring='recall')
grid_search.fit(X_train, y_train)

melhor_modelo = grid_search.best_estimator_
print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")

# 8. Exportação do Modelo
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, 'modelo_f1.pkl')

joblib.dump(melhor_modelo, model_path)
print(f"\n[OK] Pipeline completo salvo como '{model_path}'")