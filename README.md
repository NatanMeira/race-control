# 🏎️ F1 Strategy API - Sistema de Previsão de Pit Stop

![Python](https://img.shields.io/badge/python-v3.11+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v3.0+-green.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-v1.5+-orange.svg)

Sistema inteligente para previsão de estratégias de pit stop em corridas de Fórmula 1, desenvolvido como MVP para a disciplina de Engenharia de Software para Sistemas Inteligentes.

## 📋 Sobre o Projeto

Este projeto implementa uma API REST com Machine Learning para auxiliar estrategistas de F1 na tomada de decisão sobre quando chamar um piloto para o pit stop. O sistema analiza dados de telemetria e recomenda se o piloto deve **BOX** (parar nos boxes) ou **STAY OUT** (manter na pista).

### 🎯 Funcionalidades Principais

- **🧠 Modelo de ML**: Pipeline com Decision Tree otimizado
- **🌐 API REST**: Endpoint de predição documentado com Swagger
- **📊 Interface Web**: Dashboard interativo com tema F1
- **🔬 Testes Automatizados**: Validação de qualidade com PyTest

## 🏗️ Arquitetura MVC Simplificada

**Implementa padrão MVC: Controller → Service → Model**

```
sprint-3/
├── 📱 app.py                       # Aplicação Flask principal (Factory Pattern)
├── 🌐 index.html                  # Interface web (frontend)
├── 📋 requirements.txt             # Dependências Python
├── 🐍 .python-version             # Versão Python recomendada
├── 📖 README.md                   # Documentação
│
├── 🎯 controllers/                 # Camada de Controle (HTTP)
│   ├── __init__.py
│   └── prediction_controller.py   # Rotas de predição
│
├── ⚡ services/                    # Camada de Negócio (Lógica)
│   ├── __init__.py
│   ├── prediction_service.py      # Lógica de predição
│   └── exceptions.py              # Exceções customizadas
│
├── 🧠 ml/                         # Machine Learning
│   ├── model_trainer.py           # Script de treinamento
│   ├── data/                      # Datasets
│   │   └── f1_strategy_dataset_v4.csv
│   └── models/                    # Modelos treinados
│       └── modelo_f1.pkl
│
├── 🛠️ utils/                      # Utilitários
│   └── model_loader.py           # Carregamento do modelo (Singleton)
│
└── 🧪 tests/                      # Testes automatizados
    ├── test_model.py              # Testes do modelo e sistema
    └── data/                      # Dados de validação
        ├── X_test_validation.csv
        └── y_test_validation.csv
```

## 🚀 Como Executar

### Pré-requisitos

- Python 3.11+
- pip (gerenciador de pacotes Python)

### 1. Clone e Configuração

```bash
# Clone o repositório
git clone https://github.com/NatanMeira/race-control.git
cd sprint-3

# Ative o ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\\Scripts\\activate   # Windows
```

### 2. Instalação de Dependências

```bash
pip install -r requirements.txt
```

### 3. Treinar o Modelo (Primeira Execução)

```bash
cd ml
python model_trainer.py
```

### 4. Executar a API

```bash
# Voltar para a raiz do projeto
cd ..

# Executar o servidor Flask
python app.py
```

A API estará disponível em:

- **API**: http://127.0.0.1:8000
- **Documentação Swagger**: http://127.0.0.1:8000/apidocs/
- **Interface Web**: Abrir `index.html` no navegador

### 5. Executar Testes

```bash
# Testes unitários
pytest tests/ -v

# Testes com cobertura
pytest tests/ --cov=utils --cov-report=html
```

## 🔧 API Endpoints

### `POST /predict`

Previsão de estratégia de pit stop

- **URL**: `http://127.0.0.1:8000/predict`
- **Body (JSON)**:

```json
{
  "TyreLife": 18,
  "LapTime_Delta": 1.25,
  "Cumulative_Degradation": 12.5,
  "Position": 4,
  "Compound": "SOFT"
}
```

- **Resposta**:

```json
{
  "prediction": 1,
  "message": "BOX BOX BOX! (Chamar para o Pit Stop)"
}
```

### Documentação Swagger

- **URL**: `http://127.0.0.1:8000/apidocs/`

## 📊 Dados e Modelo

### Features Utilizadas

- **TyreLife**: Voltas completadas com o pneu atual
- **LapTime_Delta**: Diferença de tempo vs volta ideal (segundos)
- **Cumulative_Degradation**: Índice de degradação acumulada
- **Position**: Posição atual na corrida
- **Compound**: Composto do pneu (SOFT, MEDIUM, HARD, etc.)

### Modelo Implementado

- **Decision Tree Classifier** (otimizada com GridSearchCV)
- **Pipeline completo** com pré-processamento (StandardScaler, OneHotEncoder)

### Métricas de Avaliação

- **Recall**: ≥ 60% (detecção de situações de pit stop)
- **Validação**: Holdout com 20% dos dados
- **Cross-validation**: 5-fold para otimização

## 🎨 Interface Web

Dashboard interativo com tema F1 para entrada de dados de telemetria e visualização das recomendações de estratégia.

## 🧪 Execução dos Testes

```bash
# Todos os testes
pytest tests/ -v

# Teste específico do modelo
pytest tests/test_model.py::test_modelo_pipeline_f1 -v
```

## 🚀 Deploy

Para executar em produção:

```bash
# Usar gunicorn ao invés do servidor de desenvolvimento
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

---

<div align="center">

**🏁 "That's lights out and away we go!" 🏁**

Desenvolvido com ❤️ para a comunidade F1 e IA

</div>
