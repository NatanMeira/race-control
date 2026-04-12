from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Carregar o pipeline treinado
try:
    model = joblib.load('modelo_f1.pkl')
    print("✅ Modelo carregado com sucesso e pronto para uso!")
except Exception as e:
    print(f"❌ Erro crítico ao carregar o modelo: {e}")

@app.route('/predict', methods=['POST'])
def predict_pit_stop():
    try:
        data = request.get_json()
        print(f"📥 Dados recebidos do front-end: {data}") # Para vermos o que está a chegar
        
        # 1. Converter para DataFrame
        input_df = pd.DataFrame([data])
        
        # 2. GARANTIR a exata mesma ordem das colunas do Colab! (Isto resolve 90% dos erros 500)
        colunas_ordem_correta = ['TyreLife', 'LapTime_Delta', 'Cumulative_Degradation', 'Position', 'Compound']
        input_df = input_df[colunas_ordem_correta]
        
        # 3. Fazer a predição
        prediction = int(model.predict(input_df)[0])
        print(f"🧠 Predição do modelo: {prediction}")
        
        if prediction == 1:
            return jsonify({"prediction": 1, "message": "BOX BOX BOX! (Chamar para o Pit Stop)"})
        else:
            return jsonify({"prediction": 0, "message": "STAY OUT! (Manter na pista)"})
            
    except Exception as e:
        # ISTO VAI MOSTRAR O ERRO REAL NO SEU TERMINAL
        print(f"🔥 ERRO FATAL NA ROTA /predict: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)