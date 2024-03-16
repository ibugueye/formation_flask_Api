from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Charger les modèles et les données
model_rf = pickle.load(open("model_rf.pkl", "rb"))
model_xgb = pickle.load(open("xgboost_model.pkl", "rb"))  # Modèle XGBoost
shap_explainer = pickle.load(open("shap_explainer.pkl", "rb"))  # Explainer SHAP
pipeline = pickle.load(open("pipeline.pkl", "rb"))
df = pd.read_csv("df_final.csv")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        ID = int(data['id_client'])
        
        X = df[df['SK_ID_CURR'] == ID] 

        if X.empty:
            return jsonify({'error': 'Client non trouvé'}), 404

        # Prétraitement des données avec le pipeline
        X_preprocessed = pipeline.transform(X)

        # Prédiction avec le modèle XGBoost
        prediction_xgb = model_xgb.predict(X_preprocessed)

        # Obtention des explications SHAP
        shap_values = shap_explainer.shap_values(X_preprocessed)

        return jsonify({'prediction_rf': model_rf.predict(X_preprocessed).tolist(), 
                        'prediction_xgb': prediction_xgb.tolist(),
                        'shap_values': shap_values.tolist()})
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500
    

if __name__ == '__main__':
    app.run(debug=True)
