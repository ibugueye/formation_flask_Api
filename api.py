from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Charger le modèle, le pipeline et les données
model = pickle.load(open("model_rf.pkl", "rb"))
pipeline = pickle.load(open("pipeline.pkl", "rb"))
shap_explainer = pickle.load(open("shap_explainer.pkl", "rb"))
df = pd.read_csv("df_final.csv")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        ID = int(request.form['id_client'])
        X = df[df['SK_ID_CURR'] == ID]

        if X.empty:
            return jsonify({'error': 'Client non trouvé'}), 404

        # Prétraitement des données avec le pipeline
        X_preprocessed = pipeline.transform(X)

        # Prédiction avec le modèle
        prediction = model.predict(X_preprocessed)
        prediction_proba = model.predict_proba(X_preprocessed)

        # Calcul de l'interprétabilité locale avec SHAP
        shap_values = shap_explainer.shap_values(X_preprocessed)

        # Réponse avec probabilités, décision et explication SHAP
        response = {
            'prediction': prediction[0],
            'probability': prediction_proba[0].tolist(),
            'shap_values': shap_values[0].tolist()  # Assurez-vous que c'est le format correct
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
