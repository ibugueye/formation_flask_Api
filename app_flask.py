from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Charger le modèle et le pipeline
model = pickle.load(open("model_rf.pkl", "rb"))
pipeline = pickle.load(open("pipeline.pkl", "rb"))
df = pd.read_csv("test_preprocess_sample.csv")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    ID = data['id_client']
    seuil = 0.625
    response = {}

    if ID not in df['SK_ID_CURR'].unique():
        response['prediction'] = "Ce client n'est pas répertorié"
    else:
        # Sélectionner les données pour l'ID spécifié et préparer les données avec le pipeline
        X = df[df['SK_ID_CURR'] == ID].drop(['SK_ID_CURR'], axis=1)
        X_transformed = pipeline.transform(X)  # Assurez-vous que votre pipeline fait le prétraitement nécessaire
        probability_default_payment = model.predict_proba(X_transformed)[:, 1][0]
        prediction = "Prêt NON Accordé, risque de défaut" if probability_default_payment >= seuil else "Prêt Accordé"
        response['prediction'] = prediction
        response['probability'] = float(probability_default_payment)

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
