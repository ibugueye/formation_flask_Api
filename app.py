import streamlit as st
import requests

st.title('Prédiction de Crédit pour les Clients')

# URL de l'API Flask
URL_FLASK_API = "http://localhost:5000/predict"

# Sélection de l'ID client
id_client = st.number_input('Entrez l\'ID du client pour la prédiction', value=216904, min_value=216904, max_value=999999)

# Bouton pour lancer la prédiction
if st.button('Prédire'):
    response = requests.post(URL_FLASK_API, data={'id_client': id_client})
    if response.status_code == 200:
        prediction = response.json()
        st.success(f'La prédiction pour le client {id_client} est : {prediction["prediction"]}')
    elif response.status_code == 404:
        st.error('Client non trouvé.')
    else:
        st.error('Erreur lors de la requête vers l\'API.')

# Autres éléments de l'interface utilisateur...
if st.button('Prédire'):
    response = requests.post(URL_FLASK_API, data={'id_client': id_client})
    if response.status_code == 200:
        prediction = response.json()
        st.success(f'La prédiction pour le client {id_client} est : {prediction["prediction"]}')
    else:
        st.error(f'Erreur lors de la requête vers l\'API : {response.status_code}, {response.text}')
