from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
import os

app = Flask(__name__)

# Base de données en mémoire (pour l'historique)
historique = []
modele = None
phase_actuelle = "stable"

def entrainer_modele():
    global modele
    if len(historique) < 3:
        modele = None
        return
    df = pd.DataFrame(historique)
    features = df[['cote_H', 'cote_N', 'cote_A']].values
    knn = NearestNeighbors(n_neighbors=min(3, len(historique)), metric='euclidean')
    knn.fit(features)
    modele = knn

def predire(cote_H, cote_N, cote_A):
    if modele is None or len(historique) == 0:
        return "N", {'H':0.33, 'N':0.34, 'A':0.33}, ["1-1"], "stable"
    
    df = pd.DataFrame(historique)
    features = df[['cote_H', 'cote_N', 'cote_A']].values
    query = np.array([[cote_H, cote_N, cote_A]])
    distances, indices = modele.kneighbors(query)
    voisins = df.iloc[indices[0]]
    
    res_counts = voisins['resultat'].value_counts()
    total = res_counts.sum()
    probas = {k: v/total for k, v in res_counts.items()}
    resultat = res_counts.idxmax() if len(res_counts) > 0 else "N"
    
    scores = voisins['score'].value_counts().head(3).to_dict()
    scores_list = list(scores.keys()) if scores else ["1-1"]
    
    return resultat, probas, scores_list, phase_actuelle

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    cote_H = float(data['cote_H'])
    cote_N = float(data['cote_N'])
    cote_A = float(data['cote_A'])
    resultat, probas, scores, phase = predire(cote_H, cote_N, cote_A)
    return jsonify({
        'resultat': resultat,
        'probabilites': probas,
        'scores_probables': scores,
        'phase': phase
    })

@app.route('/feedback', methods=['POST'])
def feedback():
    global historique, modele
    data = request.get_json()
    historique.append({
        'cote_H': data['cote_H'],
        'cote_N': data['cote_N'],
        'cote_A': data['cote_A'],
        'resultat': data['resultat'],
        'score': data.get('score', '1-1'),
        'date': datetime.now().isoformat()
    })
    entrainer_modele()
    return jsonify({'status': 'ok', 'historique_len': len(historique)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
