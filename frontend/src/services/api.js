import React, { useState } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';

function App() {
  const [results, setResults] = useState({ accuracy: '--', f1: '--', algo: 'Aucun', image: null });
  const [prediction, setPrediction] = useState('');
  const [formData, setFormData] = useState({ age: 30, balance: 1000, duration: 200 });
  const [loading, setLoading] = useState(false);

  const handleTrain = async (algoName) => {
    setLoading(true);
    try {
      const res = await axios.post('http://127.0.0.1:5001/train', { algorithm: algoName });
      setResults({ ...res.data, image: `/matrix.png?t=${new Date().getTime()}` });
    } catch (err) { alert("Erreur de connexion au serveur (Port 5001)"); }
    setLoading(false);
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post('http://127.0.0.1:5001/predict', formData);
      setPrediction(res.data.result);
    } catch (err) { alert("Erreur: Entraînez un modèle d'abord."); }
  };

  return (
    <div className="container py-5 bg-light min-vh-100">
      <h1 className="text-center mb-5 text-dark fw-bold">🚀 Bank Marketing AI - Tâche 3</h1>

      <div className="row g-4">
        {/* SECTION 1 : ENTRAÎNEMENT */}
        <div className="col-lg-4">
          <div className="card shadow border-0 p-4 h-100">
            <h4 className="mb-4">1. Entraîner & Comparer</h4>
            <div className="d-grid gap-3">
              <button className="btn btn-outline-primary" onClick={() => handleTrain('Random Forest')}>Random Forest</button>
              <button className="btn btn-outline-info" onClick={() => handleTrain('Logistic Regression')}>Logistic Regression</button>
              <button className="btn btn-outline-success" onClick={() => handleTrain('KNN')}>KNN</button>
              <button className="btn btn-outline-warning" onClick={() => handleTrain('SVM')}>SVM</button>
            </div>
            {loading && <div className="mt-3 text-center text-primary spinner-border m-auto"></div>}
            <hr />
            <a href="http://127.0.0.1:5000" target="_blank" className="btn btn-dark btn-sm">Ouvrir MLflow Dashboard</a>
          </div>
        </div>

        {/* SECTION 2 : VISUALISATION */}
        <div className="col-lg-4">
          <div className="card shadow border-0 p-4 h-100 text-center">
            <h4 className="mb-4">2. Performance : {results.algo}</h4>
            <div className="d-flex justify-content-around mb-3">
              <div className="text-center"><h5>{results.accuracy}</h5><small>Accuracy</small></div>
              <div className="text-center"><h5>{results.f1}</h5><small>F1-Score</small></div>
            </div>
            {results.image ? <img src={results.image} className="img-fluid rounded border" alt="Matrix" /> : <p className="mt-5 text-muted">En attente de données...</p>}
          </div>
        </div>

        {/* SECTION 3 : PRÉDICTION */}
        <div className="col-lg-4">
          <div className="card shadow border-0 p-4 h-100">
            <h4 className="mb-4">3. Prédire un Client</h4>
            <form onSubmit={handlePredict}>
              <label className="form-label">Âge</label>
              <input type="number" className="form-control mb-2" value={formData.age} onChange={(e)=>setFormData({...formData, age: e.target.value})} />
              <label className="form-label">Solde (€)</label>
              <input type="number" className="form-control mb-2" value={formData.balance} onChange={(e)=>setFormData({...formData, balance: e.target.value})} />
              <label className="form-label">Durée Appel (sec)</label>
              <input type="number" className="form-control mb-3" value={formData.duration} onChange={(e)=>setFormData({...formData, duration: e.target.value})} />
              <button type="submit" className="btn btn-primary w-100">Estimer Subscription</button>
            </form>
            {prediction && <div className="alert alert-info mt-4 fw-bold text-center">{prediction}</div>}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;