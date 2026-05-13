import React, { useState, useEffect } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';

function App() {
  const [results, setResults] = useState({ 
    accuracy: '--', 
    accuracy_train: '--', 
    f1: '--', 
    precision: '--', 
    recall: '--', 
    algo: 'Aucun', 
    image: null, 
    show_importance: false,
    error_samples: []
  });
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [stabilityData, setStabilityData] = useState([]);
  
  // États pour les hyperparamètres
  const [maxDepth, setMaxDepth] = useState(10);
  const [neighbors, setNeighbors] = useState(5);
  const [inputs, setInputs] = useState({ age: 35, balance: 1500, duration: 400 });
  const [xgParams, setXgParams] = useState({ n_estimators: 100, max_depth: 6, learning_rate: 0.1 });
  const [adaParams, setAdaParams] = useState({ n_estimators: 50, learning_rate: 1.0 });

  const errorSamples = results.error_samples || [];

  useEffect(() => {
    const saved = localStorage.getItem('lastResult');
    if (saved) setResults(JSON.parse(saved));
  }, []);

  // Fonction d'entraînement corrigée
  const handleTrain = async (algoName, params = {}) => {
    setLoading(true);
    try {
      const payload = { ...params };

      if (algoName === 'Random Forest') {
        payload.max_depth = Number(params.max_depth ?? maxDepth);
      }
      if (algoName === 'KNN') {
        payload.n_neighbors = Number(neighbors);
      }
      if (algoName === 'XGBoost') {
        payload.n_estimators = Number(params.n_estimators);
        payload.max_depth = Number(params.max_depth);
        payload.learning_rate = Number(params.learning_rate);
      }
      if (algoName === 'AdaBoost') {
        payload.n_estimators = Number(params.n_estimators);
        payload.learning_rate = Number(params.learning_rate);
      }

      const res = await axios.post('http://127.0.0.1:5001/train', {
        algorithm: algoName,
        params: payload
      });

      console.log('Response data:', res.data);

      const timestamp = new Date().getTime();
      const data = {
        ...res.data,
        image: `http://127.0.0.1:5001/matrix.png?t=${timestamp}`
      };

      setResults(data);
      localStorage.setItem('lastResult', JSON.stringify(data));
    } catch (err) {
      alert("Erreur : Vérifiez que le serveur Flask est lancé sur le port 5001.");
    }
    setLoading(false);
  };

  // Fonction pour lancer l'étude de stabilité
  const handleStabilityTest = async () => {
    setLoading(true);
    try {
      const res = await axios.post('http://127.0.0.1:5001/stability-test');
      setStabilityData(res.data.data);
      alert("✅ Étude de stabilité terminée ! Consultez MLflow pour la comparaison.");
    } catch (err) {
      alert("Erreur lors du test de stabilité.");
    }
    setLoading(false);
  };

  return (
    <div className="container py-4 bg-light min-vh-100">
      <div className="p-5 mb-4 bg-primary text-white rounded-3 shadow">
        <h1 className="display-5 fw-bold">Bank AI : Dashboard Multi-Tâches</h1>
        <p className="col-md-8 fs-4">Comparaison d'algorithmes (T3) et Analyse de robustesse RF (T4).</p>
      </div>

      <div className="row g-4">
        {/* COLONNE DE CONTRÔLE */}
        <div className="col-md-4">
          <div className="card shadow border-0 mb-4">
            <div className="card-header bg-dark text-white fw-bold">🛠️ Configuration des Modèles</div>
            <div className="card-body">
              
              {/* TÂCHE 3 : ALGORITHMES DE BASE */}
              <h6 className="text-primary fw-bold">Tâche 3 : Comparaison Baseline</h6>
              <div className="d-grid gap-2 mb-4">
                <button className="btn btn-outline-secondary btn-sm" onClick={() => handleTrain('Logistic Regression')}>Régression Logistique</button>
                <button className="btn btn-outline-secondary btn-sm" onClick={() => handleTrain('SVM')}>SVM (Linear)</button>
                <div className="border p-2 rounded">
                  <label className="small">K-Neighbors (K={neighbors})</label>
                  <input type="range" className="form-range" min="1" max="20" value={neighbors} onChange={(e)=>setNeighbors(Number(e.target.value))} />
                  <button className="btn btn-outline-info btn-sm w-100" onClick={() => handleTrain('KNN')}>Lancer KNN</button>
                </div>
              </div>

              <div className="row g-3 mb-4">
                {/* BLOC XGBOOST */}
                <div className="col-md-6">
                  <div className="card shadow-sm border-0 h-100" style={{ borderTop: '5px solid #ffc107' }}>
                    <div className="card-body">
                      <h5 className="fw-bold text-warning">🚀 XGBoost Config</h5>
                      <div className="mb-2">
                        <label className="small">Arbres: {xgParams.n_estimators}</label>
                        <input type="range" className="form-range" min="50" max="500" value={xgParams.n_estimators}
                          onChange={(e) => setXgParams({...xgParams, n_estimators: Number(e.target.value)})} />
                      </div>
                      <div className="row">
                        <div className="col-6">
                          <label className="small">Prof. Max</label>
                          <input type="number" className="form-control form-control-sm" value={xgParams.max_depth}
                            onChange={(e) => setXgParams({...xgParams, max_depth: parseInt(e.target.value) || 6})} />
                        </div>
                        <div className="col-6">
                          <label className="small">Learning Rate</label>
                          <input type="number" className="form-control form-control-sm" step="0.01" value={xgParams.learning_rate}
                            onChange={(e) => setXgParams({...xgParams, learning_rate: parseFloat(e.target.value) || 0.1})} />
                        </div>
                      </div>
                      <button className="btn btn-warning btn-sm w-100 mt-3 fw-bold"
                        onClick={() => handleTrain('XGBoost', xgParams)}>Entraîner XGBoost</button>
                    </div>
                  </div>
                </div>

                {/* BLOC ADABOOST */}
                <div className="col-md-6">
                  <div className="card shadow-sm border-0 h-100" style={{ borderTop: '5px solid #dc3545' }}>
                    <div className="card-body">
                      <h5 className="fw-bold text-danger">⚡ AdaBoost Config</h5>
                      <div className="mb-2">
                        <label className="small">Nombre d'estimateurs: {adaParams.n_estimators}</label>
                        <input type="range" className="form-range" min="10" max="200" value={adaParams.n_estimators}
                          onChange={(e) => setAdaParams({...adaParams, n_estimators: parseInt(e.target.value) || 50})} />
                      </div>
                      <div className="mb-2">
                        <label className="small">Learning Rate (Poids)</label>
                        <input type="number" className="form-control form-control-sm" step="0.1" value={adaParams.learning_rate}
                          onChange={(e) => setAdaParams({...adaParams, learning_rate: parseFloat(e.target.value) || 1.0})} />
                      </div>
                      <button className="btn btn-outline-danger btn-sm w-100 mt-3"
                        onClick={() => handleTrain('AdaBoost', adaParams)}>Entraîner AdaBoost</button>
                    </div>
                  </div>
                </div>
              </div>

              <hr />

              {/* TÂCHE 4 : RANDOM FOREST PROFOND */}
              <h6 className="text-danger fw-bold">Tâche 4 : Analyse Random Forest</h6>
              <label className="form-label small">Profondeur choisie : <strong>{maxDepth}</strong></label>
              <input 
                type="number" 
                className="form-control mb-3" 
                min="1" 
                max="50" 
                value={maxDepth} 
                onChange={(e) => setMaxDepth(Number(e.target.value))} 
              />
              
              <div className="d-grid gap-2">
                <button className="btn btn-primary fw-bold" onClick={() => handleTrain('Random Forest')}>
                   Lancer RF (Profondeur {maxDepth})
                </button>
                
                <div className="mt-2 d-flex justify-content-between">
                    <button className="btn btn-warning btn-sm" onClick={() => handleTrain('Random Forest', 2)}>Depth 2 (Under)</button>
                    <button className="btn btn-danger btn-sm" onClick={() => handleTrain('Random Forest', 50)}>Depth 50 (Over)</button>
                </div>

                <button className="btn btn-dark w-100 mt-3 fw-bold" onClick={handleStabilityTest}>
                    🛡️ Lancer Étude de Stabilité (5 Seeds)
                </button>

                {stabilityData.length > 0 && (
                    <div className="mt-3 p-2 bg-white border rounded small">
                        <h6 className="fw-bold" style={{fontSize: '0.8rem'}}>Résultats de Robustesse :</h6>
                        {stabilityData.map((d, i) => (
                            <div key={i} className="d-flex justify-content-between">
                                <span>Seed {d.seed}:</span> <strong>{d.accuracy}%</strong>
                            </div>
                        ))}
                    </div>
                )}
              </div>
              {loading && <div className="text-center mt-3 spinner-border text-primary"></div>}
            </div>
          </div>

          {/* PRÉDICTION */}
          <div className="card shadow border-0">
            <div className="card-header bg-success text-white fw-bold">🔮 Test de Prédiction</div>
            <div className="card-body">
              <input type="number" className="form-control mb-2" placeholder="Age" value={inputs.age} onChange={(e)=>setInputs({...inputs, age: e.target.value})} />
              <input type="number" className="form-control mb-2" placeholder="Solde" value={inputs.balance} onChange={(e)=>setInputs({...inputs, balance: e.target.value})} />
              <input type="number" className="form-control mb-2" placeholder="Durée" value={inputs.duration} onChange={(e)=>setInputs({...inputs, duration: e.target.value})} />
              <button className="btn btn-success w-100 fw-bold" onClick={async () => {
                const res = await axios.post('http://127.0.0.1:5001/predict', inputs);
                setPrediction(res.data.result);
              }}>Prédire</button>
              {prediction && <div className="mt-3 p-2 bg-white border border-success rounded text-center fw-bold small">{prediction}</div>}
            </div>
          </div>
        </div>

        {/* COLONNE DES RÉSULTATS */}
        <div className="col-md-8">
          <div className="card shadow border-0 h-100">
            <div className="card-header bg-white d-flex justify-content-between align-items-center">
              <span className="fw-bold">📊 Résultats : {results.algo}</span>
              <a href="http://127.0.0.1:5000" target="_blank" rel="noreferrer" className="btn btn-sm btn-outline-dark">Ouvrir MLflow</a>
            </div>
            <div className="card-body">
              {/* METRIQUES */}
              <div className="row text-center g-2 mb-4">
                <div className="col-3"><div className="p-2 border rounded bg-light"><h6>Acc. Test</h6><span className="h5 text-primary">{results.accuracy}</span></div></div>
                <div className="col-3"><div className="p-2 border rounded bg-light"><h6>Acc. Train</h6><span className="h5 text-secondary">{results.accuracy_train}</span></div></div>
                <div className="col-3"><div className="p-2 border rounded bg-light"><h6>F1-Score</h6><span className="h5 text-success">{results.f1}</span></div></div>
                <div className="col-3"><div className="p-2 border rounded bg-light"><h6>Recall</h6><span className="h5 text-warning">{results.recall}</span></div></div>
              </div>
              
              <div className="row">
                <div className="col-md-6 text-center">
                  <p className="small fw-bold">Matrice de Confusion</p>
                  {results.image && <img src={results.image} className="img-fluid border rounded shadow-sm" alt="Matrix" />}
                </div>
                
                <div className="col-md-6 text-center">
                  <p className="small fw-bold">Analyse des Features (RF)</p>
                  {results.show_importance ? (
                    <img src={`http://127.0.0.1:5001/importance.png?t=${new Date().getTime()}`} className="img-fluid border rounded shadow-sm" alt="Importance" />
                  ) : (
                    <div className="py-5 text-muted border rounded bg-light small">L'importance des variables s'affichera ici pour le Random Forest.</div>
                  )}
                </div>
              </div>
              
              {/* MESSAGE D'AIDE POUR LE RAPPORT */}
              <div className="mt-4 alert alert-secondary small">
                <strong>💡 Aide Tâche 4 :</strong> Si l'écart entre <i>Acc. Train</i> et <i>Acc. Test</i> est grand, vous observez de la <strong>variance</strong> (Overfitting).
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* SECTION 3 : ANALYSE DES ERREURS (NOUVEAU) */}
      <div className="row g-4 mt-4">
        <div className="col-12">
          <div className="card shadow border-0 p-4 h-100">
            <h4 className="text-danger">3. Analyse Qualitative (Erreurs)</h4>
            <p className="text-muted small">Voici quelques profils réels où le modèle s'est trompé (données originales du dataset) :</p>
            
            {errorSamples.length > 0 ? (
              <div className="table-responsive">
                <table className="table table-sm table-hover mt-2">
                  <thead className="table-dark">
                    <tr>
                      <th>Age</th>
                      <th>Durée</th>
                      <th>Solde</th>
                      <th>Réel</th>
                      <th>Pred</th>
                    </tr>
                  </thead>
                  <tbody>
                    {errorSamples.map((sample, idx) => (
                      <tr key={idx} className={sample.vrai_y === 1 ? "table-warning" : "table-danger"}>
                        <td>{sample.age}</td>
                        <td>{sample.duration}s</td> 
                        <td>{sample.balance}€</td>
                        <td><span className="badge bg-success">{sample.vrai_y === 1 ? 'OUI' : 'NON'}</span></td>
                        <td><span className="badge bg-danger">{sample.pred_y === 1 ? 'OUI' : 'NON'}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="alert alert-secondary small p-2">
                  <strong>Pattern :</strong> Les lignes jaunes sont des <strong>Faux Négatifs</strong> (clients qui ont souscrit mais que l'IA n'a pas détectés). Les lignes rouges sont des <strong>Faux Positifs</strong> (l'IA a prédit une souscription mais c'était incorrect). Notez les profils des seniors (60+ ans) : le dataset Bank Marketing contient peu de seniors, ce qui explique les erreurs de prédiction sur ces segments.
                </div>
              </div>
            ) : (
              <p className="text-muted">Aucune erreur à afficher. Entraînez un modèle Random Forest pour voir l'analyse.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;