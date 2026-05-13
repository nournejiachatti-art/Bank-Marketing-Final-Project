import React, { useState } from 'react';
import { predictCustomer } from '../services/api';

const PredictForm = () => {
    const [formData, setFormData] = useState({ balance: '', duration: '' });
    const [result, setResult] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        const res = await predictCustomer(formData);
        setResult(res);
    };

    return (
        <div className="card p-4 mt-3">
            <h3>🎯 Test de Prédiction</h3>
            <form onSubmit={handleSubmit}>
                <input 
                    type="number" 
                    placeholder="Solde (Balance)" 
                    className="form-control mb-2"
                    onChange={(e) => setFormData({...formData, balance: e.target.value})}
                />
                <input 
                    type="number" 
                    placeholder="Durée (Duration)" 
                    className="form-control mb-2"
                    onChange={(e) => setFormData({...formData, duration: e.target.value})}
                />
                <button className="btn btn-primary w-100">Prédire</button>
            </form>
            {result && <div className="mt-3 alert alert-info">{result.prediction}</div>}
        </div>
    );
};

export default PredictForm;