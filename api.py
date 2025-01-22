from flask import Flask, request, jsonify
from dataclasses import dataclass
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app=app)
from new_test import BusinessDataProcessor, BusinessMetrics, BusinessModel, BusinessPredictor, generate_business_insights
# Import all the required classes from your original code
# (BusinessMetrics, BusinessModel, BusinessDataProcessor, etc. should be imported here)

@dataclass
class BusinessMetrics:
    monthly_revenue: List[Dict]
    growth_rate: Dict
    monthly_expenses: Dict
    customer_acquisition_cost: Dict
    customer_lifetime_value: Dict
    market_size: int
    potential_risks: List[str]
    potential_mitigants: List[str]

# Initialize the predictor globally
predictor = None

def initialize_predictor():
    global predictor
    if predictor is None:
        model_path = os.getenv('MODEL_PATH', 'checkpoints/model.pt')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        predictor = BusinessPredictor(model_path, device)



@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        initialize_predictor()

        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = [
            'monthly_revenue', 'growth_rate', 'monthly_expenses',
            'customer_acquisition_cost', 'customer_lifetime_value',
            'market_size', 'potential_risks', 'potential_mitigants'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}'
                }), 400

        # Convert JSON data to BusinessMetrics object
        business_metrics = BusinessMetrics(
            monthly_revenue=data['monthly_revenue'],
            growth_rate=data['growth_rate'],
            monthly_expenses=data['monthly_expenses'],
            customer_acquisition_cost=data['customer_acquisition_cost'],
            customer_lifetime_value=data['customer_lifetime_value'],
            market_size=data['market_size'],
            potential_risks=data['potential_risks'],
            potential_mitigants=data['potential_mitigants']
        )

        # Generate predictions
        insights = predictor.predict(business_metrics)
        
        return jsonify({
            'status': 'success',
            'insights': insights
        }), 200

    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/api/sample', methods=['GET'])
def get_sample_data():
    """Return sample data structure for testing"""
    sample_data = {
        "monthly_revenue": [
            {"revenue": 1000000, "customer_turnover_rate": 15},
            {"revenue": 1100000, "customer_turnover_rate": 14},
            {"revenue": 1200000, "customer_turnover_rate": 13}
        ],
        "growth_rate": {"value": 0.1, "duration": 12},
        "monthly_expenses": {"value": 800000, "duration": 12},
        "customer_acquisition_cost": {"value": 500, "duration": 12},
        "customer_lifetime_value": {"value": 2000, "duration": 24},
        "market_size": 10000000,
        "potential_risks": [
            "Economic downturns (Market & Economic Risks)",
            "Operational inefficiencies (Operational Risks)",
            "Supply chain disruptions (Supply Risks)"
        ],
        "potential_mitigants": [
            "Diversifying revenue streams (revenue growth)",
            "Implementing automated systems (operational)",
            "Building customer loyalty program (customer)"
        ]
    }
    return jsonify(sample_data), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)