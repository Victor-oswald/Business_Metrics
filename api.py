from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from new_test import BusinessMetrics, BusinessDataProcessor, BusinessModel, generate_business_insights, normalize_predictions

app = Flask(__name__)
CORS(app)

# Initialize model and processor
model = BusinessModel()
processor = BusinessDataProcessor()

# Load model weights
try:
    checkpoint = torch.load('./checkpoints/model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/analyze', methods=['POST'])
def analyze_business():
    try:
        data = request.json
        metrics = BusinessMetrics(**data)
        processed_data = processor.process_input(metrics)
        
        model.eval()
        with torch.no_grad():
            raw_results = model(processed_data['temporal_data'], 
                              processed_data['features'])
        
        normalized_results = normalize_predictions(raw_results)
        insights = generate_business_insights(normalized_results, data)
        
        return jsonify({
            "status": "success",
            "data": insights
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)