from flask import Flask, jsonify, request
import pandas as pd
from datetime import datetime
import json
import logging
import traceback
from typing import Dict, Any
from Test import ImprovedBusinessDataProcessor, TimeSeriesBusinessPredictor
from flask_cors import CORS

# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure CORS properly for all routes
CORS(app, resources={
    r"/*": {
        "origins": ["http://127.0.0.1:5500", "http://localhost:5500"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

logger.info("Initializing business data processor and predictor...")
try:
    # Initialize the predictor globally
    processor = ImprovedBusinessDataProcessor()
    predictor = TimeSeriesBusinessPredictor('improved_business_predictor.pth', processor)
    logger.info("Initialization successful")
except Exception as e:
    logger.error(f"Error initializing predictor: {str(e)}")
    logger.error(traceback.format_exc())
    raise

def format_date(date_obj):
    """Helper function to format datetime objects for JSON serialization"""
    try:
        if isinstance(date_obj, pd.Timestamp):
            return date_obj.strftime('%Y-%m-%d')
        return str(date_obj)
    except Exception as e:
        logger.error(f"Error formatting date: {str(e)}")
        raise

def process_predictions(predictions: Dict[str, Any], confidence_metrics: list) -> dict:
    """Process predictions and confidence metrics into JSON-serializable format"""
    try:
        logger.debug("Processing predictions and confidence metrics")
        processed_data = {
            'predictions': {
                'dates': [format_date(date) for date in predictions['dates']],
                'revenue': [float(rev) for rev in predictions['revenue']],
                'revenue_growth': [float(growth) for growth in predictions['revenue_growth']],
                'risk_levels': predictions['risk_levels'],
                'risk_scores': [float(score) for score in predictions['risk_scores']],
                'profitability': [float(prof) for prof in predictions['profitability']],
                'churn_probability': [float(churn) for churn in predictions['churn_probability']],
                'recommended_actions': predictions['recommended_actions'],
                'action_confidences': [
                    [float(conf) for conf in month_conf] 
                    for month_conf in predictions['action_confidences']
                ]
            },
            'confidence_metrics': [
                {
                    'risk_confidence': float(metric['risk_confidence']),
                    'action_confidence': float(metric['action_confidence']),
                    'revenue_uncertainty': float(metric['revenue_uncertainty']),
                    'prediction_stability': float(metric['prediction_stability'])
                }
                for metric in confidence_metrics
            ]
        }
        logger.debug("Successfully processed predictions")
        return processed_data
    except Exception as e:
        logger.error(f"Error processing predictions: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Endpoint to generate business predictions with detailed P&L and Balance Sheet
    """
    logger.info("Received prediction request")
    try:
        # Get and validate JSON data
        data = request.get_json()
        if not data:
            logger.error("No JSON data received")
            return jsonify({'error': 'No JSON data received'}), 400
            
        logger.debug(f"Received data: {data}")
        
        # Validate required fields
        required_fields = [
            'revenue', 'growth_rate', 'monthly_expenses', 
            'customer_acquisition_cost', 'customer_lifetime_value', 
            'market_size'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            logger.error(f"Missing required fields: {missing_fields}")
            return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400
            
        # Calculate P&L Statement components
        revenue = float(data['revenue'])
        cogs = revenue * 0.6  # Assuming COGS is 60% of revenue
        gross_profit = revenue - cogs
        operating_expenses = float(data['monthly_expenses'])
        operating_income = gross_profit - operating_expenses
        other_income = revenue * 0.05  # Assuming other income is 5% of revenue
        net_profit = operating_income + other_income

        # Calculate Balance Sheet components
        current_assets = revenue * 0.8  # Assuming current assets are 80% of revenue
        fixed_assets = revenue * 2  # Assuming fixed assets are 2x revenue
        total_assets = current_assets + fixed_assets
        current_liabilities = revenue * 0.3  # Assuming current liabilities are 30% of revenue
        long_term_liabilities = revenue * 0.5  # Assuming long-term liabilities are 50% of revenue
        total_liabilities = current_liabilities + long_term_liabilities
        equity = total_assets - total_liabilities

        # Create P&L Statement string
        pl_statement = (
            f'Revenue: {revenue:.0f}\n'
            f'COGS: {cogs:.0f}\n'
            f'Gross Profit: {gross_profit:.0f}\n'
            f'Operating Expenses: {operating_expenses:.0f}\n'
            f'Operating Income: {operating_income:.0f}\n'
            f'Other Income: {other_income:.0f}\n'
            f'Net Profit: {net_profit:.0f}'
        )

        # Create Balance Sheet string
        balance_sheet = (
            f'Current Assets: {current_assets:.0f}\n'
            f'Fixed Assets: {fixed_assets:.0f}\n'
            f'Total Assets: {total_assets:.0f}\n'
            f'Current Liabilities: {current_liabilities:.0f}\n'
            f'Long-term Liabilities: {long_term_liabilities:.0f}\n'
            f'Total Liabilities: {total_liabilities:.0f}\n'
            f'Equity: {equity:.0f}'
        )

        # Create DataFrame with detailed metrics
        try:
            logger.debug("Creating DataFrame from request data")
            current_metrics = pd.DataFrame({
                'P&L_Statement': [pl_statement],
                'Balance_Sheet': [balance_sheet],
                'Monthly_Revenue_Turnover': ['January'],
                'Revenue_Generated': [float(data['revenue'])],
                'Customer_Turnover_Rate': ['Low'],
                'Growth_Rate (%)': [float(data['growth_rate'])],
                'Average_Monthly_Expenses': [float(data['monthly_expenses'])],
                'Customer_Acquisition_Cost (₦)': [float(data['customer_acquisition_cost'])],
                'Lifetime_Value_of_Customer (₦)': [float(data['customer_lifetime_value'])],
                'Market_Size_Potential': [float(data['market_size'])],
                'Risk_Assessment': ['Low'],
                'Predicted_Revenue_Growth (%)': [20.0],
                'Profitability_Score': [75.0],
                'Churn_Rate (%)': [10.0],
                'Recommendation_Actions': ['Expand market reach']
            })
            logger.debug("DataFrame created successfully")
        except ValueError as e:
            logger.error(f"Error creating DataFrame: {str(e)}")
            return jsonify({'error': f'Invalid numeric value in input: {str(e)}'}), 400
        
        # Get optional parameters
        num_months = int(data.get('num_months', 12))
        confidence_threshold = float(data.get('confidence_threshold', 0.7))
        
        logger.debug(f"Using parameters: num_months={num_months}, confidence_threshold={confidence_threshold}")
        
        try:
            logger.info("Generating predictions")
            # Generate predictions
            predictions, confidence_metrics = predictor.predict_monthly_metrics(
                initial_data=current_metrics,
                num_months=num_months,
                confidence_threshold=confidence_threshold
            )
            logger.debug("Predictions generated successfully")
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f'Prediction generation failed: {str(e)}'}), 500
        
        # Process and return results
        try:
            logger.info("Processing prediction results")
            result = process_predictions(predictions, confidence_metrics)
            logger.info("Successfully processed and returning predictions")
            return jsonify(result), 200
        except Exception as e:
            logger.error(f"Error processing predictions: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f'Error processing predictions: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    logger.debug("Health check requested")
    return jsonify({
        'status': 'healthy', 
        'timestamp': datetime.now().isoformat()
    }), 200

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=False, host='0.0.0.0', port=5000)