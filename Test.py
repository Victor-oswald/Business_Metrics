import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from Train import ImprovedBusinessDataProcessor, EnhancedBusinessPredictor


class TimeSeriesBusinessPredictor:
    def __init__(self, model_path: str, processor: ImprovedBusinessDataProcessor):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = processor

        # Load the saved model with weights_only=True for security
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

        # Initialize the core prediction model
        self.model = EnhancedBusinessPredictor(
            input_dim=checkpoint['input_dim'],
            num_risk_classes=checkpoint['num_risk_classes'],
            num_suggestion_classes=checkpoint['num_suggestion_classes']
        ).to(self.device)

        # Load the trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Store the expected classes for validation
        self.risk_classes = checkpoint.get('risk_classes', ['Low', 'Medium', 'High'])
# Replace the original suggestion_classes with this expanded version
        self.suggestion_classes = [
            # Market Expansion & Growth
            'Expand into new geographic markets through targeted marketing campaigns',
            'Develop partnerships with complementary businesses for market penetration',
            'Launch new product lines based on customer feedback and market research',
            'Establish international presence through e-commerce platforms',
            'Create franchise opportunities for rapid market expansion',

            # Operational Optimization
            'Implement automated inventory management system to reduce costs',
            'Streamline supply chain through strategic supplier partnerships',
            'Adopt lean manufacturing principles to minimize waste',
            'Invest in employee training programs for improved productivity',
            'Upgrade technology infrastructure for better operational efficiency',

            # Customer Experience & Retention
            'Develop personalized customer loyalty program with tiered benefits',
            'Implement AI-powered customer service chatbot for 24/7 support',
            'Create customer feedback loops with regular surveys and focus groups',
            'Establish VIP customer program for high-value clients',
            'Launch customer education programs about product features and benefits',

            # Cost Management
            'Negotiate bulk purchase agreements with suppliers for better rates',
            'Optimize energy usage through smart building management systems',
            'Implement zero-based budgeting for all departments',
            'Outsource non-core business functions to reduce overhead',
            'Invest in preventive maintenance to reduce long-term costs',

            # Marketing & Brand Development
            'Launch targeted social media advertising campaigns',
            'Develop content marketing strategy with industry thought leadership',
            'Create referral program with incentives for existing customers',
            'Implement influencer marketing program in key markets',
            'Enhance brand visibility through community engagement events',

            # Digital Transformation
            'Develop mobile app for improved customer engagement',
            'Implement cloud-based solutions for remote work capability',
            'Create digital payment options for customer convenience',
            'Establish omnichannel presence for seamless customer experience',
            'Implement data analytics for better decision-making',

            # Product Innovation
            'Conduct research and development for product improvements',
            'Create sustainable/eco-friendly product alternatives',
            'Develop subscription-based service models',
            'Launch premium product line for high-end market segment',
            'Create bundled product offerings for increased value',

            # Financial Management
            'Implement dynamic pricing strategy based on market demand',
            'Develop alternative revenue streams through complementary services',
            'Optimize working capital through improved inventory management',
            'Establish strategic partnerships for shared resource utilization',
            'Create financial forecasting models for better planning',

            # Human Resources
            'Implement performance-based incentive programs',
            'Develop career advancement programs for employee retention',
            'Create flexible work arrangements for improved work-life balance',
            'Establish mentorship programs for knowledge transfer',
            'Implement employee wellness programs for improved productivity',

            # Risk Management
            'Develop business continuity plans for various scenarios',
            'Implement cybersecurity measures for data protection',
            'Create disaster recovery protocols for critical systems',
            'Establish quality control processes for consistent delivery',
            'Develop compliance monitoring systems for regulatory requirements'
        ]

        # Initialize prediction history
        self.prediction_history = []

    def _validate_prediction(self, pred_idx: int, valid_classes: List[str]) -> str:
        """Validate prediction index against known classes"""
        if 0 <= pred_idx < len(valid_classes):
            return valid_classes[pred_idx]
        return "Unknown"

    def _calculate_stability(self) -> float:
        """Calculate the stability of predictions over time."""
        if len(self.prediction_history) < 2:
            return 1.0  # If there's only one prediction, it's stable by default

        # Calculate percentage change in revenue and profitability from the previous prediction
        prev_pred = self.prediction_history[-2]
        curr_pred = self.prediction_history[-1]

        revenue_change = (curr_pred['revenue'] - prev_pred['revenue']) / prev_pred['revenue'] if prev_pred['revenue'] != 0 else 0
        profitability_change = (curr_pred['profitability'] - prev_pred['profitability']) / prev_pred['profitability'] if prev_pred['profitability'] != 0 else 0

        # Stability score based on the magnitude of changes (higher score is more stable)
        stability_score = 1 - (abs(revenue_change) + abs(profitability_change)) / 2

        return max(0, stability_score)  # Ensure stability is between 0 and 1


    def _calculate_confidence_metrics(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate confidence scores for various predictions"""
        risk_probs = F.softmax(outputs['risk'], dim=1)
        action_probs = F.softmax(outputs['suggestions'], dim=1)

        # Use robust standard deviation calculation
        revenue_std = torch.std(outputs['revenue'], unbiased=False).item() if outputs['revenue'].numel() > 1 else 0.0

        return {
            'risk_confidence': risk_probs.max().item(),
            'action_confidence': action_probs.max().item(),
            'revenue_uncertainty': revenue_std,
            'prediction_stability': self._calculate_stability()
        }


    def _update_features(self, current_features: torch.Tensor, current_metrics: Dict[str, float], month: int) -> torch.Tensor:
        """Update features with new monthly metrics and add seasonal variation."""
        # Extract relevant features from current_metrics
        base_features = [
            current_metrics['revenue'],
            current_metrics['growth_rate'],
            current_metrics['expenses'],
            current_metrics['cac'],
            current_metrics['ltv'],
            current_metrics['market_size'],
            current_metrics['ltv'] / current_metrics['cac'],
            current_metrics['revenue'] / current_metrics['market_size'],
            current_metrics['expenses'] / current_metrics['revenue'],
            current_metrics['revenue'] * np.random.normal(0.25, 0.03),  # Variable profit margin
            current_metrics['revenue'] * np.random.normal(0.2, 0.02),   # Variable operating income
        ]

        # Add seasonal variation based on month
        seasonal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * month / 12)  # Creates yearly cycle
        market_noise = np.random.normal(1.0, 0.05)  # Random market fluctuations

        # Apply seasonal and random variations
        new_features = [f * seasonal_factor * market_noise for f in base_features]

        # Scale new features using the same scaler used for training data
        new_features_scaled = self.processor.feature_scaler.transform([new_features])

        # Add random noise to prevent perfect convergence
        noise = torch.randn_like(current_features) * 0.01

        # Replace relevant feature values in the current_features tensor
        X_updated = current_features.clone()
        X_updated[0, :len(new_features_scaled[0])] = torch.tensor(new_features_scaled[0]).to(self.device).type(X_updated.dtype)
        X_updated += noise

        return X_updated

    def _calculate_risk_level(self,
                            outputs: Dict[str, torch.Tensor],
                            confidence: float,
                            threshold: float,
                            current_metrics: Dict[str, float]) -> Tuple[str, float]:
        """Calculate risk level using multiple factors"""
        risk_probs = F.softmax(outputs['risk'], dim=1)
        max_prob = risk_probs.max().item()

        # Calculate additional risk factors
        growth_risk = 1.0 if current_metrics['growth_rate'] < 0 else 0.0
        churn_risk = outputs['churn'].item()
        profitability = outputs['profitability'].item()

        # Combine risk factors
        risk_score = (
            0.4 * max_prob +
            0.2 * growth_risk +
            0.2 * churn_risk +
            0.2 * (1 - profitability/100)
        )

        # Determine risk level
        if confidence < threshold:
            return "Uncertain", risk_score
        elif risk_score < 0.3:
            return "Low", risk_score
        elif risk_score < 0.6:
            return "Medium", risk_score
        else:
            return "High", risk_score

    def _get_unique_recommendations(self, top_actions, num_recommendations, recent_actions):
        """Generate unique recommendations avoiding recent duplicates."""
        unique_recommendations = []
        unique_confidences = []

        for idx, (action_idx, confidence) in enumerate(zip(top_actions.indices[0], top_actions.values[0])):
            action_label = self._validate_prediction(action_idx.item(), self.suggestion_classes)

            # Avoid actions already recommended recently
            if action_label not in recent_actions and action_label not in unique_recommendations:
                unique_recommendations.append(action_label)
                unique_confidences.append(confidence.item())

            # Break loop if we have enough unique recommendations
            if len(unique_recommendations) >= num_recommendations:
                break

            # If not enough unique recommendations, allow some repeats with reduced confidence
            while len(unique_recommendations) < num_recommendations:
                fallback_idx = np.random.choice(top_actions.indices[0].cpu().numpy())
                fallback_label = self._validate_prediction(fallback_idx.item(), self.suggestion_classes)
                if fallback_label not in unique_recommendations:
                    unique_recommendations.append(fallback_label)
                    unique_confidences.append(0.5)  # Assign lower confidence for fallback actions

            return unique_recommendations, unique_confidences

    def predict_monthly_metrics(
        self,
        initial_data: pd.DataFrame,
        num_months: int = 12,
        confidence_threshold: float = 0.7,
        num_recommendations: int = 3  # Number of recommendations to return
    ) -> Tuple[Dict[str, List], List[Dict[str, float]]]:
        """Make predictions with multiple recommendations per period"""
        predictions = {
            'dates': [],
            'revenue': [],
            'revenue_growth': [],
            'risk_levels': [],
            'risk_scores': [],
            'profitability': [],
            'churn_probability': [],
            'recommended_actions': [],  # Now will contain lists of recommendations
            'action_confidences': [],   # Store confidence scores for each recommendation
        }

        confidence_metrics = []
        start_date = pd.Timestamp.now().replace(day=1)
        current_data = initial_data.copy()

        # Process initial data
        features, _, _, _, _, _, month_nums, quarter_nums = self.processor.process_data(current_data)
        X = torch.FloatTensor(features).to(self.device)

        # Initialize current_metrics before the loop.
        current_metrics = {
            'revenue': current_data['Revenue_Generated'].values[0],
            'growth_rate': current_data['Growth_Rate (%)'].values[0],
            'expenses': current_data['Average_Monthly_Expenses'].values[0],
            'cac': current_data['Customer_Acquisition_Cost (₦)'].values[0],
            'ltv': current_data['Lifetime_Value_of_Customer (₦)'].values[0],
            'market_size': current_data['Market_Size_Potential'].values[0]
        }

        for month in range(num_months):
            current_date = start_date + pd.DateOffset(months=month)
            current_month = current_date.month
            current_quarter = (current_month - 1) // 3 + 1

            # Prepare temporal features
            month_tensor = torch.LongTensor([current_month]).to(self.device)
            quarter_tensor = torch.LongTensor([current_quarter]).to(self.device)

            with torch.no_grad():
                # Get model outputs
                outputs = self.model(X, month_tensor, quarter_tensor)
                # print(f"outputs from model{outputs}")

                # Calculate confidence metrics
                confidence = self._calculate_confidence_metrics(outputs)

                # Compute action probabilities
                action_probs = F.softmax(outputs['suggestions'], dim=1)

                # Dynamically adjust k for torch.topk
                num_actions = action_probs.size(1)
                k = min(10, num_actions)

                # Generate recommendations with diversity constraints
                top_actions = torch.topk(action_probs, k, dim=1)  # Adjusted k
                recent_actions = predictions['recommended_actions'][-3:] if len(predictions['recommended_actions']) > 3 else []
                month_recommendations, month_confidences = self._get_unique_recommendations(
                    top_actions, num_recommendations, recent_actions
                )

            # Calculate revenue with seasonal variation
            base_growth = outputs['revenue'].item()
            seasonal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * current_month / 12)
            market_noise = np.random.normal(1.0, 0.02)
            revenue_growth = base_growth * seasonal_factor * market_noise

            current_revenue = (
                current_data['Revenue_Generated'].values[0] if month == 0
                else predictions['revenue'][-1] * (1 + revenue_growth / 100)
            )

            # Calculate risk with improved assessment
            current_metrics = {
                'revenue': current_revenue,
                'growth_rate': revenue_growth,
                'expenses': current_data['Average_Monthly_Expenses'].values[0],
                'cac': current_data['Customer_Acquisition_Cost (₦)'].values[0],
                'ltv': current_data['Lifetime_Value_of_Customer (₦)'].values[0],
                'market_size': current_data['Market_Size_Potential'].values[0]
            }

            risk_label, risk_score = self._calculate_risk_level(
                outputs,
                confidence['risk_confidence'],
                confidence_threshold,
                current_metrics
            )

            # Store predictions with added variability
            predictions['dates'].append(current_date)
            predictions['revenue'].append(current_revenue)
            predictions['revenue_growth'].append(revenue_growth)
            predictions['risk_levels'].append(risk_label)
            predictions['risk_scores'].append(risk_score)
            predictions['profitability'].append(outputs['profitability'].item() * np.random.normal(1.0, 0.05))
            predictions['churn_probability'].append(outputs['churn'].item() * np.random.normal(1.0, 0.03))
            predictions['recommended_actions'].append(month_recommendations)
            predictions['action_confidences'].append(month_confidences)

            confidence_metrics.append(confidence)

            # Update features for next prediction
            X = self._update_features(X, current_metrics, current_month)

        # Return predictions after processing all months
        return predictions, confidence_metrics


if __name__ == "__main__":
    # Create sample current metrics
    current_metrics = pd.DataFrame({
        'P&L_Statement': ['Revenue: 1000000\nCOGS: 600000\nGross Profit: 400000\n' +
                         'Operating Expenses: 200000\nOperating Income: 200000\n' +
                         'Other Income: 50000\nNet Profit: 250000'],
        'Balance_Sheet': ['Current Assets: 800000\nFixed Assets: 2000000\n' +
                         'Total Assets: 2800000\nCurrent Liabilities: 300000\n' +
                         'Long-term Liabilities: 500000\nTotal Liabilities: 800000\n' +
                         'Equity: 2000000'],
        'Monthly_Revenue_Turnover': ['January'],
        'Revenue_Generated': [1000000],
        'Customer_Turnover_Rate': ['Low'],
        'Growth_Rate (%)': [15.0],
        'Average_Monthly_Expenses': [200000],
        'Customer_Acquisition_Cost (₦)': [2000],
        'Lifetime_Value_of_Customer (₦)': [20000],
        'Market_Size_Potential': [10000],
        'Risk_Assessment': ['Low'],
        'Predicted_Revenue_Growth (%)': [20.0],
        'Profitability_Score': [75.0],
        'Churn_Rate (%)': [10.0],
        'Recommendation_Actions': ['Expand market reach']
    })

    # Initialize predictor
    processor = ImprovedBusinessDataProcessor()
    predictor = TimeSeriesBusinessPredictor('improved_business_predictor.pth', processor)

    # Make predictions
    predictions, confidence_metrics = predictor.predict_monthly_metrics(
        initial_data=current_metrics,
        num_months=12,
        confidence_threshold=0.7
    )

    # print(predictions)

    # Print predictions
    print("\nPredictions for the next 12 months:")
    for i in range(len(predictions['dates'])):
        print(f"\nMonth {predictions['dates'][i].strftime('%Y-%m')}:")
        print(f"Revenue Growth: {predictions['revenue_growth'][i]:.2f}%")
        print(f"Predicted Revenue: ₦{predictions['revenue'][i]:,.2f}")
        print(f"Risk Level: {predictions['risk_levels'][i]}")
        print(f"Profitability Score: {predictions['profitability'][i]:.2f}")
        print(f"Churn Probability: {predictions['churn_probability'][i]*100:.2f}%")
        print(f"Recommended Action: {predictions['recommended_actions'][i]}")

    # Visualize predictions
    # predictor.visualize_predictions(predictions, confidence_metrics)