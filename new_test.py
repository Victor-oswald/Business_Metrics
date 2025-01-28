import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List
import random

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

class BusinessModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=64, num_layers=2, batch_first=True)
        
        self.feature_network = nn.Sequential(
            nn.Linear(11, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32)
        )
        
        self.risk_network = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3)
        )
        
        self.kpi_network = nn.Sequential(
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 4)
        )
        
        self.growth_network = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 36)
        )
    
    def forward(self, temporal_data, features):
        temporal_encoded, _ = self.lstm(temporal_data)
        temporal_encoded = temporal_encoded[:, -1, :]
        feature_encoded = self.feature_network(features)
        combined = torch.cat([temporal_encoded, feature_encoded], dim=1)
        
        risk_analysis = self.risk_network(combined)
        kpi_metrics = self.kpi_network(combined)
        growth_raw = self.growth_network(combined)
        
        growth_predictions = {
            'absolute': growth_raw[:, :12],
            'relative': growth_raw[:, 12:24],
            'log': growth_raw[:, 24:]
        }
        
        return {
            'risk_analysis': risk_analysis,
            'kpi_metrics': kpi_metrics,
            'growth_predictions': growth_predictions
        }

class BusinessDataProcessor:
    def process_input(self, data: BusinessMetrics) -> Dict:
        temporal_data = torch.tensor([
            [
                entry['revenue'],
                entry['customer_turnover_rate']
            ] for entry in data.monthly_revenue
        ], dtype=torch.float32).unsqueeze(0)
        
        features = torch.tensor([
            data.growth_rate['value'],
            data.growth_rate['duration'],
            data.monthly_expenses['value'],
            data.monthly_expenses['duration'],
            data.customer_acquisition_cost['value'],
            data.customer_acquisition_cost['duration'],
            data.customer_lifetime_value['value'],
            data.customer_lifetime_value['duration'],
            data.market_size,
            len(data.potential_risks),
            len(data.potential_mitigants)
        ], dtype=torch.float32).unsqueeze(0)
        
        return {
            'temporal_data': temporal_data,
            'features': features
        }

    def normalize_kpi_metrics(metrics):
        # Apply different normalization for different metrics
        normalized = []
        for i, metric in enumerate(metrics):
            if i == 1:  # Customer retention
                # Scale to realistic retention range (50-95%)
                normalized.append(50 + torch.sigmoid(metric).item() * 45)
            elif i == 2:  # Profit margin
                # Scale to realistic margin range (5-60%)
                normalized.append(5 + torch.sigmoid(metric).item() * 55)
            else:
                # Other metrics can use original scaling if appropriate
                normalized.append((torch.tanh(metric).item() + 1) * 0.5 * 100)
        return normalized

    # kpi_metrics = normalize_kpi_metrics(results['kpi_metrics'].squeeze())

def generate_business_insights(normalized_results: Dict, input_data: Dict) -> Dict:
    risk_probs = normalized_results['risk_probs']
    kpi_metrics = normalized_results['kpi_metrics']
    monthly_revenues = normalized_results['monthly_revenues']
    growth_rates = normalized_results['growth_rates']
    
    def analyze_risks():
        risk_categories = {
            "economic": ["recession", "downturn", "inflation", "market"],
            "operational": ["efficiency", "process", "operational", "staff", "training"],
            "competitive": ["competition", "competitor", "market share"],
            "supply": ["supplier", "inventory", "material", "price"],
            "customer": ["customer", "retention", "attrition", "churn"]
        }
        
        risk_analysis = []
        for i, risk in enumerate(input_data['potential_risks']):
            risk_lower = risk.lower()
            risk_type = next((category for category, keywords in risk_categories.items() 
                            if any(keyword in risk_lower for keyword in keywords)), "general")
            
            risk_prob = risk_probs[i % len(risk_probs)]
            likelihood = "High" if risk_prob > 0.6 else "Medium" if risk_prob > 0.3 else "Low"
            impact = "High" if risk_prob > 0.7 else "Moderate" if risk_prob > 0.4 else "Low"
            
            recommendations = {
                "economic": [
                    "Diversify revenue streams through multiple channels",
                    "Build cash reserves for economic downturns",
                    "Develop flexible pricing strategies"
                ],
                "operational": [
                    "Implement automated workflow systems",
                    "Invest in staff training programs",
                    "Optimize resource allocation"
                ],
                "competitive": [
                    "Develop unique value propositions",
                    "Invest in market differentiation",
                    "Focus on customer experience improvements"
                ],
                "supply": [
                    "Establish multiple supplier relationships",
                    "Implement inventory management systems",
                    "Negotiate long-term contracts"
                ],
                "customer": [
                    "Enhance customer loyalty programs",
                    "Improve customer service quality",
                    "Implement feedback systems"
                ],
                "general": [
                    "Develop comprehensive risk management plan",
                    "Implement monitoring systems",
                    "Create contingency plans"
                ]
            }
            
            recommendation = random.choice(recommendations.get(risk_type, recommendations["general"]))
            
            risk_analysis.append({
                "risk": risk,
                "likelihood": likelihood,
                "impact": impact,
                "recommendation": recommendation
            })
            
        return risk_analysis

    def generate_opportunities():
        opportunities = []
        avg_monthly_growth = (monthly_revenues[-1] / monthly_revenues[0] - 1) / len(monthly_revenues)
        market_penetration = monthly_revenues[0] / (input_data['market_size'] * 12)
        customer_value = input_data['customer_lifetime_value']['value']
        
        if market_penetration < 0.3:
            opportunities.append({
                "category": "Market Expansion",
                "initiatives": [
                    f"Target {((0.3 - market_penetration) * 100):.1f}% market share increase",
                    f"Estimated revenue potential: ₦{(monthly_revenues[0] * 0.3):.0f}/month",
                    "Develop targeted marketing campaigns",
                    "Expand geographical presence"
                ]
            })
            
        if avg_monthly_growth < 0.05:
            opportunities.append({
                "category": "Growth Acceleration",
                "initiatives": [
                    "Implement customer acquisition strategy",
                    f"Target growth rate: {(avg_monthly_growth + 0.03) * 100:.1f}%",
                    "Explore new revenue streams",
                    "Optimize pricing strategy"
                ]
            })
            
        if customer_value < input_data['customer_acquisition_cost']['value'] * 5:
            opportunities.append({
                "category": "Customer Value Enhancement",
                "initiatives": [
                    "Develop premium offerings",
                    "Implement cross-selling programs",
                    f"Target customer lifetime value: ₦{customer_value * 1.5:.0f}",
                    "Create customer loyalty program"
                ]
            })
            
        return opportunities

    def generate_mitigation_strategies():
        strategies = []
        actual_margin = kpi_metrics[2]
        
        if actual_margin < 50:
            strategies.append({
                "strategy": "Cost Optimization",
                "description": f"Target {min(actual_margin + 15, 70):.1f}% margin through operational efficiency"
            })
            
        actual_churn = input_data['monthly_revenue'][0]['customer_turnover_rate']
        if actual_churn > 0:
            target_reduction = min(actual_churn * 0.3, 15)
            strategies.append({
                "strategy": "Customer Retention",
                "description": f"Reduce churn by {target_reduction:.1f}% through targeted programs"
            })
            
        for mitigant in input_data['potential_mitigants']:
            strategies.append({
                "strategy": mitigant.split(" (")[0],
                "description": f"Implement {mitigant.lower()} with projected impact of {random.randint(10, 30)}%"
            })
            
        return strategies

    def format_improvement_target(current_churn: float) -> str:
        """
        Format the improvement target string with proper range ordering
        
        Args:
            current_churn: Current customer turnover rate
            
        Returns:
            Formatted improvement target string
        """
        target_upper = max(5, int(current_churn * 0.6))
        target_lower = int(current_churn * 0.8)
        
        # Ensure the range is properly ordered (lower-upper)
        if target_lower > target_upper:
            target_lower, target_upper = target_upper, target_lower
            
        return f"{target_lower}-{target_upper} customers with optimized retention"

    return {
        "growth_predictions": {
            "projected_growth": f"{(sum(growth_rates) / len(growth_rates) * 100):.1f}%",
            "revenue_range": f"₦{min(monthly_revenues)/1000000:.1f}M - ₦{max(monthly_revenues)/1000000:.1f}M",
            "customer_metrics": {
                "daily_churn": input_data['monthly_revenue'][0]['customer_turnover_rate'],
                "improvement_target": format_improvement_target(input_data['monthly_revenue'][0]['customer_turnover_rate'])
            }
        },
        "risk_analysis": analyze_risks(),
        "opportunities": generate_opportunities(),
        "mitigation_strategies": generate_mitigation_strategies(),
        "kpi_metrics": {
            "revenue": {
                "current": f"₦{monthly_revenues[0]/1000000:.1f}M",
                "projected": f"₦{monthly_revenues[-1]/1000000:.1f}M"
            },
            "customer_retention": {
                "current": f"{kpi_metrics[1]:.1f}%",
                "target": f"{min(95, kpi_metrics[1] + random.uniform(5, 15)):.1f}% within 6 months"
            },
            "profit_margin": {
                "current": f"{kpi_metrics[2]:.1f}%",
                "target": f"{min(70, kpi_metrics[2] + random.uniform(10, 20)):.1f}% through efficiency improvements"
            }
        }
    }

def load_trained_model(model_path: str, device: str = 'cpu') -> BusinessModel:
    """
    Load a trained BusinessModel from a checkpoint file.
    
    Args:
        model_path: Path to the checkpoint file
        device: Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        Loaded BusinessModel instance
    """
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize the model
    model = BusinessModel()
    
    # Load just the model state dict from the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    return model.to(device)


def normalize_predictions(results: Dict) -> Dict:
    """
    Normalize the raw model outputs into meaningful business metrics.
    
    Args:
        results: Dictionary containing raw model outputs
        
    Returns:
        Dictionary containing normalized predictions
    """
    # Normalize risk analysis probabilities
    risk_probs = F.softmax(results['risk_analysis'], dim=1).squeeze().tolist()
    
    # Normalize KPI metrics with more realistic ranges
    raw_kpi_metrics = results['kpi_metrics'].squeeze().tolist()
    kpi_metrics = []
    
    for i, metric in enumerate(raw_kpi_metrics):
        if i == 1:  # Customer retention
            # Scale to more realistic retention range (65-85%)
            normalized_value = 65 + torch.sigmoid(torch.tensor(metric)).item() * 20
        elif i == 2:  # Profit margin
            # Scale to more realistic margin range (8-35%)
            normalized_value = 8 + torch.sigmoid(torch.tensor(metric)).item() * 27
        else:
            # Other metrics use standard normalization
            normalized_value = (torch.tanh(torch.tensor(metric)).item() + 1) * 0.5 * 100
        kpi_metrics.append(normalized_value)
    
    # Rest of the normalization code remains the same
    growth_preds = results['growth_predictions']
    base_revenue = 1000000  # Starting revenue
    
    growth_rates = torch.clamp(
        F.tanh(growth_preds['relative']) * 0.25,
        min=-0.15,
        max=0.25
    ).squeeze().tolist()
    
    monthly_revenues = []
    current_revenue = base_revenue
    for rate in growth_rates:
        current_revenue *= (1 + rate)
        monthly_revenues.append(current_revenue)
    
    return {
        'risk_probs': risk_probs,
        'kpi_metrics': kpi_metrics,
        'monthly_revenues': monthly_revenues,
        'growth_rates': growth_rates
    }
class BusinessPredictor:
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path: Path to the checkpoint file
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = device
        self.model = load_trained_model(model_path, device)
        self.processor = BusinessDataProcessor()
    
    @torch.no_grad()
    def predict(self, data: BusinessMetrics) -> Dict:
        """
        Make predictions using the trained model.
        """
        processed_data = self.processor.process_input(data)
        
        temporal_data = processed_data['temporal_data'].to(self.device)
        features = processed_data['features'].to(self.device)
        
        model_output = self.model(temporal_data, features)
        normalized_results = normalize_predictions(model_output)
        
        return generate_business_insights(normalized_results, data.__dict__)

def main():
    # Sample data
    sample_data = BusinessMetrics(
        monthly_revenue=[
            {"revenue": 1000000, "customer_turnover_rate": 15},
            {"revenue": 1100000, "customer_turnover_rate": 14},
            {"revenue": 1200000, "customer_turnover_rate": 13}
        ],
        growth_rate={"value": 0.1, "duration": 12},
        monthly_expenses={"value": 800000, "duration": 12},
        customer_acquisition_cost={"value": 500, "duration": 12},
        customer_lifetime_value={"value": 2000, "duration": 24},
        market_size=10000000,
        potential_risks=[
            "Economic downturns (Market & Economic Risks)",
            "Operational inefficiencies (Operational Risks)",
            "Supply chain disruptions (Supply Risks)"
        ],
        potential_mitigants=[
            "Diversifying revenue streams (revenue growth)",
            "Implementing automated systems (operational)",
            "Building customer loyalty program (customer)"
        ]
    )
    
    # Initialize predictor with trained model
    predictor = BusinessPredictor(
        model_path='checkpoints/model.pt',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Generate predictions and insights
    insights = predictor.predict(sample_data)
    return insights

if __name__ == "__main__":
    insights = main()
    print(insights)