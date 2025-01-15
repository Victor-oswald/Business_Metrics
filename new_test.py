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

def normalize_predictions(results: Dict) -> Dict:
    risk_probs = F.softmax(results['risk_analysis'], dim=1).squeeze().tolist()
    kpi_metrics = torch.sigmoid(results['kpi_metrics']).squeeze().tolist()
    growth_preds = results['growth_predictions']
    base_revenue = 1000000
    
    growth_rates = torch.clamp(
        F.tanh(growth_preds['relative']) * 0.15 + 0.05,
        min=-0.10,
        max=0.20
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

def generate_business_insights(normalized_results: Dict, input_data: Dict) -> Dict:
    """Generate dynamic business insights based on input data and model predictions"""
    risk_probs = normalized_results['risk_probs']
    kpi_metrics = normalized_results['kpi_metrics']
    monthly_revenues = normalized_results['monthly_revenues']
    growth_rates = normalized_results['growth_rates']
    
    def analyze_risks():
        risk_analysis = []
        risk_categories = {
            "economic": ["recession", "downturn", "inflation", "market"],
            "operational": ["efficiency", "process", "operational", "staff", "training"],
            "competitive": ["competition", "competitor", "market share"],
            "supply": ["supplier", "inventory", "material", "price"],
            "customer": ["customer", "retention", "attrition", "churn"]
        }
        
        for risk in input_data['potential_risks']:
            risk_lower = risk.lower()
            risk_type = next((category for category, keywords in risk_categories.items() 
                            if any(keyword in risk_lower for keyword in keywords)), "general")
            
            likelihood = "High" if risk_probs[0] > 0.6 else "Medium" if risk_probs[0] > 0.3 else "Low"
            
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
                    "Implement monitoring and early warning systems",
                    "Create contingency plans"
                ]
            }
            
            risk_analysis.append({
                "risk": risk,
                "likelihood": likelihood,
                "impact": "High" if risk_probs[1] > 0.5 else "Moderate",
                "recommendation": random.choice(recommendations.get(risk_type, recommendations["general"]))
            })
            
        return risk_analysis

    def generate_opportunities():
        opportunities = []
        avg_monthly_growth = (monthly_revenues[-1] / monthly_revenues[0] - 1) / 12
        market_penetration = monthly_revenues[0] / (input_data['market_size'] * 12)
        customer_value = input_data['customer_lifetime_value']['value']
        
        if market_penetration < 0.3:
            opportunities.append({
                "category": "Market Expansion",
                "initiatives": [
                    f"Target {((0.3 - market_penetration) * 100):.1f}% market share increase",
                    f"Estimated revenue potential: ${(monthly_revenues[0] * 0.3):.0f}/month",
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
                    f"Target customer lifetime value: ${customer_value * 1.5:.0f}",
                    "Create customer loyalty program"
                ]
            })
            
        return opportunities

    def generate_mitigation_strategies():
        strategies = []
        
        if kpi_metrics[2] < 0.2:
            strategies.append({
                "strategy": "Cost Optimization",
                "description": f"Target {(kpi_metrics[2] * 100 + 5):.1f}% margin through operational efficiency"
            })
            
        if input_data['monthly_revenue'][0]['customer_turnover_rate'] > 15:
            strategies.append({
                "strategy": "Customer Retention",
                "description": f"Reduce churn by {(input_data['monthly_revenue'][0]['customer_turnover_rate'] - 10):.1f}% through targeted programs"
            })
            
        for mitigant in input_data['potential_mitigants']:
            strategies.append({
                "strategy": mitigant.split(" (")[0],
                "description": f"Implement {mitigant.lower()} with measurable KPIs"
            })
            
        return strategies

    return {
        "growth_predictions": {
            "projected_growth": f"{(sum(growth_rates) / len(growth_rates) * 100):.1f}%",
            "revenue_range": f"${min(monthly_revenues)/1000000:.1f}M - ${max(monthly_revenues)/1000000:.1f}M",
            "customer_metrics": {
                "daily_churn": input_data['monthly_revenue'][0]['customer_turnover_rate'],
                "improvement_target": f"{int(input_data['monthly_revenue'][0]['customer_turnover_rate'] * 0.7)}-{int(input_data['monthly_revenue'][0]['customer_turnover_rate'] * 0.85)} customers with optimized retention"
            }
        },
        "risk_analysis": analyze_risks(),
        "opportunities": generate_opportunities(),
        "mitigation_strategies": generate_mitigation_strategies(),
        "kpi_metrics": {
            "revenue": {
                "current": f"${monthly_revenues[0]/1000000:.1f}M",
                "projected": f"${monthly_revenues[-1]/1000000:.1f}M"
            },
            "customer_retention": {
                "current": f"{kpi_metrics[1] * 100:.1f}%",
                "target": f"{min(85, kpi_metrics[1] * 100 + 10):.1f}% within 6 months"
            },
            "profit_margin": {
                "current": f"{kpi_metrics[2] * 100:.1f}%",
                "target": f"{min(35, kpi_metrics[2] * 100 + 15):.1f}% through efficiency improvements"
            }
        }
    }