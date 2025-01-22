import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import json
import copy

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

class BusinessDataGenerator:
    def __init__(self, num_samples: int = 70000):
        self.num_samples = num_samples
        self.risk_categories = [
            "Economic downturns",
            "Operational inefficiencies",
            "Supply chain disruptions",
            "Cybersecurity threats",
            "Regulatory changes",
            "Market competition",
            "Customer churn risks",
            "Technology obsolescence",
            "Talent retention issues",
            "Quality control risks"
        ]

        self.mitigant_categories = [
            "Diversifying revenue streams",
            "Implementing quality control",
            "Employee training",
            "Cybersecurity measures",
            "Contingency planning",
            "Strategic partnerships",
            "Customer retention programs",
            "Technology adoption",
            "Market positioning",
            "Operational optimization"
        ]

    def generate_monthly_revenue(self, months: int = 12) -> List[Dict]:
        base_revenue = random.uniform(800000, 1200000)
        growth_factor = random.uniform(0.98, 1.02)
        turnover_base = random.uniform(15, 30)

        monthly_data = []
        for month in range(months):
            revenue = base_revenue * (growth_factor ** month)
            turnover = max(min(turnover_base + random.uniform(-5, 5), 40), 10)
            monthly_data.append({
                "month": month,
                "revenue": round(revenue, 2),
                "customer_turnover_rate": round(turnover, 2)
            })
        return monthly_data

    def generate_sample(self) -> Dict:
        return {
            "monthly_revenue": self.generate_monthly_revenue(),
            "growth_rate": {
                "value": round(random.uniform(10, 20), 2),
                "duration": random.randint(24, 48)
            },
            "monthly_expenses": {
                "value": round(random.uniform(150000, 250000), 2),
                "duration": random.randint(6, 18)
            },
            "customer_acquisition_cost": {
                "value": round(random.uniform(1500, 2500), 2),
                "duration": random.randint(12, 24)
            },
            "customer_lifetime_value": {
                "value": round(random.uniform(15000, 25000), 2),
                "duration": random.randint(3, 12)
            },
            "market_size": random.randint(8000, 12000),
            "potential_risks": random.sample(self.risk_categories, random.randint(2, 4)),
            "potential_mitigants": random.sample(self.mitigant_categories, random.randint(2, 4))
        }

    def generate_target_metrics(self, sample: Dict) -> Dict:
        revenues = [entry["revenue"] for entry in sample["monthly_revenue"]]
        growth_rates = [(revenues[i+1] - revenues[i]) / revenues[i] * 100
                       for i in range(len(revenues)-1)]
        avg_growth = np.mean(growth_rates) if growth_rates else 0

        risk_score = 0
        risk_score += len(sample["potential_risks"]) * 0.2
        risk_score += (sample["monthly_expenses"]["value"] / revenues[-1]) * 0.3
        risk_score += (sample["customer_acquisition_cost"]["value"] /
                      sample["customer_lifetime_value"]["value"]) * 0.5

        return {
            "growth_predictions": {
                "next_12_months": [
                    revenues[-1] * (1 + (avg_growth/100)) ** (i+1)
                    for i in range(12)
                ]
            },
            "risk_analysis": {
                "Low": 1 - min(risk_score, 0.9),
                "Medium": risk_score * 0.5,
                "High": max(0, risk_score - 0.5)
            },
            "kpi_metrics": {
                "revenue_trend": avg_growth,
                "customer_retention": 100 - np.mean([x["customer_turnover_rate"]
                                                   for x in sample["monthly_revenue"]]),
                "growth_rate": avg_growth,
                "profit_margin": (revenues[-1] - sample["monthly_expenses"]["value"]) / revenues[-1]
            }
        }

class BusinessDataProcessor:
    def __init__(self):
        self.revenue_scaler = StandardScaler()
        self.feature_scaler = StandardScaler()

    def process_input(self, metrics: BusinessMetrics) -> Dict[str, torch.Tensor]:
        # Process temporal data (monthly revenue)
        temporal_features = []
        for month_data in metrics.monthly_revenue:
            temporal_features.append([
                month_data["revenue"],
                month_data["customer_turnover_rate"]
            ])

        temporal_data = torch.tensor(temporal_features, dtype=torch.float32).unsqueeze(0)

        # Process static features
        static_features = [
            metrics.growth_rate["value"],
            metrics.growth_rate["duration"],
            metrics.monthly_expenses["value"],
            metrics.monthly_expenses["duration"],
            metrics.customer_acquisition_cost["value"],
            metrics.customer_acquisition_cost["duration"],
            metrics.customer_lifetime_value["value"],
            metrics.customer_lifetime_value["duration"],
            metrics.market_size,
            len(metrics.potential_risks),
            len(metrics.potential_mitigants)
        ]

        features = torch.tensor(static_features, dtype=torch.float32).unsqueeze(0)

        return {
            "temporal_data": temporal_data,
            "features": features
        }

class BusinessDataset(Dataset):
    def __init__(self, samples: List[Dict], targets: List[Dict]):
        self.samples = samples
        self.targets = targets
        self.processor = BusinessDataProcessor()
        self.revenue_scaler = StandardScaler()

        # Fit the scaler on all revenue data
        all_revenues = []
        for target in targets:
            all_revenues.extend(target['growth_predictions']['next_12_months'])
        self.revenue_scaler.fit(np.array(all_revenues).reshape(-1, 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        target = self.targets[idx]

        processed_input = self.processor.process_input(BusinessMetrics(**sample))
        processed_input['temporal_data'] = processed_input['temporal_data'].squeeze(0)
        processed_input['features'] = processed_input['features'].squeeze(0)

        processed_target = self._process_target(target)
        return processed_input, processed_target

    def _process_target(self, target: Dict) -> Dict[str, torch.Tensor]:
        growth_preds = np.array(target['growth_predictions']['next_12_months'])

        relative_growth = np.diff(growth_preds) / growth_preds[:-1] * 100
        relative_growth = np.insert(relative_growth, 0, 0)

        scaled_growth = self.revenue_scaler.transform(growth_preds.reshape(-1, 1)).flatten()
        log_growth = np.log1p(growth_preds)

        return {
            'growth_predictions': {
                'absolute': torch.tensor(scaled_growth, dtype=torch.float32),
                'relative': torch.tensor(relative_growth, dtype=torch.float32),
                'log': torch.tensor(log_growth, dtype=torch.float32)
            },
            'risk_analysis': torch.tensor([
                target['risk_analysis']['Low'],
                target['risk_analysis']['Medium'],
                target['risk_analysis']['High']
            ], dtype=torch.float32),
            'kpi_metrics': torch.tensor([
                target['kpi_metrics']['revenue_trend'],
                target['kpi_metrics']['customer_retention'],
                target['kpi_metrics']['growth_rate'],
                target['kpi_metrics']['profit_margin']
            ], dtype=torch.float32)
        }

class BusinessModel(nn.Module):
    def __init__(self, temporal_dim: int = 2, feature_dim: int = 11):
        super().__init__()

        # LSTM for temporal data
        self.lstm = nn.LSTM(
            input_size=temporal_dim,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # MLP for static features
        self.feature_network = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # Combined networks for different predictions
        self.growth_network = nn.Sequential(
            nn.Linear(96, 64),  # 64 from LSTM + 32 from feature network
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 36)  # 12 months * 3 metrics (absolute, relative, log)
        )

        self.risk_network = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # Increased dropout
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 3)  # 3 risk categories
        )

        for m in self.risk_network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.kpi_network = nn.Sequential(
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 4)  # 4 KPI metrics
        )

    def forward(self, temporal_data: torch.Tensor, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Process temporal data
        lstm_out, _ = self.lstm(temporal_data)
        lstm_features = lstm_out[:, -1, :]  # Take last LSTM output

        # Process static features
        feature_out = self.feature_network(features)

        # Combine features
        combined = torch.cat([lstm_features, feature_out], dim=1)

        # Generate predictions
        growth_out = self.growth_network(combined)
        risk_out = self.risk_network(combined)
        kpi_out = self.kpi_network(combined)

        # Split growth predictions into absolute, relative, and log components
        growth_preds = {
            'absolute': growth_out[:, :12],
            'relative': growth_out[:, 12:24],
            'log': growth_out[:, 24:]
        }

        return {
            'growth_predictions': growth_preds,
            'risk_analysis': risk_out,
            'kpi_metrics': kpi_out
        }

class BusinessTrainer:
    def __init__(self, model: nn.Module, learning_rate: float = 1e-4):
        self.model = model
        # Improved optimizer configuration
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        # Modified learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )

    def compute_loss(self, predictions: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        losses = {}
        
        # Improved loss calculation with weights
        losses['growth_absolute'] = F.mse_loss(
            predictions['growth_predictions']['absolute'],
            targets['growth_predictions']['absolute']
        )

        # Huber loss for relative growth prediction
        losses['growth_relative'] = F.smooth_l1_loss(
            predictions['growth_predictions']['relative'],
            targets['growth_predictions']['relative']
        )

        losses['growth_log'] = F.mse_loss(
            predictions['growth_predictions']['log'],
            targets['growth_predictions']['log']
        )

        # Improved risk analysis loss
        pred_probs = F.softmax(predictions['risk_analysis'], dim=-1)
        target_probs = F.softmax(targets['risk_analysis'], dim=-1)
        losses['risk'] = F.cross_entropy(
            predictions['risk_analysis'],
            targets['risk_analysis']
        )
        losses['kpi'] = F.smooth_l1_loss(
            predictions['kpi_metrics'],
            targets['kpi_metrics']
        )

        # Balanced weighted loss
        losses['total'] = (
            0.25 * losses['growth_absolute'] +
            0.15 * losses['growth_relative'] +
            0.15 * losses['growth_log'] +
            0.25 * losses['risk'] + 
            0.20 * losses['kpi']
        )
        
        return losses


    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        epoch_losses = defaultdict(float)

        for batch_inputs, batch_targets in train_loader:
            self.optimizer.zero_grad()

            predictions = self.model(
                batch_inputs['temporal_data'],
                batch_inputs['features']
            )

            losses = self.compute_loss(predictions, batch_targets)
            losses['total'].backward()

            self.optimizer.step()

            for k, v in losses.items():
                epoch_losses[k] += v.item()

        val_losses = self.validate(train_loader)
        self.scheduler.step(val_losses['total']) 

        return {k: v/len(train_loader) for k, v in epoch_losses.items()}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        val_losses = defaultdict(float)

        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                predictions = self.model(
                    batch_inputs['temporal_data'],
                    batch_inputs['features']
                )

                losses = self.compute_loss(predictions, batch_targets)

                for k, v in losses.items():
                    val_losses[k] += v.item()

        return {k: v/len(val_loader) for k, v in val_losses.items()}

def collate_fn(batch):
    inputs, targets = zip(*batch)

    temporal_data = torch.stack([item['temporal_data'] for item in inputs])
    features = torch.stack([item['features'] for item in inputs])

    stacked_targets = {
        'growth_predictions': {
            'absolute': torch.stack([item['growth_predictions']['absolute'] for item in targets]),
            'relative': torch.stack([item['growth_predictions']['relative'] for item in targets]),
            'log': torch.stack([item['growth_predictions']['log'] for item in targets])
        },
        'risk_analysis': torch.stack([item['risk_analysis'] for item in targets]),
        'kpi_metrics': torch.stack([item['kpi_metrics'] for item in targets])
    }

    return {'temporal_data': temporal_data, 'features': features}, stacked_targets

def prepare_training_data(num_samples: int = 50000) -> Tuple[DataLoader, DataLoader]:
    generator = BusinessDataGenerator(num_samples)
    samples = []
    targets = []

    print("Generating data samples...")
    for i in range(num_samples):
        if i % 10000 == 0:
            print(f"Generated {i}/{num_samples} samples")
        sample = generator.generate_sample()
        target = generator.generate_target_metrics(sample)
        samples.append(sample)
        targets.append(target)

    train_size = int(0.8 * num_samples)
    train_samples = samples[:train_size]
    train_targets = targets[:train_size]
    val_samples = samples[train_size:]
    val_targets = targets[train_size:]

    print("Creating datasets...")
    train_dataset = BusinessDataset(train_samples, train_targets)
    val_dataset = BusinessDataset(val_samples, val_targets)

    print("Creating dataloaders...")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader

def save_model_checkpoint(model: nn.Module, trainer: BusinessTrainer, epoch: int,
                         train_losses: Dict, val_losses: Dict, filepath: str):
    """Save model checkpoint with all necessary information"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'scheduler_state_dict': trainer.scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model_config': {
            'temporal_dim': model.lstm.input_size,
            'feature_dim': model.feature_network[0].in_features
        }
    }
    torch.save(checkpoint, filepath)
    print(f"Model checkpoint saved to {filepath}")

def load_model_checkpoint(filepath: str) -> Tuple[BusinessModel, BusinessTrainer, Dict]:
    """Load model checkpoint and return model, trainer, and training info"""
    checkpoint = torch.load(filepath)

    # Create model with saved configuration
    model = BusinessModel(
        temporal_dim=checkpoint['model_config']['temporal_dim'],
        feature_dim=checkpoint['model_config']['feature_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create trainer and load optimizer state
    trainer = BusinessTrainer(model)
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    training_info = {
        'epoch': checkpoint['epoch'],
        'train_losses': checkpoint['train_losses'],
        'val_losses': checkpoint['val_losses']
    }

    return model, trainer, training_info

def train_model(model: nn.Module, num_epochs: int = 100,
                checkpoint_dir: str = './checkpoints') -> Tuple[nn.Module, str]:
    import os
    from datetime import datetime

    os.makedirs(checkpoint_dir, exist_ok=True)

    print("Preparing training data...")
    train_loader, val_loader = prepare_training_data(num_samples=70000)  # Increased samples

    print("Initializing trainer...")
    trainer = BusinessTrainer(model)

    # Gradient clipping with higher threshold
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

    best_val_loss = float('inf')
    best_model = None
    patience = 30  # Increased patience
    patience_counter = 0
    min_epochs = 50  # Minimum number of epochs before early stopping

    # Generate unique model ID
    model_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_path = os.path.join(checkpoint_dir, f'model_{model_id}_best.pt')

    print("Starting training...")
    try:
        for epoch in range(num_epochs):
            train_losses = trainer.train_epoch(train_loader)
            val_losses = trainer.validate(val_loader)

            # Update learning rate scheduler
            trainer.scheduler.step(val_losses['total'])

            print(f"Epoch {epoch+1}/{num_epochs}")
            print("Train losses:", {k: f"{v:.4f}" for k, v in train_losses.items()})
            print("Val losses:", {k: f"{v:.4f}" for k, v in val_losses.items()})

            # Save checkpoint every 20 epochs and on the last epoch
            if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f'model_{model_id}_epoch_{epoch+1}.pt'
                )
                save_model_checkpoint(
                    model, trainer, epoch,
                    train_losses, val_losses,
                    checkpoint_path
                )

            # Improved early stopping logic
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                best_model = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                patience_counter = 0
                
                save_model_checkpoint(
                    model, trainer, epoch,
                    train_losses, val_losses,
                    best_model_path
                )
            else:
                patience_counter += 1
                # Only apply early stopping after minimum epochs
                if epoch >= min_epochs and patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break

    except Exception as e:
        print(f"Training interrupted: {str(e)}")
        if best_model is not None:
            print("Saving last best model before exit...")
            model.load_state_dict(best_model)
            save_model_checkpoint(
                model, trainer, best_epoch,
                train_losses, val_losses,
                best_model_path
            )
        raise e

    if best_model is not None:
        model.load_state_dict(best_model)
        print(f"Loaded best model from epoch {best_epoch+1} "
              f"with validation loss: {best_val_loss:.4f}")

    return model, best_model_path

if __name__ == "__main__":
    # Initialize and train model
    model = BusinessModel()
    trained_model, checkpoint_path = train_model(model, num_epochs=100)
    
    # Now checkpoint_path is properly defined before loading
    loaded_model, loaded_trainer, training_info = load_model_checkpoint(checkpoint_path)

    # Generate a sample prediction
    generator = BusinessDataGenerator()
    sample = generator.generate_sample()
    processor = BusinessDataProcessor()
    
    # Make prediction
    loaded_model.eval()
    with torch.no_grad():
        sample_processed = processor.process_input(BusinessMetrics(**sample))
        predictions = loaded_model(
            sample_processed['temporal_data'],
            sample_processed['features']
        )
    
    print("\nPrediction Results:")
    print("Risk Analysis:", F.softmax(predictions['risk_analysis'], dim=-1))
    print("KPI Metrics:", predictions['kpi_metrics'])