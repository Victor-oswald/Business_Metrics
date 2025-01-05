import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import defaultdict
from faker import Faker
import random
import torch.nn.functional as F
from typing import Dict, List
import matplotlib.pyplot as plt


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class BusinessDataset(Dataset):
    def __init__(self, X, y_revenue, y_risk, y_profitability, y_churn, y_suggestions, month_nums, quarter_nums):
        self.X = X if torch.is_tensor(X) else torch.FloatTensor(X)
        self.y_revenue = y_revenue if torch.is_tensor(y_revenue) else torch.FloatTensor(y_revenue)
        self.y_risk = y_risk if torch.is_tensor(y_risk) else torch.LongTensor(y_risk)
        self.y_profitability = y_profitability if torch.is_tensor(y_profitability) else torch.FloatTensor(y_profitability)
        self.y_churn = y_churn if torch.is_tensor(y_churn) else torch.FloatTensor(y_churn)
        self.y_suggestions = y_suggestions if torch.is_tensor(y_suggestions) else torch.LongTensor(y_suggestions)
        self.month_nums = month_nums if torch.is_tensor(month_nums) else torch.LongTensor(month_nums)
        self.quarter_nums = quarter_nums if torch.is_tensor(quarter_nums) else torch.LongTensor(quarter_nums)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.y_revenue[idx],
            self.y_risk[idx],
            self.y_profitability[idx],
            self.y_churn[idx],
            self.y_suggestions[idx],
            self.month_nums[idx],
            self.quarter_nums[idx]
        )


class EnhancedBusinessPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_risk_classes: int = 3, num_suggestion_classes: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Temporal embedding layers
        self.month_embedding = nn.Embedding(13, 16)  # 12 months + padding
        self.quarter_embedding = nn.Embedding(5, 8)   # 4 quarters + padding

        # Linear layer to match dimensions before LSTM
        self.input_proj = nn.Linear(input_dim + 24, hidden_dim)

        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )

        # Task-specific layers with dropout
        self.dropout = nn.Dropout(0.3)

        # Revenue prediction
        self.revenue_layer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        # Risk assessment
        self.risk_layer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_risk_classes)
        )

        # Profitability prediction
        self.profitability_layer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        # Churn prediction
        self.churn_layer = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Action recommendation
        self.suggestion_layer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_suggestion_classes)
        )

    def forward(self, x, months, quarters):
        # Get temporal embeddings
        month_emb = self.month_embedding(months)
        quarter_emb = self.quarter_embedding(quarters)
        temporal_features = torch.cat([month_emb, quarter_emb], dim=-1)

        # Combine features
        combined = torch.cat([x, temporal_features], dim=-1)

        # Project to correct dimension
        projected = self.input_proj(combined)
        projected = projected.unsqueeze(1)  # Add sequence dimension

        # Process through LSTM
        lstm_out, _ = self.lstm(projected)

        # Get final hidden state
        features = lstm_out.squeeze(1)
        features = self.dropout(features)

        # Task-specific predictions
        return {
            'revenue': self.revenue_layer(features),
            'risk': self.risk_layer(features),
            'profitability': self.profitability_layer(features),
            'churn': self.churn_layer(features),
            'suggestions': self.suggestion_layer(features)
        }



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Linear(in_channels, out_channels)
        self.conv2 = nn.Linear(out_channels, out_channels)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(0.2)

        # Projection shortcut if dimensions don't match
        self.shortcut = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = F.gelu(x)  # GELU activation for better gradient flow
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.dropout(x)

        x += identity
        return F.gelu(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence length dimension
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.norm(x)
        return x.squeeze(1)

class TaskNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 32], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def generate_business_data(n_samples):
    np.random.seed(42)
    fake = Faker()

    def generate_pl_statement():
        revenue = np.random.randint(500000, 5000000)
        cogs = np.random.randint(100000, revenue)
        gross_profit = revenue - cogs
        operating_expenses = np.random.randint(100000, 1000000)
        operating_income = gross_profit - operating_expenses
        other_income = np.random.randint(10000, 100000)
        net_profit = operating_income + other_income
        return f"Revenue: {revenue}\nCOGS: {cogs}\nGross Profit: {gross_profit}\n" \
               f"Operating Expenses: {operating_expenses}\nOperating Income: {operating_income}\n" \
               f"Other Income: {other_income}\nNet Profit: {net_profit}"

    def generate_balance_sheet():
        current_assets = np.random.randint(500000, 2000000)
        fixed_assets = np.random.randint(1000000, 5000000)
        total_assets = current_assets + fixed_assets
        current_liabilities = np.random.randint(100000, 500000)
        long_term_liabilities = np.random.randint(200000, 1000000)
        total_liabilities = current_liabilities + long_term_liabilities
        equity = total_assets - total_liabilities
        return f"Current Assets: {current_assets}\nFixed Assets: {fixed_assets}\n" \
               f"Total Assets: {total_assets}\nCurrent Liabilities: {current_liabilities}\n" \
               f"Long-term Liabilities: {long_term_liabilities}\nTotal Liabilities: {total_liabilities}\n" \
               f"Equity: {equity}"

    def generate_recommendation_actions():
        actions = [
            "Expand market reach",
            "Optimize costs",
            "Invest in marketing",
            "Diversify products",
            "Improve customer service",
            "Increase R&D",
            "Enhance loyalty programs",
            "Cut overhead costs"
        ]
        return random.choice(actions)

    data = {
        'P&L_Statement': [generate_pl_statement() for _ in range(n_samples)],
        'Balance_Sheet': [generate_balance_sheet() for _ in range(n_samples)],
        'Monthly_Revenue_Turnover': [fake.month_name() for _ in range(n_samples)],
        'Revenue_Generated': np.random.randint(50000, 1000000, n_samples),
        'Customer_Turnover_Rate': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'Growth_Rate (%)': np.round(np.random.uniform(5, 25, n_samples), 2),
        'Average_Monthly_Expenses': np.random.randint(10000, 500000, n_samples),
        'Customer_Acquisition_Cost (₦)': np.random.randint(500, 5000, n_samples),
        'Lifetime_Value_of_Customer (₦)': np.random.randint(5000, 100000, n_samples),
        'Market_Size_Potential': np.random.randint(500, 50000, n_samples),
        'Risk_Assessment': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'Predicted_Revenue_Growth (%)': np.round(np.random.uniform(5, 30, n_samples), 2),
        'Profitability_Score': np.round(np.random.uniform(50, 100, n_samples), 2),
        'Churn_Rate (%)': np.round(np.random.uniform(5, 40, n_samples), 2),
        'Recommendation_Actions': [generate_recommendation_actions() for _ in range(n_samples)]
    }

    return pd.DataFrame(data)

def improved_preprocessing(df):
    # Enhanced feature engineering
    df['Revenue_Growth_Rate'] = df.groupby('Month_Num')['Revenue_Generated'].pct_change()
    df['Profit_Margin'] = df['Net_Profit'] / df['Revenue_Generated']
    df['Operating_Margin'] = df['Operating_Income'] / df['Revenue_Generated']
    df['Customer_Efficiency'] = df['Revenue_Generated'] / df['Market_Size_Potential']

    # Add cyclical encoding for months
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month_Num'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month_Num'] / 12)

    # Create interaction features
    df['CAC_CLV_Ratio'] = df['Customer_Acquisition_Cost (₦)'] / df['Lifetime_Value_of_Customer (₦)']
    df['Revenue_per_Expense'] = df['Revenue_Generated'] / df['Average_Monthly_Expenses']

    # Add rolling statistics
    df['Rolling_Avg_Revenue'] = df.groupby('Month_Num')['Revenue_Generated'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    if 'Month_Num' not in df or df['Revenue_Generated'].isnull().any():
      raise ValueError("Required columns are missing or contain NaNs.")


    return df


class ImprovedBusinessDataProcessor:
    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.label_encoder_turnover = LabelEncoder()
        self.label_encoder_risk = LabelEncoder()
        self.label_encoder_suggestions = LabelEncoder()

    def process_data(self, df):
        df = df.copy()

        # Add month and quarter numbers
        month_map = {
            month: idx for idx, month in enumerate(
                pd.date_range('2024-01-01', '2024-12-31', freq='ME').strftime('%B'), 1
            )
        }
        df['Month_Num'] = df['Monthly_Revenue_Turnover'].map(month_map)
        df['Quarter_Num'] = ((df['Month_Num'] - 1) // 3) + 1

        # Handle categorical variables
        df['Customer_Turnover_Rate'] = self.label_encoder_turnover.fit_transform(df['Customer_Turnover_Rate'])
        df['Risk_Assessment'] = self.label_encoder_risk.fit_transform(df['Risk_Assessment'])
        df['Recommendation_Actions'] = self.label_encoder_suggestions.fit_transform(df['Recommendation_Actions'])

        # Extract financial metrics from P&L Statement
        df['Net_Profit'] = df['P&L_Statement'].apply(lambda x: float(x.split('Net Profit: ')[1]))
        df['Operating_Income'] = df['P&L_Statement'].apply(lambda x: float(x.split('Operating Income: ')[1].split('\n')[0]))

        # Add derived features
        df['Customer_Lifetime_Value_Ratio'] = df['Lifetime_Value_of_Customer (₦)'] / df['Customer_Acquisition_Cost (₦)']
        df['Revenue_per_Market_Size'] = df['Revenue_Generated'] / df['Market_Size_Potential']
        df['Expense_Ratio'] = df['Average_Monthly_Expenses'] / df['Revenue_Generated']

        # Feature columns for scaling
        feature_columns = [
            'Revenue_Generated',
            'Growth_Rate (%)',
            'Average_Monthly_Expenses',
            'Customer_Acquisition_Cost (₦)',
            'Lifetime_Value_of_Customer (₦)',
            'Market_Size_Potential',
            'Customer_Lifetime_Value_Ratio',
            'Revenue_per_Market_Size',
            'Expense_Ratio',
            'Net_Profit',
            'Operating_Income'
        ]

        # Scale features
        X = self.feature_scaler.fit_transform(df[feature_columns])

        return (
            X,
            df['Predicted_Revenue_Growth (%)'].values,
            df['Risk_Assessment'].values,
            df['Profitability_Score'].values,
            df['Churn_Rate (%)'].values / 100,  # Convert to proportion
            df['Recommendation_Actions'].values,
            df['Month_Num'].values,
            df['Quarter_Num'].values
        )

def calculate_metrics(outputs, y_risk_batch, y_sug_batch):
    risk_preds = torch.argmax(outputs['risk'], dim=1)
    sug_preds = torch.argmax(outputs['suggestions'], dim=1)

    risk_acc = (risk_preds == y_risk_batch).float().mean()
    sug_acc = (sug_preds == y_sug_batch).float().mean()

    return risk_acc.item(), sug_acc.item()

def find_learning_rate(model, train_loader, device, start_lr=1e-7, end_lr=1, num_iterations=100):
    """
    Implementation of the learning rate range test with better error handling.
    """
    import copy
    init_state = copy.deepcopy(model.state_dict())

    lrs = []
    losses = []
    best_loss = float('inf')

    # Create optimizer with initial learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr)

    # Calculate multiplication factor
    mult = (end_lr / start_lr) ** (1/num_iterations)

    # Loss functions
    regression_criterion = nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss()

    model.train()
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= num_iterations:
            break

        X_batch, y_revenue_batch, y_risk_batch, y_prof_batch, y_churn_batch, y_sug_batch, month_nums, quarter_nums = [
            b.to(device) for b in batch
        ]

        optimizer.zero_grad()

        try:
            outputs = model(X_batch, month_nums, quarter_nums)

            # Calculate losses
            revenue_loss = regression_criterion(outputs['revenue'].squeeze(), y_revenue_batch)
            risk_loss = classification_criterion(outputs['risk'], y_risk_batch)
            prof_loss = regression_criterion(outputs['profitability'].squeeze(), y_prof_batch)
            churn_loss = regression_criterion(outputs['churn'].squeeze(), y_churn_batch)
            sug_loss = classification_criterion(outputs['suggestions'], y_sug_batch)

            total_loss = revenue_loss + risk_loss + prof_loss + churn_loss + sug_loss

            if not torch.isnan(total_loss) and not torch.isinf(total_loss):
                total_loss.backward()
                optimizer.step()

                lrs.append(optimizer.param_groups[0]['lr'])
                losses.append(total_loss.item())

                # Update learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= mult

                # Early stopping if loss explodes
                if total_loss.item() > 4 * best_loss:
                    break
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
            else:
                break

        except RuntimeError as e:
            print(f"Error during LR finding: {e}")
            break

    # Restore initial model state
    model.load_state_dict(init_state)

    if len(losses) > 1:
        # Find the point of steepest descent
        loss_diff = np.diff(losses)
        optimal_idx = np.argmin(loss_diff)
        optimal_lr = lrs[optimal_idx] if optimal_idx < len(lrs) else lrs[-1]
        return optimal_lr / 10, lrs, losses
    else:
        return 1e-4, lrs, losses

def improved_train_model(model, train_loader, val_loader, device, num_epochs=100):
    """
    Enhanced training function with better optimization and monitoring
    Fixed loss handling to maintain computational graph for backpropagation
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4,
        epochs=num_epochs, steps_per_epoch=len(train_loader)
    )

    # Loss functions with label smoothing for classification
    regression_criterion = nn.HuberLoss(delta=1.0)  # More robust than MSE
    classification_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_loss = float('inf')
    best_model_state = None
    patience = 20
    patience_counter = 0

    def calculate_metrics(outputs, targets):
        revenue_batch, risk_batch, prof_batch, churn_batch, sug_batch = targets

        # Calculate accuracies (detached for metrics)
        risk_acc = (outputs['risk'].argmax(dim=1) == risk_batch).float().mean().item()
        churn_acc = ((outputs['churn'].squeeze(1) > 0.5) == churn_batch).float().mean().item()
        sug_acc = (outputs['suggestions'].argmax(dim=1) == sug_batch).float().mean().item()

        # Calculate losses (keeping computational graph)
        losses = {
            'revenue': regression_criterion(outputs['revenue'].squeeze(1), revenue_batch),
            'risk': classification_criterion(outputs['risk'], risk_batch),
            'profitability': regression_criterion(outputs['profitability'].squeeze(1), prof_batch),
            'churn': F.binary_cross_entropy(outputs['churn'].squeeze(1), churn_batch),
            'suggestions': classification_criterion(outputs['suggestions'], sug_batch)
        }

        # Store loss values for metrics
        loss_values = {k: v.item() for k, v in losses.items()}

        accuracies = {
            'risk': risk_acc,
            'churn': churn_acc,
            'suggestions': sug_acc
        }

        return losses, loss_values, accuracies

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        train_metrics = {
            'risk_acc': [], 'churn_acc': [], 'suggestions_acc': [],
            'revenue_loss': [], 'risk_loss': [], 'profitability_loss': [],
            'churn_loss': [], 'suggestions_loss': []
        }

        # Training phase
        for batch in train_loader:
            X_batch, y_revenue_batch, y_risk_batch, y_prof_batch, \
            y_churn_batch, y_sug_batch, month_nums, quarter_nums = [b.to(device) for b in batch]

            optimizer.zero_grad()
            outputs = model(X_batch, month_nums, quarter_nums)

            losses, loss_values, accuracies = calculate_metrics(
                outputs,
                (y_revenue_batch, y_risk_batch, y_prof_batch, y_churn_batch, y_sug_batch)
            )

            # Sum losses while maintaining computational graph
            total_loss = sum(losses.values())
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Record metrics using detached values
            train_losses.append(sum(loss_values.values()))
            for k, v in loss_values.items():
                train_metrics[f'{k}_loss'].append(v)
            for k, v in accuracies.items():
                train_metrics[f'{k}_acc'].append(v)

        scheduler.step()

        # Validation phase
        model.eval()
        val_losses = []
        val_metrics = {
            'risk_acc': [], 'churn_acc': [], 'suggestions_acc': [],
            'revenue_loss': [], 'risk_loss': [], 'profitability_loss': [],
            'churn_loss': [], 'suggestions_loss': []
        }

        with torch.no_grad():
            for batch in val_loader:
                X_batch, y_revenue_batch, y_risk_batch, y_prof_batch, \
                y_churn_batch, y_sug_batch, month_nums, quarter_nums = [b.to(device) for b in batch]

                outputs = model(X_batch, month_nums, quarter_nums)

                _, loss_values, accuracies = calculate_metrics(
                    outputs,
                    (y_revenue_batch, y_risk_batch, y_prof_batch, y_churn_batch, y_sug_batch)
                )

                val_losses.append(sum(loss_values.values()))

                # Record metrics
                for k, v in loss_values.items():
                    val_metrics[f'{k}_loss'].append(v)
                for k, v in accuracies.items():
                    val_metrics[f'{k}_acc'].append(v)

        # Calculate average metrics
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        # Print epoch metrics
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("Training Metrics:")
        print(f"Total Loss: {avg_train_loss:.4f}")
        for k, v in train_metrics.items():
            print(f"{k}: {np.mean(v):.4f}")

        print("\nValidation Metrics:")
        print(f"Total Loss: {avg_val_loss:.4f}")
        for k, v in val_metrics.items():
            print(f"{k}: {np.mean(v):.4f}")

        # Model checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, {'best_val_loss': best_val_loss}

if __name__ == "__main__":
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Generating data...")
    df = generate_business_data(80000)

    print("Processing data...")
    processor = ImprovedBusinessDataProcessor()
    try:
        X, y_revenue, y_risk, y_profitability, y_churn, y_suggestions, month_nums, quarter_nums = processor.process_data(df)
    except Exception as e:
        print(f"Error during data processing: {e}")
        exit(1)

    print("Splitting data...")
    X_train, X_val, y_revenue_train, y_revenue_val, \
    y_risk_train, y_risk_val, y_profitability_train, y_profitability_val, \
    y_churn_train, y_churn_val, y_suggestions_train, y_suggestions_val, \
    month_train, month_val, quarter_train, quarter_val = train_test_split(
        X, y_revenue, y_risk, y_profitability, y_churn, y_suggestions, month_nums, quarter_nums, test_size=0.2, random_state=42
    )

    # Create datasets and loaders
    train_dataset = BusinessDataset(
        X_train, y_revenue_train, y_risk_train, y_profitability_train, y_churn_train,
        y_suggestions_train, month_train, quarter_train
    )
    val_dataset = BusinessDataset(
        X_val, y_revenue_val, y_risk_val, y_profitability_val, y_churn_val,
        y_suggestions_val, month_val, quarter_val
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print("Initializing model...")
    input_dim = X.shape[1]
    num_risk_classes = len(np.unique(y_risk))
    num_suggestion_classes = len(np.unique(y_suggestions))

    model = EnhancedBusinessPredictor(
        input_dim=input_dim,
        num_risk_classes=num_risk_classes,
        num_suggestion_classes=num_suggestion_classes
    ).to(device)

    print("Finding optimal learning rate...")
    optimal_lr, lrs, losses = find_learning_rate(model, train_loader, device)
    print(f"Optimal learning rate found: {optimal_lr}")

    print("Training model...")
    trained_model, metrics = improved_train_model(
        model, train_loader, val_loader, device, num_epochs=100
    )

    print("\nTraining completed!")
    print("Best validation loss:", metrics['best_val_loss'])

    # Save the model
    # Save the model
    model_save_path = "improved_business_predictor.pth"
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'input_dim': input_dim,
        'num_risk_classes': num_risk_classes,
        'num_suggestion_classes': num_suggestion_classes
    }, model_save_path)
    print(f"Model saved at: {model_save_path}")
