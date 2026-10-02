"""Generate Sample Banking Churn Data for Testing"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_RAW

def generate_sample_data(n_samples=10000, random_state=42):
    """
    Generate synthetic banking customer data for testing.
    
    Args:
        n_samples: Number of customer records to generate
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with synthetic customer data
    """
    np.random.seed(random_state)
    
    print(f"Generating {n_samples} sample customer records...")
    
    # Generate customer IDs
    customer_ids = [f"CUST{str(i).zfill(8)}" for i in range(1, n_samples + 1)]
    
    # Generate surnames (sample list)
    surnames = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 
                'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez',
                'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin',
                'Lee', 'Perez', 'Thompson', 'White', 'Harris', 'Sanchez', 'Clark']
    
    # Base distributions
    data = {
        'RowNumber': range(1, n_samples + 1),
        'CustomerId': customer_ids,
        'Surname': np.random.choice(surnames, n_samples),
        'CreditScore': np.random.normal(650, 80, n_samples).astype(int).clip(300, 850),
        'Geography': np.random.choice(['France', 'Spain', 'Germany'], n_samples, p=[0.50, 0.25, 0.25]),
        'Gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.55, 0.45]),
        'Age': np.random.normal(40, 12, n_samples).astype(int).clip(18, 92),
        'Tenure': np.random.randint(0, 11, n_samples),
        'Balance': np.random.exponential(60000, n_samples).clip(0, 250000),
        'NumOfProducts': np.random.choice([1, 2, 3, 4], n_samples, p=[0.50, 0.40, 0.08, 0.02]),
        'HasCrCard': np.random.choice([0, 1], n_samples, p=[0.30, 0.70]),
        'IsActiveMember': np.random.choice([0, 1], n_samples, p=[0.49, 0.51]),
        'EstimatedSalary': np.random.uniform(10000, 200000, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate churn with realistic patterns
    churn_probability = 0.15  # Base churn rate
    
    # Adjust churn probability based on features
    churn_score = np.zeros(n_samples)
    
    # Geography impact (Germany higher churn)
    churn_score += (df['Geography'] == 'Germany').astype(int) * 0.15
    
    # Age impact (middle-aged higher churn)
    churn_score += ((df['Age'] >= 40) & (df['Age'] <= 60)).astype(int) * 0.08
    
    # Tenure impact (newer customers higher churn)
    churn_score += (df['Tenure'] <= 2).astype(int) * 0.12
    
    # Balance impact (zero balance very high churn)
    churn_score += (df['Balance'] == 0).astype(int) * 0.25
    churn_score += ((df['Balance'] > 0) & (df['Balance'] < 50000)).astype(int) * 0.05
    
    # Product impact (single product higher churn)
    churn_score += (df['NumOfProducts'] == 1).astype(int) * 0.10
    
    # Activity impact (inactive much higher churn)
    churn_score += (df['IsActiveMember'] == 0).astype(int) * 0.15
    
    # Gender impact (slight difference)
    churn_score += (df['Gender'] == 'Female').astype(int) * 0.05
    
    # Add random noise
    churn_score += np.random.normal(0, 0.05, n_samples)
    
    # Convert to probability and generate binary outcome
    churn_probability = 1 / (1 + np.exp(-churn_score * 5))  # Logistic function
    df['Exited'] = (np.random.random(n_samples) < churn_probability).astype(int)
    
    # Round numerical columns
    df['Balance'] = df['Balance'].round(2)
    df['EstimatedSalary'] = df['EstimatedSalary'].round(2)
    
    # Add some zero balances (realistic pattern)
    zero_balance_mask = np.random.random(n_samples) < 0.10
    df.loc[zero_balance_mask, 'Balance'] = 0.0
    
    print(f"\nGenerated data summary:")
    print(f"  Total customers: {len(df)}")
    print(f"  Churned customers: {df['Exited'].sum()} ({df['Exited'].mean()*100:.2f}%)")
    print(f"  Geography distribution: {df['Geography'].value_counts().to_dict()}")
    print(f"  Gender distribution: {df['Gender'].value_counts().to_dict()}")
    print(f"  Average age: {df['Age'].mean():.1f}")
    print(f"  Average balance: €{df['Balance'].mean():,.2f}")
    print(f"  Active members: {df['IsActiveMember'].sum()} ({df['IsActiveMember'].mean()*100:.1f}%)")
    
    return df

def save_sample_data(df, filename='bank_churn.csv'):
    """
    Save generated data to CSV file.
    
    Args:
        df: DataFrame to save
        filename: Output filename
    """
    output_path = DATA_RAW / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"\nSample data saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.2f} KB")
    
    return output_path

def main():
    """
    Main function to generate and save sample data.
    """
    print("="*60)
    print("SAMPLE DATA GENERATOR - European Banking Churn")
    print("="*60)
    
    # Generate data
    df = generate_sample_data(n_samples=10000, random_state=42)
    
    # Save to file
    output_path = save_sample_data(df)
    
    print("\n" + "="*60)
    print("SAMPLE DATA GENERATION COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the data: head data/raw/bank_churn.csv")
    print("2. Train models: python scripts/train_models.py")
    print("3. Launch dashboard: streamlit run app/main.py")
    print("="*60)

if __name__ == "__main__":
    main()
