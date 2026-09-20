# Dataset Information

## How to Get the Dataset

1. Click on "Access Dataset" link from the Uniflied Mentor project page
2. Download the European Banking Churn dataset
3. Place the CSV file in this directory (`data/raw/`)
4. The expected filename is `bank_churn.csv` (you can rename it if different)

## Expected Dataset Structure

The dataset should contain customer information with columns like:

### Customer Demographics
- CustomerId: Unique customer identifier
- Geography: Customer's country (France, Germany, Spain)
- Gender: Male/Female
- Age: Customer age

### Account Information
- Tenure: Years with the bank
- Balance: Account balance
- NumOfProducts: Number of bank products used
- HasCrCard: Has credit card (0/1)
- IsActiveMember: Active membership status (0/1)

### Financial Information
- CreditScore: Customer credit score
- EstimatedSalary: Estimated annual salary

### Target Variable
- Exited: Whether customer churned (0/1)

## Once you have the dataset:

1. Place it in `data/raw/bank_churn.csv`
2. Run the notebooks in order (starting with `01_data_exploration.ipynb`)
3. The preprocessing pipeline will clean and prepare the data automatically

## Alternative: Use Sample Data

If you don't have access to the dataset yet, we can generate synthetic data for testing the pipeline.