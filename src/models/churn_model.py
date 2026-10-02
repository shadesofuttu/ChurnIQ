"""Churn Prediction Model Training and Evaluation"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, accuracy_score,
    precision_score, recall_score
)
from imblearn.over_sampling import SMOTE
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import MODELS_DIR, RANDOM_STATE, TEST_SIZE, CV_FOLDS

class ChurnPredictor:
    """
    Train and evaluate churn prediction models.
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the churn predictor.
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'logistic')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.metrics = {}
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=RANDOM_STATE
            )
        elif model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            )
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'Exited') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for modeling.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
        
        Returns:
            Tuple of (X, y)
        """
        # Drop non-predictive columns
        drop_cols = [target_col, 'CustomerId', 'Surname', 'RowNumber']
        drop_cols = [col for col in drop_cols if col in df.columns]
        
        X = df.drop(columns=drop_cols)
        y = df[target_col] if target_col in df.columns else None
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        return X, y
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              use_smote: bool = True, tune_hyperparameters: bool = False) -> None:
        """
        Train the churn prediction model.
        
        Args:
            X_train: Training features
            y_train: Training target
            use_smote: Whether to use SMOTE for class balancing
            tune_hyperparameters: Whether to perform hyperparameter tuning
        """
        # Store feature columns if not already set
        if self.feature_columns is None:
            self.feature_columns = X_train.columns.tolist()
        
        # Scale features FIRST (fit on original training data)
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # IMPORTANT: Cross-validation BEFORE SMOTE to avoid data leakage
        # CV on original (imbalanced) training data
        print(f"Running cross-validation on original training data...")
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, 
            cv=CV_FOLDS, scoring='roc_auc'
        )
        self.metrics['cv_auc_mean'] = cv_scores.mean()
        self.metrics['cv_auc_std'] = cv_scores.std()
        print(f"Cross-validation AUC (pre-SMOTE): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # NOW apply SMOTE only for final training
        if use_smote:
            smote = SMOTE(random_state=RANDOM_STATE)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
            print(f"Applied SMOTE - New training size: {len(y_train)}")
        
        # Hyperparameter tuning (if requested)
        if tune_hyperparameters:
            print("Tuning hyperparameters...")
            self.model = self._tune_hyperparameters(X_train_scaled, y_train)
        
        # Train final model
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train_scaled, y_train)
    
    def _tune_hyperparameters(self, X_train, y_train):
        """Perform grid search for hyperparameter tuning."""
        if self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            }
        elif self.model_type == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        else:  # logistic
            param_grid = {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model performance with comprehensive metrics.
        
        Args:
            X_test: Test features
            y_test: Test target
        
        Returns:
            Dictionary of evaluation metrics
        """
        # Scale test data
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate ROC curve data
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        
        # Calculate metrics
        self.metrics.update({
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0),
            'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds},
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        })
        
        print("\n=== Model Evaluation ===")
        print(f"Accuracy:  {self.metrics['accuracy']:.4f}")
        print(f"Precision: {self.metrics['precision']:.4f}")
        print(f"Recall:    {self.metrics['recall']:.4f}")
        print(f"F1 Score:  {self.metrics['f1_score']:.4f}")
        print(f"ROC AUC:   {self.metrics['roc_auc']:.4f}")
        print("\nConfusion Matrix:")
        print(self.metrics['confusion_matrix'])
        
        return self.metrics
    
    def predict_churn_probability(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict churn probability for new data.
        
        Args:
            X: Input features
        
        Returns:
            Array of churn probabilities
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with feature importance
        """
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            return importance_df
        else:
            print("Feature importance not available for this model type")
            return None
    
    def save_model(self, model_path: str = None) -> None:
        """
        Save trained model and scaler.
        
        Args:
            model_path: Path to save model (default: config.CHURN_MODEL)
        """
        if model_path is None:
            model_path = MODELS_DIR / f"churn_model_{self.model_type}.pkl"
            scaler_path = MODELS_DIR / "scaler.pkl"
        else:
            model_path = Path(model_path)
            scaler_path = model_path.parent / "scaler.pkl"
        
        # Create directory if it doesn't exist
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Validate feature_columns before saving
        if self.feature_columns is None:
            raise ValueError("feature_columns is None. Ensure prepare_features() or train() is called before save_model().")
        
        if not isinstance(self.feature_columns, (list, tuple)):
            raise TypeError(f"feature_columns must be a list or tuple, got {type(self.feature_columns)}")
        
        # Save model and scaler
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Save feature columns
        feature_path = model_path.parent / "feature_columns.txt"
        with open(feature_path, 'w') as f:
            f.write('\n'.join(self.feature_columns))
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path: str = None) -> None:
        """
        Load trained model and scaler.
        
        Args:
            model_path: Path to load model from
        """
        if model_path is None:
            model_path = MODELS_DIR / f"churn_model_{self.model_type}.pkl"
            scaler_path = MODELS_DIR / "scaler.pkl"
        else:
            model_path = Path(model_path)
            scaler_path = model_path.parent / "scaler.pkl"
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Load feature columns
        feature_path = model_path.parent / "feature_columns.txt"
        with open(feature_path, 'r') as f:
            self.feature_columns = [line.strip() for line in f]
        
        print(f"Model loaded from {model_path}")

def compare_models(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """
    Compare performance of different model types.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
    
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for model_type in ['logistic', 'random_forest', 'gradient_boosting']:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()} model")
        print(f"{'='*50}")
        
        predictor = ChurnPredictor(model_type=model_type)
        # Ensure feature columns are set from X_train
        predictor.feature_columns = X_train.columns.tolist()
        predictor.train(X_train, y_train, use_smote=True)
        metrics = predictor.evaluate(X_test, y_test)
        
        results.append({
            'Model': model_type,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1_score'],
            'ROC AUC': metrics['roc_auc'],
            'CV AUC': metrics['cv_auc_mean']
        })
    
    comparison_df = pd.DataFrame(results)
    print("\n=== Model Comparison ===")
    print(comparison_df.to_string(index=False))
    
    return comparison_df