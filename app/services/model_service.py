import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
from app.config.settings import MODEL_PARAMS, MODEL_PATH, SCALER_PATH, TARGET_COLUMN
from app.utils.logger import logger

class ModelService:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None

    def train_model(self, df: pd.DataFrame) -> dict:
        """Train the model and return training metrics"""
        try:
            # Prepare features and target
            if TARGET_COLUMN not in df.columns:
                raise ValueError(f"Target column '{TARGET_COLUMN}' not found in the data")

            X = df.drop(TARGET_COLUMN, axis=1)
            y = df[TARGET_COLUMN]
            
            # Convert categorical variables to numeric
            X = pd.get_dummies(X, drop_first=True)
            y = pd.get_dummies(y, drop_first=True)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Scale the features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Initialize base model
            base_model = RandomForestClassifier(
                random_state=42,
                n_jobs=-1,
                bootstrap=True,
                oob_score=True
            )
            
            # Perform grid search
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=MODEL_PARAMS,
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            
            # Fit the grid search
            grid_search.fit(X_train_scaled, y_train)
            
            # Get best model
            self.model = grid_search.best_estimator_
            self.feature_names = X.columns
            
            # Calculate metrics
            train_accuracy = self.model.score(X_train_scaled, y_train)
            test_accuracy = self.model.score(X_test_scaled, y_test)
            
            # Calculate feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Save the model and scaler
            joblib.dump(self.model, MODEL_PATH)
            joblib.dump(self.scaler, SCALER_PATH)
            
            return {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'feature_importance': feature_importance.to_dict('records'),
                'best_params': grid_search.best_params_
            }
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise

    def load_model(self):
        """Load the trained model and scaler"""
        try:
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, data: pd.DataFrame) -> dict:
        """Make predictions using the trained model"""
        try:
            if self.model is None or self.scaler is None:
                self.load_model()
            
            # Scale the features
            X_scaled = self.scaler.transform(data)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            
            return {
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise 