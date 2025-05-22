import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Any, List, Tuple
import os
from sklearn.metrics import accuracy_score, classification_report

class ModelTrainer:
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = models_dir
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
    def prepare_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Convert patient data into model features."""
        features = [
            data['age'],
            data['weight'],
            data['height'],
            data['bmi'],
            data.get('water_intake', 0),
            data.get('waist_measurement', 0)
        ]
        return np.array(features).reshape(1, -1)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train the model with the given data."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        # Calculate training accuracy
        train_predictions = self.model.predict(X_scaled)
        train_accuracy = np.mean(train_predictions == y)
        
        return {
            'train_accuracy': train_accuracy
        }
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate the model on test data."""
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'classification_report': report
        }
    
    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """Predict PCOS type and probability."""
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        probability = np.max(self.model.predict_proba(features_scaled))
        
        # Convert numeric prediction back to original label
        original_label = self.label_encoder.inverse_transform([prediction])[0]
        
        return original_label, probability
    
    def save_model(self, model_path: str = None) -> None:
        """Save the trained model and preprocessors."""
        if model_path is None:
            model_path = os.path.join(self.models_dir, 'pcos_model.joblib')
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }
        joblib.dump(model_data, model_path)
    
    def load_model(self, model_path: str = None) -> None:
        """Load a trained model and preprocessors."""
        if model_path is None:
            model_path = os.path.join(self.models_dir, 'pcos_model.joblib')
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder'] 