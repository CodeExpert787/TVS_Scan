import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Tuple
import os

class ModelTrainer:
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = models_dir
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
    def prepare_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Convert patient data into model features."""
        features = [
            data['age'],
            data['weight'],
            data['height'],
            data['bmi'],
            data.get('menstrual_cycle_length', 0),
            len(data.get('symptoms', [])),
            len(data.get('medications', [])),
            len(data.get('allergies', [])),
            len(data.get('dietary_restrictions', [])),
            data.get('sleep_hours', 0),
            data.get('stress_level', 0)
        ]
        return np.array(features).reshape(1, -1)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model with the given data."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
    
    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """Predict PCOS type and probability."""
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        probability = np.max(self.model.predict_proba(features_scaled))
        return prediction, probability
    
    def save_model(self, model_path: str = None) -> None:
        """Save the trained model and scaler."""
        if model_path is None:
            model_path = os.path.join(self.models_dir, 'pcos_model.joblib')
        model_data = {
            'model': self.model,
            'scaler': self.scaler
        }
        joblib.dump(model_data, model_path)
    
    def load_model(self, model_path: str = None) -> None:
        """Load a trained model and scaler."""
        if model_path is None:
            model_path = os.path.join(self.models_dir, 'pcos_model.joblib')
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler'] 