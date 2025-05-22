import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from models.pcos_patient import PatientData, PCOSType
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataProcessor:
    def __init__(self, data_dir: str):
        """
        Initialize the data processor
        Args:
            data_dir: Directory containing the PCOS patient data files
        """
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and process all data files
        Returns:
            Tuple of (training_data, test_data) DataFrames
        """
        all_data = []
        
        # Process all files in the directory
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.docx'):
                file_path = os.path.join(self.data_dir, filename)
                patient_data = self._process_single_file(file_path)
                if patient_data:
                    all_data.append(patient_data)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Split into training and test sets (90% training, 10% test)
        train_data, test_data = train_test_split(
            df, 
            test_size=0.1, 
            random_state=42
        )
        
        return train_data, test_data
    
    def _process_single_file(self, file_path: str) -> Dict:
        """
        Process a single PCOS patient data file
        Args:
            file_path: Path to the data file
        Returns:
            Dictionary containing processed patient data
        """
        try:
            # Read the file content
            # Note: You'll need to implement the actual file reading logic
            # based on your file format (e.g., docx, txt, etc.)
            
            # Example structure (modify according to your actual file format):
            data = {
                'name': '',
                'age': 0,
                'weight': 0.0,
                'height': 0.0,
                'bmi': 0.0,
                'healthy_weight_range': (0.0, 0.0),
                'water_intake': 0.0,
                'pcos_type': None,  # Changed from pcos_types list to single pcos_type
                'waist_measurement': None
            }
            
            return data
            
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            return None
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data for model training
        Args:
            df: Input DataFrame
        Returns:
            Tuple of (features, labels)
        """
        # Extract features
        features = df[['age', 'weight', 'height', 'bmi', 'water_intake']].values
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Extract and encode labels (PCOS type)
        labels = self.label_encoder.fit_transform(df['pcos_type'].values)
        
        return scaled_features, labels
    
    def save_processed_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame, output_dir: str):
        """
        Save processed data to files
        Args:
            train_data: Training data DataFrame
            test_data: Test data DataFrame
            output_dir: Directory to save processed data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        train_data.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
        test_data.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)
        
        # Save scaler and label encoder for later use
        import joblib
        joblib.dump({
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }, os.path.join(output_dir, 'preprocessors.joblib')) 