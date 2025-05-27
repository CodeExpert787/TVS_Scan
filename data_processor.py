import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from models.pcos_patient import PatientData, Outcome
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import docx
import json
import re

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
        
        if not all_data:
            raise ValueError("No valid data files found in the data directory")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Split into training and test sets (90% training, 10% test)
        train_data, test_data = train_test_split(
            df, 
            test_size=0.1, 
            random_state=42
        )
        
        return train_data, test_data
    
    def _extract_outcome(self, text: str) -> str:
        """Extract PCOS type from text."""
        if "PCOS Rintangan Insulin + PCOS Adrenal" in text:
            return PCOSType.COMBINED.value
        elif "PCOS Rintangan Insulin" in text:
            return PCOSType.INSULIN_RESISTANCE.value
        elif "PCOS Adrenal" in text:
            return PCOSType.ADRENAL.value
        return PCOSType.UNKNOWN.value

    def _extract_healthy_weight_range(self, text: str) -> tuple[float, float]:
        """Extract healthy weight range from text."""
        try:
            # Look for pattern like "45.0 kgs - 60.8 kgs"
            match = re.search(r'(\d+\.?\d*)\s*kgs?\s*-\s*(\d+\.?\d*)\s*kgs?', text)
            if match:
                return (float(match.group(1)), float(match.group(2)))
        except:
            pass
        return (0.0, 0.0)

    def _extract_water_intake(self, text: str) -> float:
        """Extract water intake from text."""
        try:
            # Look for pattern like "3 Litres"
            match = re.search(r'(\d+\.?\d*)\s*Litres?', text)
            if match:
                return float(match.group(1))
        except:
            pass
        return 0.0

    def _process_single_file(self, file_path: str) -> Dict:
        """
        Process a single PCOS patient data file
        Args:
            file_path: Path to the data file
        Returns:
            Dictionary containing processed patient data
        """
        try:
            # Read the docx file
            doc = docx.Document(file_path)
            
            # Extract text from paragraphs
            text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            
            # Initialize data dictionary
            data = {
                'name': '',
                'age': 0,
                'weight': 0.0,
                'height': 0.0,
                'bmi': 0.0,
                'healthy_weight_range': (0.0, 0.0),
                'water_intake': 0.0,
                'outcome': PCOSType.UNKNOWN.value,
                'waist_measurement': 0.0
            }
            
            # Extract PCOS type
            data['outcome'] = self._extract_outcome(text)
            
            # Process each line
            for line in text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                # Extract name
                if line.startswith('Nama'):
                    data['name'] = line.split('Nama')[1].strip()
                
                # Extract age
                elif line.startswith('Umur'):
                    try:
                        data['age'] = int(line.split('Umur')[1].strip())
                    except:
                        pass
                
                # Extract weight
                elif line.startswith('Berat'):
                    try:
                        data['weight'] = float(line.split('Berat')[1].strip().split()[0])
                    except:
                        pass
                
                # Extract height
                elif line.startswith('Height'):
                    try:
                        data['height'] = float(line.split('Height')[1].strip().split()[0])
                    except:
                        pass
                
                # Extract BMI
                elif line.startswith('BMI'):
                    try:
                        data['bmi'] = float(line.split('BMI')[1].strip().split()[0])
                    except:
                        pass
                
                # Extract healthy weight range
                elif 'Berat yang sihat' in line:
                    data['healthy_weight_range'] = self._extract_healthy_weight_range(line)
                
                # Extract water intake
                elif 'Jumlah Air yang perlu diminum' in line:
                    data['water_intake'] = self._extract_water_intake(line)
            
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
        features = df[['age', 'weight', 'height', 'bmi', 'water_intake', 'waist_measurement']].values
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Extract and encode labels (PCOS type)
        labels = self.label_encoder.fit_transform(df['outcome'].values)
        
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