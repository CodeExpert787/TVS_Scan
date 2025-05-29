import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from models.pcos_patient import PatientData, Outcome
import os
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
        
    def load_data(self) -> List[Dict]:
        """
        Load and process all data files
        Returns:
            List of dictionaries containing patient data
        """
        all_data = []
        
        # Process Excel file first
        excel_path = os.path.join(self.data_dir, 'PCOS Profilling(Responses).xlsx')
        if os.path.exists(excel_path):
            try:
                df = pd.read_excel(excel_path)
                # Convert DataFrame to list of dictionaries
                excel_data = df.to_dict('records')
                all_data.extend(excel_data)
            except Exception as e:
                print(f"Error processing Excel file: {str(e)}")
        
        # Process docx files
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.docx'):
                file_path = os.path.join(self.data_dir, filename)
                patient_data = self._process_single_file(file_path)
                if patient_data:
                    all_data.append(patient_data)
        
        if not all_data:
            raise ValueError("No valid data files found in the data directory")
        
        return all_data
    
    def get_excel_columns(self) -> List[str]:
        """
        Get column names from the Excel file
        Returns:
            List of column names
        """
        excel_path = os.path.join(self.data_dir, 'PCOS Profilling(Responses).xlsx')
        if os.path.exists(excel_path):
            try:
                df = pd.read_excel(excel_path)
                return df.columns.tolist()
            except Exception as e:
                print(f"Error reading Excel columns: {str(e)}")
        return []
    
    def get_excel_preview(self, n_rows: int = 5) -> Dict:
        """
        Get preview of Excel data
        Args:
            n_rows: Number of rows to preview
        Returns:
            Dictionary containing preview data and column info
        """
        excel_path = os.path.join(self.data_dir, 'PCOS Profilling(Responses).xlsx')
        if os.path.exists(excel_path):
            try:
                df = pd.read_excel(excel_path)
                preview = {
                    'columns': df.columns.tolist(),
                    'data': df.head(n_rows).to_dict('records'),
                    'total_rows': len(df),
                    'total_columns': len(df.columns)
                }
                return preview
            except Exception as e:
                print(f"Error getting Excel preview: {str(e)}")
        return {}
    
    def _extract_outcome(self, text: str) -> str:
        """Extract PCOS type from text."""
        if "PCOS Rintangan Insulin + PCOS Adrenal" in text:
            return "Combined"
        elif "PCOS Rintangan Insulin" in text:
            return "Rintangan Insulin"
        elif "PCOS Adrenal" in text:
            return "Adrenal"
        return "Unknown"

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
                'outcome': 'Unknown',
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