from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from models.pcos_patient import PatientData, Outcome
from meal_plan_generator import MealPlanGenerator
from data_processor import DataProcessor
from typing import List, Dict, Optional
import uvicorn
import os
import shutil
import logging
from datetime import datetime, date, time
from starlette.middleware.sessions import SessionMiddleware
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import uuid
import pickle

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="PCOS Data Analysis Platform",
    description="Advanced data analytics and visualization platform for PCOS research",
    version="1.0.0"
)

# Add session middleware
app.add_middleware(SessionMiddleware, secret_key="your-secret-key-here")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Initialize components
meal_plan_generator = MealPlanGenerator()
data_processor = DataProcessor('data/raw')

# Create necessary directories
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

def save_upload_file(upload_file: UploadFile, destination: str):
    """Save uploaded file to destination"""
    try:
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()

# Training routes
@app.get("/training", response_class=HTMLResponse)
async def get_training_page(request: Request):
    """Serve the training page"""
    return templates.TemplateResponse("training.html", {"request": request})

@app.post("/training/upload")
async def upload_training_data(file: UploadFile = File(...)):
    try:
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Save the uploaded file
        file_path = f"data/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        # Clean column names
        df.columns = [str(col).replace('/', '_').strip() for col in df.columns]
        
        # Extract field names
        field_names = {
            "fields": df.columns.tolist(),
            "total_fields": len(df.columns),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save field names to JSON
        fields_file = "data/field_names.json"
        with open(fields_file, "w", encoding="utf-8") as f:
            json.dump(field_names, f, indent=4, ensure_ascii=False)
        
        # Convert DataFrame to records with proper NaN handling
        def clean_value(val):
            if pd.isna(val):
                return None
            if isinstance(val, (pd.Timestamp, datetime)):
                return val.strftime('%Y-%m-%d %H:%M:%S')
            if isinstance(val, time):
                return val.strftime('%H:%M:%S')
            if isinstance(val, date):
                return val.strftime('%Y-%m-%d')
            return val

        records = []
        for _, row in df.iterrows():
            record = {}
            for col in df.columns:
                record[col] = clean_value(row[col])
            records.append(record)
        
        return {
            "success": True,
            "message": "File uploaded and processed successfully",
            "data": records,
            "fields": field_names
        }
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return {
            "success": False,
            "error": f"Error processing file: {str(e)}"
        }

@app.post("/training/start")
async def start_training(request: Request):
    """Start model training process"""
    try:
        data = await request.json()
        filename = data.get('filename')
        
        if not filename:
            raise ValueError("No filename provided")
            
        # Load the data from the uploaded file
        file_path = os.path.join("data", filename)
        if not os.path.exists(file_path):
            # Try to find the file in data/raw directory
            raw_file_path = os.path.join("data/raw", filename)
            if os.path.exists(raw_file_path):
                file_path = raw_file_path
            else:
                # List available files for debugging
                available_files = []
                for root, dirs, files in os.walk("data"):
                    for file in files:
                        if file.endswith('.xlsx') or file.endswith('.xls'):
                            available_files.append(os.path.join(root, file))
                raise ValueError(f"File not found: {filename}. Available files: {available_files}")
            
        logger.info(f"Loading file from: {file_path}")
        
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        # Clean column names
        df.columns = [str(col).replace('/', '_').strip() for col in df.columns]
        
        # Handle datetime columns
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                # Convert datetime to string format
                df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            elif pd.api.types.is_object_dtype(df[col]):
                # Convert any remaining datetime objects to string
                df[col] = df[col].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if isinstance(x, (pd.Timestamp, datetime)) else x)
        
        # Prepare features and target
        target_column = "Jika anda sudah membuat Quiz Jenis PCOS, sila nyatakan Jenis PCOS anda._ If you had done the PCOS Type Quiz, please state your PCOS Type"
        if target_column not in df.columns:
            available_columns = df.columns.tolist()
            raise ValueError(f"Target column '{target_column}' not found in the data. Available columns: {available_columns}")
            
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Convert categorical variables to numeric
        X = pd.get_dummies(X, drop_first=True)
        y = pd.get_dummies(y, drop_first=True)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define simplified parameter grid for faster training
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5]
        }
        
        # Initialize base model with optimized parameters
        base_model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,  # Use all available CPU cores
            bootstrap=True,
            oob_score=True  # Enable out-of-bag score
        )
        
        # Perform grid search with reduced cross-validation folds
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,  # Reduced from 5 to 3 folds
            scoring='accuracy',
            n_jobs=-1,  # Use all available CPU cores
            verbose=0
        )
        
        # Fit the grid search
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best model
        model = grid_search.best_estimator_
        
        # Calculate metrics
        train_accuracy = model.score(X_train_scaled, y_train)
        test_accuracy = model.score(X_test_scaled, y_test)
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save the model and scaler
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", "pcos_model.joblib")
        scaler_path = os.path.join("models", "scaler.joblib")
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        # Save feature names and importance
        model_info = {
            'feature_names': X.columns.tolist(),
            'feature_importance': feature_importance.to_dict('records'),
            'best_params': grid_search.best_params_,
            'cv_scores': grid_search.cv_results_['mean_test_score'].tolist()
        }
        
        # Save model info and feature names
        with open(os.path.join("models", "model_info.json"), 'w') as f:
            json.dump(model_info, f, indent=4)
            
        # Save feature names separately for easy access
        with open(os.path.join("models", "feature_names.json"), 'w') as f:
            json.dump(X.columns.tolist(), f, indent=4)
        
        # Prepare detailed response
        response = {
            "success": True,
            "progress": 100,
            "message": f"Model training completed successfully",
            "metrics": {
                "training_accuracy": f"{train_accuracy:.2%}",
                "test_accuracy": f"{test_accuracy:.2%}",
                "best_parameters": grid_search.best_params_,
                "feature_importance": feature_importance.head(10).to_dict('records'),
                "cross_validation_scores": {
                    "mean": f"{grid_search.cv_results_['mean_test_score'].mean():.2%}",
                    "std": f"{grid_search.cv_results_['mean_test_score'].std():.2%}"
                }
            }
        }
        
        return JSONResponse(content=response)
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return JSONResponse(content={
            "success": False,
            "error": str(e)
        })

# Analysis routes
@app.get("/analyze", response_class=HTMLResponse)
async def analyze_page(request: Request):
    try:
        # Load field names from JSON file
        field_names_path = os.path.join('data', 'field_names.json')
        if not os.path.exists(field_names_path):
            return templates.TemplateResponse("analyze.html", {
                "request": request,
                "error": "Please upload and train a model first."
            })
        
        with open(field_names_path, 'r', encoding='utf-8') as f:
            field_data = json.load(f)
            questions = field_data.get('fields', [])
        
        if not questions:
            return templates.TemplateResponse("analyze.html", {
                "request": request,
                "error": "No questions available. Please upload and train a model first."
            })
        
        return templates.TemplateResponse("analyze.html", {
            "request": request,
            "questions": questions
        })
    except Exception as e:
        logger.error(f"Error loading questions: {str(e)}")
        return templates.TemplateResponse("analyze.html", {
            "request": request,
            "error": "Error loading questions. Please ensure training data is available."
        })

@app.post("/analyze")
async def analyze(request: Request):
    try:
        form_data = await request.form()
        data = dict(form_data)
        
        # Get PCOS type from the form data
        pcos_type_field = "Jika anda sudah membuat Quiz Jenis PCOS, sila nyatakan Jenis PCOS anda._ If you had done the PCOS Type Quiz, please state your PCOS Type"
        pcos_type = data.get(pcos_type_field)
        
        if not pcos_type:
            raise ValueError("PCOS type not provided")
            
        # Load the model and scaler
        model_path = os.path.join("models", "pcos_model.joblib")
        scaler_path = os.path.join("models", "scaler.joblib")
        model_info_path = os.path.join("models", "model_info.json")
        
        if not all(os.path.exists(path) for path in [model_path, scaler_path, model_info_path]):
            raise ValueError("Model files not found. Please train the model first.")
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Load feature names from model_info.json
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
            feature_names = model_info.get('feature_names', [])
            
        if not feature_names:
            raise ValueError("Feature names not found in model info")
        
        # Create a dictionary with all features initialized to 0
        input_dict = {feature: 0 for feature in feature_names}
        
        # Update with actual input values
        for key, value in data.items():
            if key in feature_names:
                input_dict[key] = value
        
        # Ensure the input data columns match the expected feature order
        input_data = pd.DataFrame([input_dict])
        input_scaled = scaler.transform(input_data)

        # Calculate BMI, Healthy Weight, and Daily Water Intake
        try:
            weight_kg = float(data.get('Berat_ Weight (kg)', 0))
            height_cm = float(data.get('Tinggi _Height (cm)', 0))

            if height_cm > 0:
                height_m = height_cm / 100
                bmi = weight_kg / (height_m ** 2)
                healthy_weight_min = 18.5 * (height_m ** 2)
                healthy_weight_max = 24.9 * (height_m ** 2)
                healthy_weight = f"{healthy_weight_min:.1f} kgs - {healthy_weight_max:.1f} kgs"
            else:
                bmi = 0
                healthy_weight = "N/A"

            daily_water_min_ml = weight_kg * 30
            daily_water_max_ml = weight_kg * 35
            daily_water = f"{daily_water_min_ml/1000:.1f} liters - {daily_water_max_ml/1000:.1f} liters"

            # Add calculated values to data passed to template
            data['BMI'] = f"{bmi:.1f} kg/m2"
            data['Berat yang sihat'] = healthy_weight
            data['Jumlah Air yang perlu diminum sepanjang hari'] = daily_water

        except ValueError:
            logger.error("Error calculating BMI, Healthy Weight, or Daily Water Intake. Invalid input values.")
            data['BMI'] = "Invalid Input"
            data['Berat yang sihat'] = "Invalid Input"
            data['Jumlah Air yang perlu diminum sepanjang hari'] = "Invalid Input"

        # Make prediction
        prediction_proba = model.predict_proba(input_scaled)
        prediction = model.predict(input_scaled)
        
        # Debug logging
        logger.info(f"Prediction shape: {prediction.shape}")
        logger.info(f"Prediction type: {type(prediction)}")
        logger.info(f"Prediction content: {prediction}")
        logger.info(f"Prediction proba type: {type(prediction_proba)}")
        logger.info(f"Prediction proba content: {prediction_proba}")
        
        # Handle prediction - it's a 2D array with shape (1, 4)
        if isinstance(prediction, np.ndarray):
            # Get the index of True value in the first row
            pred_idx = np.where(prediction[0])[0][0]
            prediction = pred_idx
        else:
            prediction = int(prediction)
        
        # Handle probability
        if isinstance(prediction_proba, list):
            # Get the probability array for the predicted class
            proba_array = prediction_proba[prediction]
            # Get the probability of the positive class (index 1)
            probability = float(proba_array[0][1])
        else:
            probability = float(prediction_proba)
        
        # Generate meal plan based on PCOS type
        meal_plan = generate_meal_plan(pcos_type)
        
        return templates.TemplateResponse("results.html", {
            "request": request,
            "prediction": prediction,
            "probability": probability,
            "input_data": data,
            "pcos_type": pcos_type,
            "meal_plan": meal_plan
        })
    except Exception as e:
        logger.error(f"Error analyzing data: {str(e)}")
        logger.error(f"Error details:", exc_info=True)  # Add full traceback
        return templates.TemplateResponse("analyze.html", {
            "request": request,
            "error": str(e)
        })

def generate_meal_plan(pcos_type):
    """Generate meal plan based on PCOS type"""
    meal_plans = {
        "PCOS Rintangan Insulin/Insulin Resistance": {
            "title": "PCOS Rintangan Insulin/Insulin Resistance",
            "treatments": {
                "month1": [
                    "Rawatan 1# Sarapan Bebas PCOS",
                    "Rawatan 2# Diet Bebas PCOS",
                    "Rawatan 3# Mengurangkan pengambilan Fructose",
                    "Rawatan 4# Pilih Masa Makan anda",
                    "Rawatan 5# Supplement - amalkan Feminira untuk 3 bulan -Seimbangkan hormon dan tingkatkan kualiti telur",
                    "Rawatan 6# Senaman"
                ],
                "month2": [
                    "PCOS Adrenal",
                    "Rawatan #1: Bebaskan dari Stress",
                    "Rawatan #2: Buat Ritual Pagi yang Mengurangkan Stress",
                    "Rawatan #3: Luangkan Masa Untuk Aktiviti yang anda nikmati",
                    "Rawatan #4: Buang atau Kurangkan Kafein",
                    "Rawatan #5: Nilai Tahap Senaman Anda",
                    "Rawatan #6: Seimbangkan Melatonin dan Kortisol Anda",
                    "Rawatan #7: Pertimbangkan Campuran Herba Mengurangkan Stress (Feminira)"
                ]
            },
            "breakfast": {
                "title": "Sarapan Pagi Bebas PCOS",
                "instructions": [
                    "WAJIB ambil sarapan satu jam selepas bangun tidur, dan TIDAK mengambil sebarang bahan yang mengandungi gula (Cthnya: Madu, Stevia, gula etc).",
                    "Jika anda mengambil kopi sila kurangkan atau berhenti kopi untuk mengatasi masalah PCOS Adrenal untuk mengelakkan kenaikkan hormon kortisol. Boleh gantikan dengan green tea or decaf",
                    "Objektif utama sarapan PCOS adalah mengambil 30-40g clean protein pada waktu pagi untuk menstabilkan gula dalam darah"
                ],
                "options": [
                    {
                        "title": "Pilihan 1",
                        "items": ["4 biji telur putih + 1 kuning telur (Rebus/Separuh masak/scramble)"]
                    },
                    {
                        "title": "Pilihan 2",
                        "items": ["150g dada ayam (Bakar)- Boleh marinate dengan garam dan lada hitam atau sebarang rempah"]
                    },
                    {
                        "title": "Pilihan 3",
                        "items": ["150g Ayam Brand Tuna Chunk in water (Perah satu biji limau dan makan)"]
                    },
                    {
                        "title": "Pilihan 4",
                        "items": ["200g Tempe (Goreng dengan sedikit minyak/airfryer)"]
                    },
                    {
                        "title": "Pilihan 5",
                        "items": [
                            "Boleh buat waktu malam dan makan untuk sarapan. Jika tiada berries tiada masalah, tapi JANGAN gantikan dengan buahan lain.",
                            "150ml almond milk/coconut milk",
                            "¼ cup chia seeds",
                            "1 scoop protein (Tiada gula)",
                            "½ cup of fresh or frozen berries (e.g. raspberries, strawberries or blueberries)"
                        ]
                    },
                    {
                        "title": "Pilihan 6",
                        "items": [
                            "Berries smoothies",
                            "½ cup frozen raspberries",
                            "1 scoop protein- (Tiada gula)",
                            "150ml coconut/almond milk"
                        ]
                    }
                ]
            },
            "mediterranean_diet": {
                "title": "Diet Mediterranean (Makan tengah hari & Makan malam)",
                "guidelines": [
                    "Protein - sasarkan jumlah sebesar tapak tangan (saiz dan ketebalan badan anda tapak tangan)",
                    "Karbohidrat- ambil lebih kurang ½ cawan selepas masak",
                    "Sayur-sayuran yang tidak berkanji ambil sebanyak yang boleh"
                ],
                "menu": {
                    "items": [
                        "120g ayam/ikan/daging",
                        "120g nasi/kentang (Jika boleh cuba tukar kepada brown rice/basmati (low GI)/wholemeal bread)",
                        "240g sayur-sayuran tidak berkanji",
                        "20g Kacang (Almond/Walnut)/10g Dark Chocolate 80% keatas/30g Chia Seed",
                        "1 biji buah epal/oren (Jika rasa nak ambil makanan manis)"
                    ],
                    "notes": [
                        "Cara masakkan kurang minyak dan tidak menggunakan gula. Elakkan mengambil makanan fast-food dan sebarang gula untuk tempoh empat minggu pertama.",
                        "p/s: Jika makan diluar, sila minta makanan kurang minyak dan tiada ajinamoto.",
                        "4 minggu pertama elakkan mengambil fast food."
                    ]
                },
                 "anti_inflammatory_foods":{
                    "title": "Makanan yang baik diambil (Makanan Anti-radang)",
                    "items": [
                        "Ikan berlemak seperti salmon, sardin dan makarel ini mengandungi tahap asid lemak omega-3 yang tinggi, yang merendahkan keradangan serta menyokong keseimbangan hormon yang sihat dan mood yang stabil",
                        "Biji chia dan biji rami tanah ini kaya dengan ALA– pendahulu berasaskan tumbuhan kepada omega-3",
                        "Minyak zaitun ini mempunyai sebatian anti-radang seperti oleocanthal",
                        "Beri merah gelap seperti strawberry, raspberi, beri biru, dan beri hitam ini tinggi dengan antioksidan yang melawan radikal bebas",
                        "Sayur-sayuran berdaun gelap seperti kangkung, bayam, bok choy dan silverbeet ini penuh dengan banyak nutrien termasuk vitamin K, kalsium dan vitamin B",
                        "Kacang seperti badam, walnut, kacang Brazil, dan biji seperti biji labu dan biji bunga matahari untuk tahap vitamin E dan selenium yang bermanfaat.",
                        "Sayuran cruciferous seperti brokoli, kembang kol, kubis, dan pucuk Brussels yang mengandungi sebatian yang menyokong detoksifikasi hormon",
                        "Herba dan rempah ratus seperti halia, kunyit, kayu manis, bawang putih, lada hitam, dan rosemary ini membantu menjadikan rasa makanan lebih enak dan juga mengandungi sebatian anti-radang."
                    ]
                 },
                 "avoid_foods": {
                    "title": "Makanan yang perlu dielakkan untuk rawatan keradangan",
                    "items": [
                        "Makanan yang mengandung gluten seperti apa saja jenis roti, jadi gantikan dengan brown rice",
                        "Elakan Produk susu lembu (Boleh ambil susu kambing/susu almond/susu kelapa)",
                        "Makanan Diproses dan Bergoreng"
                    ]
                 }
            },
            "meal_timing": {
                "title": "Masa makan",
                "instructions": "Target untuk puasa selama 12 jam dimana jarak dari makan malam ke sarapan pagi",
                "schedule": [
                    "Sarapan pagi - 1 jam selepas bangun tidur",
                    "Makan tengah hari - 12 tgh hari atau 1 petang",
                    "Makan malam 7mlm-8mlm"
                ]
            },
            "waist_measurement": {
                "title": "Ukur Lilitan Pinggang",
                "instructions": "Ukur Lilitan Pinggang sebelum mula diet plan: __________inchi",
                "weeks": [
                    "Minggu Pertama",
                    "Minggu Kedua",
                    "Minggu Ketiga",
                    "Minggu Keempat"
                ]
            },
            "exercise_plan": {
                "title": "Plan Senaman:",
                "options": [
                    [
                        "Lakukan brist walk 2-3 kali seminggu dengan melengkapkan 10,000 steps",
                        "Untuk aktiviti harian cuba banyakkan berjalan dengan parking lebih jauh dan capai lebih daripada 5000 step sehari"
                    ],
                    [
                         "Lakukan 20 minute senaman HIIT di rumah 2-3 Minggu sekali-Contoh senaman sila rujuk ebook Protokol Bebas PCOS."
                    ]
                ]
            }
        },
        "PCOS Adrenal": {
            "title": "PCOS Adrenal",
            "treatments": {
                "month1": [
                    "Rawatan #1: Bebaskan dari Stress",
                    "Rawatan #2: Buat Ritual Pagi yang Mengurangkan Stress",
                    "Rawatan #3: Luangkan Masa Untuk Aktiviti yang anda nikmati",
                    "Rawatan #4: Buang atau Kurangkan Kafein",
                    "Rawatan #5: Nilai Tahap Senaman Anda",
                    "Rawatan #6: Seimbangkan Melatonin dan Kortisol Anda",
                    "Rawatan #7: Pertimbangkan Campuran Herba Mengurangkan Stress (Feminira)"
                ],
                "month2": [
                    "PCOS Keradangan",
                    "Rawatan #1: Sembuhkan Lapisan Usus Anda",
                    "Rawatan #2: Amalkan Makanan Anti Radang",
                    "Rawatan #3: Pertimbangkan Suplemen Anti Radang -Feminira",
                    "Rawatan #4: Seimbangkan Nutrisi - Sarapan Bebas PCOS - Diet Mediterranean",
                    "Rawatan #5: Kurangkan Pendedahan kepada Toksin Persekitaran",
                    "Rawatan #6: Senaman yang sesuai"
                ]
            },
            "breakfast": {
                "title": "Sarapan Pagi Bebas PCOS",
                "instructions": [
                    "WAJIB ambil sarapan satu jam selepas bangun tidur, dan TIDAK mengambil sebarang bahan yang mengandungi gula (Cthnya: Madu, Stevia, gula etc).",
                    "Jika anda mengambil kopi sila kurangkan atau berhenti kopi untuk mengatasi masalah PCOS Adrenal untuk mengelakkan kenaikkan hormon kortisol",
                    "Objektif utama sarapan PCOS adalah mengambil 30-40g clean protein pada waktu pagi untuk menstabilkan gula dalam darah"
                ],
                "options": [
                    {
                        "title": "Pilihan 1",
                        "items": ["4 biji telur putih + 1 kuning telur (Rebus/Separuh masak/scramble)"]
                    },
                    {
                        "title": "Pilihan 2",
                        "items": ["150g dada ayam (Bakar)- Boleh marinate dengan garam dan lada hitam atau sebarang rempah"]
                    },
                    {
                        "title": "Pilihan 3",
                        "items": ["150g Ayam Brand Tuna Chunk in water (Perah satu biji limau dan makan)"]
                    },
                    {
                        "title": "Pilihan 4",
                        "items": ["200g Tempe (Goreng dengan sedikit minyak/airfryer)"]
                    },
                    {
                        "title": "Pilihan 5",
                        "items": [
                            "Boleh buat waktu malam dan makan untuk sarapan. Jika tiada berries tiada masalah, tapi JANGAN gantikan dengan buahan lain.",
                            "150ml almond milk/coconut milk",
                            "¼ cup chia seeds",
                            "1 scoop protein (Tiada gula)",
                            "½ cup of fresh or frozen berries (e.g. raspberries, strawberries or blueberries)"
                        ]
                    },
                    {
                        "title": "Pilihan 6",
                        "items": [
                            "Berries smoothies",
                            "½ cup frozen raspberries",
                            "1 scoop protein- (Tiada gula)",
                            "150ml coconut/almond milk"
                        ]
                    }
                ]
            },
            "mediterranean_diet": {
                "title": "Diet Mediterranean (Makan tengah hari & Makan malam)",
                "guidelines": [
                    "Protein - sasarkan jumlah sebesar tapak tangan (saiz dan ketebalan badan anda tapak tangan)",
                    "Karbohidrat- ambil lebih kurang ½ cawan selepas masak",
                    "Sayur-sayuran yang tidak berkanji ambil sebanyak yang boleh"
                ],
                "menu": {
                    "items": [
                        "120g ayam/ikan/daging",
                        "120g nasi/kentang (Jika boleh cuba tukar kepada brown rice/basmati (low GI)/wholemeal bread)",
                        "240g sayur-sayuran tidak berkanji",
                        "20g Kacang (Almond/Walnut)/10g Dark Chocolate 80% keatas/30g Chia Seed",
                        "1 biji buah epal/oren (Jika rasa nak ambil makanan manis)"
                    ],
                    "notes": [
                        "Cara masakkan kurang minyak dan tidak menggunakan gula. Elakkan mengambil makanan fast-food dan sebarang gula untuk tempoh empat minggu pertama.",
                        "p/s: Jika makan diluar, sila minta makanan kurang minyak dan tiada ajinamoto.",
                        "4 minggu pertama elakkan mengambil fast food."
                    ]
                },
                 "anti_inflammatory_foods":{
                    "title": "Makanan yang baik diambil (Makanan Anti-radang)",
                    "items": [
                        "Ikan berlemak seperti salmon, sardin dan makarel ini mengandungi tahap asid lemak omega-3 yang tinggi, yang merendahkan keradangan serta menyokong keseimbangan hormon yang sihat dan mood yang stabil",
                        "Biji chia dan biji rami tanah ini kaya dengan ALA– pendahulu berasaskan tumbuhan kepada omega-3",
                        "Minyak zaitun ini mempunyai sebatian anti-radang seperti oleocanthal",
                        "Beri merah gelap seperti strawberry, raspberi, beri biru, dan beri hitam ini tinggi dengan antioksidan yang melawan radikal bebas",
                        "Sayur-sayuran berdaun gelap seperti kangkung, bayam, bok choy dan silverbeet ini penuh dengan banyak nutrien termasuk vitamin K, kalsium dan vitamin B",
                        "Kacang seperti badam, walnut, kacang Brazil, dan biji seperti biji labu dan biji bunga matahari untuk tahap vitamin E dan selenium yang bermanfaat.",
                        "Sayuran cruciferous seperti brokoli, kembang kol, kubis, dan pucuk Brussels yang mengandungi sebatian yang menyokong detoksifikasi hormon",
                        "Herba dan rempah ratus seperti halia, kunyit, kayu manis, bawang putih, lada hitam, dan rosemary ini membantu menjadikan rasa makanan lebih enak dan juga mengandungi sebatian anti-radang."
                    ]
                 },
                 "avoid_foods": {
                    "title": "Makanan yang perlu dielakkan untuk rawatan keradangan",
                    "items": [
                        "Makanan yang mengandung gluten seperti apa saja jenis roti, jadi gantikan dengan brown rice",
                        "Elakan Produk susu lembu (Boleh ambil susu kambing/susu almond/susu kelapa)",
                        "Makanan Diproses dan Bergoreng"
                    ]
                 }
            },
            "meal_timing": {
                "title": "Masa makan",
                "instructions": "Jadual makan yang konsisten untuk mengurangkan stress",
                "schedule": [
                    "Sarapan pagi - 1 jam selepas bangun tidur",
                    "Makan tengah hari - 12 tgh hari",
                    "Makan malam - 7 malam"
                ]
            },
            "waist_measurement": {
                "title": "Ukur Lilitan Pinggang",
                "weeks": [
                    "Minggu Pertama",
                    "Minggu Kedua",
                    "Minggu Ketiga",
                    "Minggu Keempat"
                ]
            },
            "exercise_plan": {
                "title": "Plan Senaman",
                "options": [
                    [
                        "Senaman ringan 2-3 kali seminggu",
                        "Yoga atau stretching setiap hari"
                    ],
                    [
                        "Berjalan 30 minit setiap hari",
                        "Elakkan senaman yang terlalu intensif"
                    ]
                ]
            }
        },
        "PCOS Pos Pil Perancang/Post Birth Control": {
            "title": "PCOS POST Pil Perancang",
            "treatments": {
                "month1_2_3": [
                    "PCOS POST Pil Perancang",
                    "1. Ambil Suplemen untuk menyeimbangkan hormon - Amalkan *Feminira* untuk sekurang-kurangnya 2-3 bulan",
                    "2. Elakkan Susu Lembu",
                    "3. Kurangkan Makanan mengandungi Fruktos yang tinggi",
                    "4. Sokong Kesihatan Usus Anda"
                ]
            },
             "breakfast": {
                "title": "Sarapan Pagi Bebas PCOS",
                "instructions": [
                    "WAJIB ambil sarapan satu jam selepas bangun tidur, dan TIDAK mengambil sebarang bahan yang mengandungi gula (Cthnya: Madu, Stevia, gula etc).",
                    "Objektif utama sarapan PCOS adalah mengambil 30-40g clean protein pada waktu pagi untuk menstabilkan gula dalam darah"
                ],
                "options": [
                     {
                        "title": "Pilihan 1",
                        "items": ["4 biji telur putih + 1 kuning telur (Rebus/Separuh masak/scramble)"]
                    },
                    {
                        "title": "Pilihan 2",
                        "items": ["150g dada ayam (Bakar)- Boleh marinate dengan garam dan lada hitam atau sebarang rempah"]
                    },
                    {
                        "title": "Pilihan 3",
                        "items": ["150g Ayam Brand Tuna Chunk in water (Perah satu biji limau dan makan)"]
                    },
                    {
                        "title": "Pilihan 4",
                        "items": ["200g Tempe (Goreng dengan sedikit minyak/airfryer)"]
                    },
                    {
                        "title": "Pilihan 5",
                        "items": [
                            "Boleh buat waktu malam dan makan untuk sarapan. Jika tiada berries tiada masalah, tapi JANGAN gantikan dengan buahan lain.",
                            "150ml almond milk/coconut milk",
                            "¼ cup chia seeds",
                            "1 scoop protein (Tiada gula)",
                            "½ cup of fresh or frozen berries (e.g. raspberries, strawberries or blueberries)"
                        ]
                    },
                    {
                        "title": "Pilihan 6",
                        "items": [
                            "Berries smoothies",
                            "½ cup frozen raspberries",
                            "1 scoop protein- (Tiada gula)",
                            "150ml coconut/almond milk"
                        ]
                    }
                ]
            },
            "mediterranean_diet": {
                "title": "Diet Mediterranean (Makan tengah hari & Makan malam)",
                 "guidelines": [
                    "Protein - sasarkan jumlah sebesar tapak tangan (saiz dan ketebalan badan anda tapak tangan)",
                    "Karbohidrat- ambil lebih kurang ½ cawan selepas masak",
                    "Sayur-sayuran yang tidak berkanji ambil sebanyak yang boleh"
                ],
                "menu": {
                     "items": [
                        "120g ayam/ikan/daging",
                        "120g nasi/kentang (Jika boleh cuba tukar kepada brown rice/basmati (low GI)/wholemeal bread)",
                        "240g sayur-sayuran tidak berkanji",
                        "20g Kacang (Almond/Walnut)/10g Dark Chocolate 80% keatas/30g Chia Seed",
                        "1 biji buah epal/oren (Jika rasa nak ambil makanan manis)"
                    ],
                    "notes": [
                         "Cara masakkan kurang minyak dan tidak menggunakan gula. Elakkan mengambil makanan fast-food dan sebarang gula untuk tempoh empat minggu pertama.",
                        "p/s: Jika makan diluar, sila minta makanan kurang minyak dan tiada ajinamoto.",
                        "4 minggu pertama elakkan mengambil fast food."
                    ]
                },
                "anti_inflammatory_foods":{
                    "title": "Makanan yang baik diambil (Makanan Anti-radang)",
                    "items": [
                         "Ikan berlemak seperti salmon, sardin dan makarel ini mengandungi tahap asid lemak omega-3 yang tinggi, yang merendahkan keradangan serta menyokong keseimbangan hormon yang sihat dan mood yang stable",
                        "Biji chia dan biji rami tanah ini kaya dengan ALA– pendahulu berasaskan tumbuhan kepada omega-3",
                        "Minyak zaitun ini mempunyai sebatian anti-radang seperti oleocanthal",
                        "Beri merah gelap seperti strawberry, raspberi, beri biru, dan beri hitam ini tinggi dengan antioksidan yang melawan radikal bebas",
                        "Sayur-sayuran berdaun gelap seperti kangkung, bayam, bok choy dan silverbeet ini penuh dengan banyak nutrien termasuk vitamin K, kalsium dan vitamin B",
                        "Kacang seperti badam, walnut, kacang Brazil, dan biji seperti biji labu dan biji bunga matahari untuk tahap vitamin E dan selenium yang bermanfaat.",
                        "Sayuran cruciferous seperti brokoli, kembang kol, kubis, dan pucuk Brussels yang mengandungi sebatian yang menyokong detoksifikasi hormon",
                        "Herba dan rempah ratus seperti halia, kunyit, kayu manis, bawang putih, lada hitam, dan rosemary ini membantu menjadikan rasa makanan lebih enak dan juga mengandungi sebatian anti-radang."
                    ]
                 },
                 "avoid_foods": {
                    "title": "Makanan yang perlu dielakkan untuk rawatan keradangan",
                    "items": [
                        "Makanan yang mengandung gluten seperti apa saja jenis roti, jadi gantikan dengan brown rice",
                        "Elakan Produk susu lembu (Boleh ambil susu kambing/susu almond/susu kelapa)",
                        "Makanan Diproses dan Bergoreng"
                    ]
                 }
            },
             "meal_timing": {
                "title": "Masa makan",
                "instructions": "Target untuk puasa selama 12 jam dimana jarak dari makan malam ke sarapan pagi",
                "schedule": [
                    "Sarapan pagi - 1 hour selepas bangun tidur",
                    "Makan tengah hari - 12 tgh hari atau 1 petang",
                    "Makan malam 7mlm-8mlm"
                ]
            },
            "waist_measurement": {
                "title": "Ukur Lilitan Pinggang",
                 "instructions": "Ukur Lilitan Pinggang sebelum mula diet plan: __________inchi",
                "weeks": [
                    "Minggu Pertama",
                    "Minggu Kedua",
                    "Minggu Ketiga",
                    "Minggu Keempat"
                ]
            },
            "exercise_plan": {
                "title": "Plan Senaman:",
                "options": [
                    [
                        "Lakukan brist walk 2-3 kali seminggu dengan melengkapkan 10,000 steps",
                        "Untuk aktiviti harian cuba banyakkan berjalan dengan parking lebih jauh dan capai lebih daripada 5000 step sehari"
                    ],
                    [
                         "Lakukan 20 minute senaman HIIT di rumah 2-3 Minggu sekali-Contoh senaman sila rujuk ebook Protokol Bebas PCOS."
                    ]
                ]
            }
        },
        "PCSO Keradagan/ Infllamantion": {
            "title": "PCOS Keradangan",
            "treatments": {
                "month1_2_3": [
                    "PCOS Keradangan",
                    "1.Sembuhkan Lapisan Usus Anda",
                    "2. Amalkan Makanan Anti Radang",
                    "3. Pertimbangkan Suplemen Anti Radang-Feminira",
                    "4. Seimbangkan Nutrisi",
                    "5. Kurangkan Pendedahan kepada Toksin Persekitaran",
                    "6. Senaman yang sesuai"
                ],
                "recommendation": "Anda di nasihatkan untuk membuat ujian blood test Hormon Dehydroepiandrosterone sulfate (DHEAS) untuk mengesahkan masalah PCOS Adrenal anda."
            },
            "breakfast": {
                "title": "Sarapan Pagi Bebas PCOS",
                "instructions": [
                    "WAJIB ambil sarapan satu jam selepas bangun tidur, dan TIDAK mengambil sebarang bahan yang mengandungi gula (Cthnya: Madu, Stevia, gula etc).",
                    "Objektif utama sarapan PCOS adalah mengambil 30-40g clean protein pada waktu pagi untuk menstabilkan gula dalam darah"
                ],
                "options": [
                     {
                        "title": "Pilihan 1",
                        "items": ["4 biji telur putih + 1 kuning telur (Rebus/Separuh masak/scramble)"]
                    },
                    {
                        "title": "Pilihan 2",
                        "items": ["150g dada ayam (Bakar)- Boleh marinate dengan garam dan lada hitam atau sebarang rempah"]
                    },
                    {
                        "title": "Pilihan 3",
                        "items": ["150g Ayam Brand Tuna Chunk in water (Perah satu biji limau dan makan)"]
                    },
                    {
                        "title": "Pilihan 4",
                        "items": ["200g Tempe (Goreng dengan sedikit minyak/airfryer)"]
                    },
                    {
                        "title": "Pilihan 5",
                        "items": [
                            "Boleh buat waktu malam dan makan untuk sarapan. Jika tiada berries tiada masalah, tapi JANGAN gantikan dengan buahan lain.",
                            "150ml almond milk/coconut milk",
                            "¼ cup chia seeds",
                            "1 scoop protein (Tiada gula)",
                            "½ cup of fresh or frozen berries (e.g. raspberries, strawberries or blueberries)"
                        ]
                    },
                    {
                        "title": "Pilihan 6",
                        "items": [
                            "Berries smoothies",
                            "½ cup frozen raspberries",
                            "1 scoop protein- (Tiada gula)",
                            "150ml coconut/almond milk"
                        ]
                    }
                ]
            },
            "mediterranean_diet": {
                "title": "Diet Mediterranean (Makan tengah hari & Makan malam)",
                 "guidelines": [
                    "Protein - sasarkan jumlah sebesar tapak tangan (saiz dan ketebalan badan anda tapak tangan)",
                    "Karbohidrat- ambil lebih kurang ½ cawan selepas masak",
                    "Sayur-sayuran yang tidak berkanji ambil sebanyak yang boleh"
                ],
                "menu": {
                     "items": [
                        "120g ayam/ikan/daging",
                        "120g nasi/kentang (Jika boleh cuba tukar kepada brown rice/basmati (low GI)/wholemeal bread)",
                        "240g sayur-sayuran tidak berkanji",
                        "20g Kacang (Almond/Walnut)/10g Dark Chocolate 80% keatas/30g Chia Seed",
                        "1 biji buah epal/oren (Jika rasa nak ambil makanan manis)"
                    ],
                    "notes": [
                         "Cara masakkan kurang minyak dan tidak menggunakan gula. Elakkan mengambil makanan fast-food dan sebarang gula untuk tempoh empat minggu pertama.",
                        "p/s: Jika makan diluar, sila minta makanan kurang minyak dan tiada ajinamoto.",
                        "4 minggu pertama elakkan mengambil fast food."
                    ]
                },
                "anti_inflammatory_foods":{
                    "title": "Makanan yang baik diambil (Makanan Anti-radang)",
                    "items": [
                         "Ikan berlemak seperti salmon, sardin dan makarel ini mengandungi tahap asid lemak omega-3 yang tinggi, yang merendahkan keradangan serta menyokong keseimbangan hormon yang sihat dan mood yang stabil",
                        "Biji chia dan biji rami tanah ini kaya dengan ALA– pendahulu berasaskan tumbuhan kepada omega-3",
                        "Minyak zaitun ini mempunyai sebatian anti-radang seperti oleocanthal",
                        "Beri merah gelap seperti strawberry, raspberi, beri biru, dan beri hitam ini tinggi dengan antioksidan yang melawan radikal bebas",
                        "Sayur-sayuran berdaun gelap seperti kangkung, bayam, bok choy dan silverbeet ini penuh dengan banyak nutrien termasuk vitamin K, kalsium dan vitamin B",
                        "Kacang seperti badam, walnut, kacang Brazil, dan biji seperti biji labu dan biji bunga matahari untuk tahap vitamin E dan selenium yang bermanfaat.",
                        "Sayuran cruciferous seperti brokoli, kembang kol, kubis, dan pucuk Brussels yang mengandungi sebatian yang menyokong detoksifikasi hormon",
                        "Herba dan rempah ratus seperti halia, kunyit, kayu manis, bawang putih, lada hitam, dan rosemary ini membantu menjadikan rasa makanan lebih enak dan juga mengandungi sebatian anti-radang."
                    ]
                 },
                 "avoid_foods": {
                    "title": "Makanan yang perlu dielakkan untuk rawatan keradangan",
                    "items": [
                        "Makanan yang mengandung gluten seperti apa saja jenis roti, jadi gantikan dengan brown rice",
                        "Elakan Produk susu lembu (Boleh ambil susu kambing/susu almond/susu kelapa)",
                        "Makanan Diproses dan Bergoreng"
                    ]
                 }
            },
             "meal_timing": {
                "title": "Masa makan",
                "instructions": "Target untuk puasa selama 12 jam dimana jarak dari makan malam ke sarapan pagi",
                "schedule": [
                    "Sarapan pagi - 1 jam selepas bangun tidur",
                    "Makan tengah hari - 12 tgh hari atau 1 petang",
                    "Makan malam 7mlm-8mlm"
                ]
            },
            "waist_measurement": {
                "title": "Ukur Lilitan Pinggang",
                 "instructions": "Ukur Lilitan Pinggang sebelum mula diet plan: __________inchi",
                "weeks": [
                    "Minggu Pertama",
                    "Minggu Kedua",
                    "Minggu Ketiga",
                    "Minggu Keempat"
                ]
            },
            "exercise_plan": {
                "title": "Plan Senaman:",
                "options": [
                    [
                        "Lakukan brist walk 2-3 kali seminggu dengan melengkapkan 10,000 steps",
                        "Untuk aktiviti harian cuba banyakkan berjalan dengan parking lebih jauh dan capai lebih daripada 5000 step sehari"
                    ],
                    [
                         "Lakukan 20 minute senaman HIIT di rumah 2-3 Minggu sekali-Contoh senaman sila rujuk ebook Protokol Bebas PCOS."
                    ]
                ]
            }
        }
    }
    
    return meal_plans.get(pcos_type, {
        "error": "No specific meal plan available for this PCOS type"
    })

@app.get("/analyze/random-data")
async def get_random_data():
    try:
        # Find the Excel file in data directory
        excel_files = []
        for root, dirs, files in os.walk("data"):
            for file in files:
                if file.endswith('.xlsx') or file.endswith('.xls'):
                    excel_files.append(os.path.join(root, file))
        
        if not excel_files:
            return JSONResponse(content={
                "success": False,
                "error": "No Excel file found. Please upload data first."
            })
        
        # Use the first Excel file found
        file_path = excel_files[0]
        
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        # Clean column names
        df.columns = [str(col).replace('/', '_').strip() for col in df.columns]
        
        # Get a random row
        random_row = df.sample(n=1).iloc[0]
        
        # Convert to dictionary and handle all values as strings
        random_data = {}
        for col in df.columns:
            value = random_row[col]
            
            # Convert any value to string, handling special cases
            if pd.isna(value):
                random_data[col] = "0"  # Default to "0" for NaN values
            elif isinstance(value, (pd.Timestamp, datetime)):
                random_data[col] = value.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(value, time):
                random_data[col] = value.strftime('%H:%M:%S')
            elif isinstance(value, date):
                random_data[col] = value.strftime('%Y-%m-%d')
            elif isinstance(value, (int, float)):
                random_data[col] = str(value)
            else:
                # For any other type, convert to string and handle empty values
                str_value = str(value).strip()
                random_data[col] = str_value if str_value else "0"
        
        # Load field names to ensure we have all required fields
        field_names_path = os.path.join('data', 'field_names.json')
        if os.path.exists(field_names_path):
            with open(field_names_path, 'r', encoding='utf-8') as f:
                field_data = json.load(f)
                required_fields = field_data.get('fields', [])
                
                # Ensure all required fields have a value
                for field in required_fields:
                    if field not in random_data:
                        random_data[field] = "0"
        
        return JSONResponse(content={
            "success": True,
            "data": random_data
        })
    except Exception as e:
        logger.error(f"Error getting random data: {str(e)}")
        return JSONResponse(content={
            "success": False,
            "error": str(e)
        })

# Visualization routes
@app.get("/visualize", response_class=HTMLResponse)
async def get_visualize_page(request: Request):
    """Serve the visualization page"""
    return templates.TemplateResponse("visualize.html", {"request": request})

# Home and About routes
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    """Serve the home page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/test", response_class=HTMLResponse)
async def test_page(request: Request):
    """Test page"""
    return HTMLResponse(content="<h1>Test Page</h1>")

@app.get("/about", response_class=HTMLResponse)
async def get_about_page(request: Request):
    """Serve the about page"""
    return templates.TemplateResponse("about.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 