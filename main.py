from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from models.pcos_patient import PatientData, PCOSType
from meal_plan_generator import MealPlanGenerator
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from typing import List
import uvicorn
import os
import shutil
import logging
from datetime import datetime
from starlette.middleware.sessions import SessionMiddleware

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
    title="PCOS Meal Plan Generator",
    description="A personalized meal plan generator for PCOS patients based on their test results and physical data",
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
model_trainer = ModelTrainer('models')

# Global variable to track training status
training_status = {
    "is_training": False,
    "progress": 0,
    "status": "idle",
    "last_trained": None
}

def save_upload_file(upload_file: UploadFile, destination: str):
    """Save uploaded file to destination"""
    try:
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()

async def train_model_background():
    """Background task for model training"""
    global training_status
    try:
        training_status["is_training"] = True
        training_status["status"] = "processing"
        training_status["progress"] = 0

        # Load and process data
        logger.info("Loading and processing data...")
        train_data, test_data = data_processor.load_data()
        training_status["progress"] = 30

        # Save processed data
        logger.info("Saving processed data...")
        data_processor.save_processed_data(train_data, test_data, 'data/processed')
        training_status["progress"] = 50

        # Preprocess data
        logger.info("Preprocessing data...")
        X_train, y_train = data_processor.preprocess_data(train_data)
        X_test, y_test = data_processor.preprocess_data(test_data)
        training_status["progress"] = 70

        # Train model
        logger.info("Training model...")
        train_metrics = model_trainer.train(X_train, y_train)
        training_status["progress"] = 90

        # Evaluate and save model
        logger.info("Evaluating model...")
        test_metrics = model_trainer.evaluate(X_test, y_test)
        model_trainer.save_model()
        
        training_status["progress"] = 100
        training_status["status"] = "completed"
        training_status["last_trained"] = datetime.now().isoformat()
        training_status["metrics"] = {
            "train_accuracy": train_metrics["train_accuracy"],
            "accuracy": test_metrics["accuracy"],
            "classification_report": test_metrics["classification_report"]
        }
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        training_status["status"] = "failed"
        training_status["error"] = str(e)
    finally:
        training_status["is_training"] = False

@app.get("/", response_class=HTMLResponse)
async def get_home(request:Request):
    """Serve the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
async def get_about_page(request: Request):
    """Serve the about page"""
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/training", response_class=HTMLResponse)
async def get_training_page(request:Request):
    """Serve the training page"""
    return templates.TemplateResponse("training.html", {"request": request})

@app.get("/analyze", response_class=HTMLResponse)
async def get_analyze_page(request: Request):
    """Serve the analyze page"""
    return templates.TemplateResponse(
        "analyze.html",
        {
            "request": request,
            "form_values": {},
            "show_results": False,
            "recommendations": None,
            "error": None
        }
    )

@app.get("/meal-plan", response_class=HTMLResponse)
async def get_meal_plan_page(request: Request):
    """Serve the meal plan page with generated meal plan"""
    try:
        # Get the patient data from the session or request
        patient_data = request.session.get('patient_data')
        if not patient_data:
            return templates.TemplateResponse(
                "meal_plan.html",
                {
                    "request": request,
                    "error": "Please complete the analysis first to generate a meal plan."
                }
            )

        # Generate meal plan
        meal_plan = meal_plan_generator.generate_meal_plan(patient_data)
        
        # Generate dietary guidelines based on PCOS type
        dietary_guidelines = []
        pcos_type = patient_data.get("pcos_type", "")
        
        if "Rintangan Insulin" in pcos_type:
            dietary_guidelines.extend([
                "Focus on low glycemic index foods",
                "Include protein with every meal",
                "Limit refined carbohydrates",
                "Choose complex carbs over simple sugars"
            ])
        elif "Adrenal" in pcos_type:
            dietary_guidelines.extend([
                "Anti-inflammatory diet",
                "Omega-3 rich foods",
                "Limit caffeine and alcohol",
                "Include magnesium-rich foods"
            ])
        else:
            dietary_guidelines.extend([
                "Balanced low-glycemic diet",
                "Regular meal timing",
                "Anti-inflammatory foods",
                "Adequate protein intake"
            ])

        return templates.TemplateResponse(
            "meal_plan.html",
            {
                "request": request,
                "meal_plan": meal_plan,
                "dietary_guidelines": dietary_guidelines
            }
        )
    except Exception as e:
        logger.error(f"Error generating meal plan: {str(e)}")
        return templates.TemplateResponse(
            "meal_plan.html",
            {
                "request": request,
                "error": "Error generating meal plan. Please try again."
            }
        )

@app.post("/upload-training-data")
async def upload_training_data(files: List[UploadFile] = File(...)):
    """
    Upload multiple training data files (docx, csv, xlsx)
    """
    try:
        # Create data/raw directory if it doesn't exist
        os.makedirs('data/raw', exist_ok=True)
        
        uploaded_files = []
        # Save each uploaded file
        for file in files:
            file_path = os.path.join('data/raw', file.filename)
            save_upload_file(file, file_path)
            uploaded_files.append(file.filename)
        
        return JSONResponse(
            status_code=200,
            content={"message": f"{len(uploaded_files)} files uploaded successfully", "files": uploaded_files}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start-training")
async def start_training(background_tasks: BackgroundTasks):
    """
    Start the model training process
    """
    if training_status["is_training"]:
        raise HTTPException(
            status_code=400,
            detail="Training is already in progress"
        )
    
    background_tasks.add_task(train_model_background)
    return {"message": "Training started", "status": "processing"}

@app.get("/training-status")
async def get_training_status():
    """
    Get the current training status
    """
    return training_status

@app.post("/generate-meal-plan")
async def generate_meal_plan(patient_data: PatientData):
    """
    Generate a personalized meal plan based on patient data
    """
    try:
        if not os.path.exists(os.path.join('models', 'pcos_model.joblib')):
            raise HTTPException(
                status_code=400,
                detail="Model not trained yet. Please train the model first."
            )
            
        meal_plan = meal_plan_generator.generate_meal_plan(patient_data)
        return meal_plan
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pcos-types")
async def get_pcos_types():
    """
    Get available PCOS types
    """
    return {"pcos_types": [type.value for type in PCOSType]}

def generate_recommendations(patient_data: PatientData) -> dict:
    """
    Generate personalized recommendations based on patient data
    """
    recommendations = {
        "dietary_recommendations": [],
        "lifestyle_recommendations": [],
        "exercise_recommendations": [],
        "supplement_recommendations": []
    }
    
    # PCOS Type specific recommendations
    if patient_data.pcos_type == "Rintangan Insulin":
        recommendations["dietary_recommendations"].extend([
            "Focus on low glycemic index foods",
            "Include protein with every meal",
            "Limit refined carbohydrates",
            "Choose complex carbs over simple sugars"
        ])
        recommendations["lifestyle_recommendations"].extend([
            "Regular meal timing",
            "Adequate sleep (7-8 hours)",
            "Stress management techniques"
        ])
        recommendations["exercise_recommendations"].extend([
            "30 minutes of moderate exercise daily",
            "Include both cardio and strength training",
            "Avoid high-intensity workouts if stressed"
        ])
        recommendations["supplement_recommendations"].extend([
            "Inositol (if recommended by doctor)",
            "Vitamin D",
            "Magnesium"
        ])
    
    elif patient_data.pcos_type == "Adrenal":
        recommendations["dietary_recommendations"].extend([
            "Anti-inflammatory diet",
            "Omega-3 rich foods",
            "Limit caffeine and alcohol",
            "Include magnesium-rich foods"
        ])
        recommendations["lifestyle_recommendations"].extend([
            "Stress reduction techniques",
            "Regular sleep schedule",
            "Mindfulness practices"
        ])
        recommendations["exercise_recommendations"].extend([
            "Gentle yoga",
            "Walking",
            "Swimming"
        ])
        recommendations["supplement_recommendations"].extend([
            "Vitamin B complex",
            "Magnesium",
            "Adaptogenic herbs (consult doctor)"
        ])
    
    else:  # Unknown or combined
        recommendations["dietary_recommendations"].extend([
            "Balanced low-glycemic diet",
            "Regular meal timing",
            "Anti-inflammatory foods",
            "Adequate protein intake"
        ])
        recommendations["lifestyle_recommendations"].extend([
            "Stress management",
            "Regular sleep schedule",
            "Mind-body practices"
        ])
        recommendations["exercise_recommendations"].extend([
            "Moderate exercise 4-5 times per week",
            "Mix of cardio and strength training",
            "Stress-reducing activities"
        ])
        recommendations["supplement_recommendations"].extend([
            "Inositol (if recommended)",
            "Vitamin D",
            "Magnesium",
            "Omega-3"
        ])
    
    # BMI-based recommendations
    if patient_data.bmi > 25:
        recommendations["dietary_recommendations"].append(
            "Focus on portion control and balanced meals"
        )
        recommendations["exercise_recommendations"].append(
            "Include regular physical activity to support weight management"
        )
    
    # Water intake recommendations
    if patient_data.water_intake < 2.5:
        recommendations["lifestyle_recommendations"].append(
            "Increase water intake to at least 2.5-3 liters per day"
        )
    
    # Stress level recommendations
    if patient_data.stress_level and patient_data.stress_level > 7:
        recommendations["lifestyle_recommendations"].extend([
            "Practice daily stress reduction techniques",
            "Consider meditation or deep breathing exercises",
            "Ensure adequate sleep and rest"
        ])
    
    return recommendations

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request):
    """Analyze patient data and generate recommendations."""
    try:
        form_data = await request.form()
        
        # Get form data
        name = form_data.get('name', '')
        age = int(form_data.get('age', 0))
        weight = float(form_data.get('weight', 0))
        height = float(form_data.get('height', 0))
        bmi = float(form_data.get('bmi', 0))
        water_intake = float(form_data.get('water_intake', 0))
        waist_measurement = float(form_data.get('waist_measurement', 0))
        pcos_type = form_data.get('pcos_type', 'Unknown')
        
        # Get optional form data
        menstrual_cycle_length = form_data.get('menstrual_cycle_length')
        symptoms = form_data.getlist('symptoms')
        medical_history = form_data.get('medical_history', '')
        medications = form_data.get('medications', '').split('\n')
        allergies = form_data.get('allergies', '').split('\n')
        dietary_restrictions = form_data.get('dietary_restrictions', '').split('\n')
        activity_level = form_data.get('activity_level', '')
        sleep_hours = form_data.get('sleep_hours')
        stress_level = form_data.get('stress_level')
        
        # Create patient data dictionary
        patient_data = {
            "name": name,
            "age": age,
            "weight": weight,
            "height": height,
            "bmi": bmi,
            "healthy_weight_range": (0.0, 0.0),  # Will be calculated
            "water_intake": water_intake,
            "pcos_type": pcos_type,
            "waist_measurement": waist_measurement,
            "menstrual_cycle_length": int(menstrual_cycle_length) if menstrual_cycle_length else None,
            "symptoms": symptoms,
            "medical_history": medical_history,
            "medications": [m.strip() for m in medications if m.strip()],
            "allergies": [a.strip() for a in allergies if a.strip()],
            "dietary_restrictions": [d.strip() for d in dietary_restrictions if d.strip()],
            "activity_level": activity_level,
            "sleep_hours": float(sleep_hours) if sleep_hours else None,
            "stress_level": int(stress_level) if stress_level else None
        }
        
        # Store patient data in session
        request.session['patient_data'] = patient_data
        
        # Generate recommendations
        recommendations = generate_recommendations(PatientData(**patient_data))
        
        # Create form data dictionary to preserve input values
        form_values = {
            "name": name,
            "age": age,
            "weight": weight,
            "height": height,
            "bmi": bmi,
            "water_intake": water_intake,
            "waist_measurement": waist_measurement,
            "pcos_type": pcos_type,
            "menstrual_cycle_length": menstrual_cycle_length,
            "symptoms": symptoms,
            "medical_history": medical_history,
            "medications": "\n".join(medications),
            "allergies": "\n".join(allergies),
            "dietary_restrictions": "\n".join(dietary_restrictions),
            "activity_level": activity_level,
            "sleep_hours": sleep_hours,
            "stress_level": stress_level
        }
        
        return templates.TemplateResponse(
            'analyze.html',
            {
                "request": request,
                "patient_data": patient_data,
                "recommendations": recommendations,
                "form_values": form_values,
                "show_results": True,
                "error": None
            }
        )
        
    except Exception as e:
        logger.error(f"Error analyzing data: {str(e)}")
        return templates.TemplateResponse(
            'analyze.html',
            {
                "request": request,
                "error": str(e),
                "form_values": form_values if 'form_values' in locals() else {},
                "show_results": False,
                "recommendations": None
            }
        )

if __name__ == "__main__":
    # Create necessary folders
    from setup_folders import create_folder_structure
    create_folder_structure()
    
    # Run the application
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 