from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
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
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        training_status["status"] = "failed"
        training_status["error"] = str(e)
    finally:
        training_status["is_training"] = False

@app.get("/", response_class=HTMLResponse)
async def get_home():
    """Serve the main page"""
    with open("templates/index.html") as f:
        return f.read()

@app.post("/upload-training-data")
async def upload_training_data(file: UploadFile = File(...)):
    """
    Upload a training data file (docx)
    """
    try:
        # Create data/raw directory if it doesn't exist
        os.makedirs('data/raw', exist_ok=True)
        
        # Save the uploaded file
        file_path = os.path.join('data/raw', file.filename)
        save_upload_file(file, file_path)
        
        return JSONResponse(
            status_code=200,
            content={"message": f"File {file.filename} uploaded successfully"}
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

if __name__ == "__main__":
    # Create necessary folders
    from setup_folders import create_folder_structure
    create_folder_structure()
    
    # Run the application
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 