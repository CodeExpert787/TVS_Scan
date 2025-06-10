from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import json
from datetime import datetime
import os
from app.config.settings import DATA_DIR, FIELD_NAMES_PATH
from app.services.model_service import ModelService
from app.utils.logger import logger

router = APIRouter(prefix="/training", tags=["training"])
templates = Jinja2Templates(directory="templates")
model_service = ModelService()

@router.get("/", response_class=HTMLResponse)
async def get_training_page(request: Request):
    """Serve the training page"""
    return templates.TemplateResponse("training.html", {"request": request})

@router.post("/upload")
async def upload_training_data(file: UploadFile = File(...)):
    """Handle training data upload"""
    try:
        # Create data directory if it doesn't exist
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Save the uploaded file
        file_path = os.path.join(DATA_DIR, file.filename)
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
        with open(FIELD_NAMES_PATH, "w", encoding="utf-8") as f:
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
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start")
async def start_training(request: Request):
    """Start model training process"""
    try:
        data = await request.json()
        filename = data.get('filename')
        
        if not filename:
            raise HTTPException(status_code=400, detail="No filename provided")
            
        # Load the data from the uploaded file
        file_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")
            
        logger.info(f"Loading file from: {file_path}")
        
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        # Clean column names
        df.columns = [str(col).replace('/', '_').strip() for col in df.columns]
        
        # Handle datetime columns
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            elif pd.api.types.is_object_dtype(df[col]):
                df[col] = df[col].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if isinstance(x, (pd.Timestamp, datetime)) else x)
        
        # Train the model
        training_results = model_service.train_model(df)
        
        return {
            "success": True,
            "message": "Model training completed successfully",
            "results": training_results
        }
        
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 