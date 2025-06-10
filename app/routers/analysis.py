from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
from app.services.model_service import ModelService
from app.utils.logger import logger

router = APIRouter(prefix="/analyze", tags=["analysis"])
templates = Jinja2Templates(directory="templates")
model_service = ModelService()

@router.get("/", response_class=HTMLResponse)
async def analyze_page(request: Request):
    """Serve the analysis page"""
    return templates.TemplateResponse("analyze.html", {"request": request})

@router.post("/")
async def analyze(request: Request):
    """Analyze patient data and make predictions"""
    try:
        data = await request.json()
        
        # Convert the data to a DataFrame
        df = pd.DataFrame([data])
        
        # Make predictions
        predictions = model_service.predict(df)
        
        # Generate meal plan based on PCOS type
        meal_plan = generate_meal_plan(predictions['predictions'][0])
        
        return {
            "success": True,
            "predictions": predictions,
            "meal_plan": meal_plan
        }
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/random-data")
async def get_random_data():
    """Generate random data for testing"""
    try:
        # Generate random data with appropriate ranges
        data = {
            "Age": np.random.randint(18, 45),
            "BMI": np.random.uniform(18.5, 35),
            "Waist_Hip_Ratio": np.random.uniform(0.7, 1.1),
            "Cycle_Regularity": np.random.choice(["Regular", "Irregular"]),
            "Cycle_Length": np.random.randint(21, 35),
            "Menstrual_Flow": np.random.choice(["Light", "Medium", "Heavy"]),
            "Acne": np.random.choice(["Yes", "No"]),
            "Hair_Growth": np.random.choice(["Yes", "No"]),
            "Hair_Loss": np.random.choice(["Yes", "No"]),
            "Weight_Gain": np.random.choice(["Yes", "No"]),
            "Insulin_Resistance": np.random.choice(["Yes", "No"]),
            "Diabetes": np.random.choice(["Yes", "No"]),
            "Hypertension": np.random.choice(["Yes", "No"]),
            "Family_History_PCOS": np.random.choice(["Yes", "No"]),
            "Family_History_Diabetes": np.random.choice(["Yes", "No"]),
            "Family_History_Hypertension": np.random.choice(["Yes", "No"]),
            "Exercise_Frequency": np.random.choice(["Never", "Rarely", "Sometimes", "Regularly"]),
            "Diet_Type": np.random.choice(["Regular", "Vegetarian", "Vegan", "Keto", "Other"]),
            "Stress_Level": np.random.choice(["Low", "Medium", "High"]),
            "Sleep_Hours": np.random.randint(4, 10)
        }
        
        return {
            "success": True,
            "data": data
        }
        
    except Exception as e:
        logger.error(f"Error generating random data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_meal_plan(pcos_type: str) -> dict:
    """Generate a meal plan based on PCOS type"""
    meal_plans = {
        "Type A": {
            "breakfast": "Oatmeal with berries and nuts",
            "lunch": "Grilled chicken salad with olive oil dressing",
            "dinner": "Baked salmon with steamed vegetables",
            "snacks": ["Greek yogurt", "Apple with almond butter"],
            "recommendations": [
                "Focus on high-protein, low-carb meals",
                "Include omega-3 rich foods",
                "Limit processed foods and sugars"
            ]
        },
        "Type B": {
            "breakfast": "Greek yogurt with honey and walnuts",
            "lunch": "Quinoa bowl with vegetables and tofu",
            "dinner": "Lentil soup with whole grain bread",
            "snacks": ["Mixed nuts", "Carrot sticks with hummus"],
            "recommendations": [
                "Emphasize plant-based proteins",
                "Include complex carbohydrates",
                "Stay hydrated with water and herbal teas"
            ]
        },
        "Type C": {
            "breakfast": "Smoothie bowl with protein powder",
            "lunch": "Turkey wrap with whole grain tortilla",
            "dinner": "Stir-fried vegetables with lean beef",
            "snacks": ["Hard-boiled eggs", "Cucumber slices"],
            "recommendations": [
                "Balance protein and complex carbs",
                "Include anti-inflammatory foods",
                "Regular meal timing"
            ]
        }
    }
    
    return meal_plans.get(pcos_type, {
        "breakfast": "Consult with a nutritionist",
        "lunch": "Consult with a nutritionist",
        "dinner": "Consult with a nutritionist",
        "snacks": ["Consult with a nutritionist"],
        "recommendations": ["Please consult with a healthcare provider for personalized meal planning"]
    }) 