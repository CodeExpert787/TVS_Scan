from typing import List, Dict
from models.pcos_patient import PatientData, PCOSType

class MealPlanGenerator:
    def __init__(self):
        # Breakfast options categorized by type
        self.breakfast_options = {
            "protein_rich": [
                "4 egg whites + 1 yolk (boiled/half-boiled/scrambled)",
                "150g chicken breast (grilled)",
                "150g tuna in water with lemon",
                "200g tempeh (fried/air-fried)"
            ],
            "smoothie": [
                "150ml almond/coconut milk + ¼ cup chia seeds + 1 scoop protein + ½ cup berries",
                "½ cup frozen berries + 1 scoop protein + 150ml coconut/almond milk"
            ]
        }
        
        # Mediterranean diet guidelines
        self.mediterranean_diet = {
            "protein": "120g chicken/fish/meat",
            "carbs": "120g rice/potato (preferably brown rice/basmati/wholemeal bread)",
            "vegetables": "240g non-starchy vegetables",
            "snacks": ["20g nuts (Almond/Walnut)", "10g Dark Chocolate 80%+", "30g Chia Seeds"],
            "fruits": ["1 apple", "1 orange"]
        }

    def generate_meal_plan(self, patient_data: PatientData) -> Dict:
        """
        Generate a complete meal plan based on patient data
        """
        meal_plan = {
            "patient_info": {
                "name": patient_data.name,
                "bmi": patient_data.bmi,
                "healthy_weight_range": patient_data.healthy_weight_range,
                "water_intake": f"{patient_data.water_intake} liters per day"
            },
            "breakfast": self._generate_breakfast(patient_data),
            "lunch": self._generate_mediterranean_meal(),
            "dinner": self._generate_mediterranean_meal(),
            "meal_timing": {
                "breakfast": "1 hour after waking up",
                "lunch": "12:00 PM - 1:00 PM",
                "dinner": "7:00 PM - 8:00 PM"
            },
            "exercise_plan": self._generate_exercise_plan(patient_data),
            "supplements": self._generate_supplements(patient_data)
        }
        
        return meal_plan

    def _generate_breakfast(self, patient_data: PatientData) -> Dict:
        """
        Generate breakfast options based on PCOS type
        For insulin-resistant PCOS, prioritize protein-rich breakfast
        """
        if PCOSType.INSULIN_RESISTANT in patient_data.pcos_types:
            return {
                "type": "protein_rich",
                "options": self.breakfast_options["protein_rich"],
                "notes": "Must be consumed within 1 hour of waking up. No sugar allowed."
            }
        return {
            "type": "balanced",
            "options": self.breakfast_options["protein_rich"] + self.breakfast_options["smoothie"],
            "notes": "Must be consumed within 1 hour of waking up. No sugar allowed."
        }

    def _generate_mediterranean_meal(self) -> Dict:
        """
        Generate Mediterranean diet meal guidelines
        """
        return {
            "protein": self.mediterranean_diet["protein"],
            "carbs": self.mediterranean_diet["carbs"],
            "vegetables": self.mediterranean_diet["vegetables"],
            "snacks": self.mediterranean_diet["snacks"],
            "fruits": self.mediterranean_diet["fruits"],
            "notes": "Cook with minimal oil, no sugar. Avoid fast food and MSG."
        }

    def _generate_exercise_plan(self, patient_data: PatientData) -> Dict:
        """
        Generate exercise recommendations based on patient's condition
        """
        return {
            "weekly_plan": [
                "Brisk walk 2-3 times per week (10,000 steps)",
                "Daily activity: Park further and aim for 5,000+ steps",
                "OR 20-minute HIIT workout 2-3 times per week"
            ],
            "notes": "Choose activities based on your comfort level and energy"
        }

    def _generate_supplements(self, patient_data: PatientData) -> Dict:
        """
        Generate supplement recommendations based on PCOS type
        """
        supplements = []
        if PCOSType.INSULIN_RESISTANT in patient_data.pcos_types:
            supplements.append("Feminira (for 3 months) - Balance hormones and improve egg quality")
        if PCOSType.ADRENAL in patient_data.pcos_types:
            supplements.append("Consider stress-reducing herbal supplements")
        return {"recommendations": supplements} 