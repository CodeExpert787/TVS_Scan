from typing import List, Dict
from models.pcos_patient import PatientData, PCOSType

class MealPlanGenerator:
    def __init__(self):
        # Breakfast options categorized by type
        self.breakfast_options = {
            "protein_rich": [
                {
                    "name": "Protein-Packed Breakfast Bowl",
                    "description": "4 egg whites + 1 yolk (boiled/half-boiled/scrambled) with 1/2 avocado and 1/4 cup cherry tomatoes",
                    "calories": 350,
                    "protein": 25,
                    "carbs": 12,
                    "fat": 22
                },
                {
                    "name": "Grilled Chicken Breakfast",
                    "description": "150g chicken breast (grilled) with 1/2 cup sautéed spinach and 1/4 cup mushrooms",
                    "calories": 280,
                    "protein": 35,
                    "carbs": 8,
                    "fat": 12
                },
                {
                    "name": "Tuna Protein Bowl",
                    "description": "150g tuna in water with lemon, 1/4 cup cucumber, and 1/4 cup bell peppers",
                    "calories": 220,
                    "protein": 30,
                    "carbs": 5,
                    "fat": 8
                },
                {
                    "name": "Tempeh Power Bowl",
                    "description": "200g tempeh (fried/air-fried) with 1/2 cup steamed broccoli and 1/4 cup carrots",
                    "calories": 320,
                    "protein": 28,
                    "carbs": 15,
                    "fat": 18
                }
            ],
            "smoothie": [
                {
                    "name": "Berry Protein Smoothie",
                    "description": "150ml almond/coconut milk + ¼ cup chia seeds + 1 scoop protein + ½ cup berries",
                    "calories": 280,
                    "protein": 22,
                    "carbs": 25,
                    "fat": 12
                },
                {
                    "name": "Green Protein Smoothie",
                    "description": "½ cup frozen berries + 1 scoop protein + 150ml coconut/almond milk + 1 cup spinach",
                    "calories": 250,
                    "protein": 20,
                    "carbs": 22,
                    "fat": 10
                }
            ]
        }
        
        # Mediterranean diet guidelines with detailed nutritional info
        self.mediterranean_diet = {
            "protein": {
                "options": [
                    {
                        "name": "Grilled Chicken Breast",
                        "portion": "120g",
                        "calories": 180,
                        "protein": 35,
                        "carbs": 0,
                        "fat": 4
                    },
                    {
                        "name": "Baked Salmon",
                        "portion": "120g",
                        "calories": 220,
                        "protein": 25,
                        "carbs": 0,
                        "fat": 14
                    },
                    {
                        "name": "Lean Beef",
                        "portion": "120g",
                        "calories": 200,
                        "protein": 30,
                        "carbs": 0,
                        "fat": 10
                    }
                ]
            },
            "carbs": {
                "options": [
                    {
                        "name": "Brown Rice",
                        "portion": "120g",
                        "calories": 220,
                        "protein": 5,
                        "carbs": 45,
                        "fat": 2
                    },
                    {
                        "name": "Sweet Potato",
                        "portion": "120g",
                        "calories": 180,
                        "protein": 3,
                        "carbs": 40,
                        "fat": 0
                    },
                    {
                        "name": "Quinoa",
                        "portion": "120g",
                        "calories": 200,
                        "protein": 8,
                        "carbs": 38,
                        "fat": 3
                    }
                ]
            },
            "vegetables": {
                "options": [
                    {
                        "name": "Mixed Vegetables",
                        "portion": "240g",
                        "description": "Broccoli, carrots, bell peppers, and zucchini",
                        "calories": 120,
                        "protein": 6,
                        "carbs": 20,
                        "fat": 2
                    }
                ]
            },
            "snacks": [
                {
                    "name": "Mixed Nuts",
                    "portion": "20g",
                    "calories": 120,
                    "protein": 4,
                    "carbs": 4,
                    "fat": 10
                },
                {
                    "name": "Dark Chocolate",
                    "portion": "10g",
                    "calories": 50,
                    "protein": 1,
                    "carbs": 4,
                    "fat": 3
                },
                {
                    "name": "Chia Seeds",
                    "portion": "30g",
                    "calories": 140,
                    "protein": 5,
                    "carbs": 12,
                    "fat": 9
                }
            ],
            "fruits": [
                {
                    "name": "Apple",
                    "portion": "1 medium",
                    "calories": 95,
                    "protein": 0.5,
                    "carbs": 25,
                    "fat": 0.3
                },
                {
                    "name": "Orange",
                    "portion": "1 medium",
                    "calories": 80,
                    "protein": 1.5,
                    "carbs": 19,
                    "fat": 0.2
                }
            ]
        }

    def generate_meal_plan(self, patient_data: PatientData) -> Dict:
        """
        Generate a complete meal plan based on patient data
        """
        # Convert patient_data to dict if it's not already
        if not isinstance(patient_data, dict):
            patient_data = patient_data.dict()

        # Generate meal components
        breakfast = self._generate_breakfast(patient_data)
        lunch = self._generate_mediterranean_meal()
        dinner = self._generate_mediterranean_meal()

        # Calculate total nutrition values
        total_calories = 0
        total_protein = 0
        total_carbs = 0
        total_fat = 0

        # Add breakfast nutrition (using first option)
        if breakfast["options"]:
            option = breakfast["options"][0]
            total_calories += option["calories"]
            total_protein += option["protein"]
            total_carbs += option["carbs"]
            total_fat += option["fat"]

        # Add lunch nutrition (using first option of each type)
        if lunch["protein"]["options"]:
            option = lunch["protein"]["options"][0]
            total_calories += option["calories"]
            total_protein += option["protein"]
            total_carbs += option["carbs"]
            total_fat += option["fat"]

        if lunch["carbs"]["options"]:
            option = lunch["carbs"]["options"][0]
            total_calories += option["calories"]
            total_protein += option["protein"]
            total_carbs += option["carbs"]
            total_fat += option["fat"]

        if lunch["vegetables"]["options"]:
            option = lunch["vegetables"]["options"][0]
            total_calories += option["calories"]
            total_protein += option["protein"]
            total_carbs += option["carbs"]
            total_fat += option["fat"]

        # Add dinner nutrition (same as lunch)
        if dinner["protein"]["options"]:
            option = dinner["protein"]["options"][0]
            total_calories += option["calories"]
            total_protein += option["protein"]
            total_carbs += option["carbs"]
            total_fat += option["fat"]

        if dinner["carbs"]["options"]:
            option = dinner["carbs"]["options"][0]
            total_calories += option["calories"]
            total_protein += option["protein"]
            total_carbs += option["carbs"]
            total_fat += option["fat"]

        if dinner["vegetables"]["options"]:
            option = dinner["vegetables"]["options"][0]
            total_calories += option["calories"]
            total_protein += option["protein"]
            total_carbs += option["carbs"]
            total_fat += option["fat"]

        # Add snacks nutrition (using first snack)
        if lunch["snacks"]:
            snack = lunch["snacks"][0]
            total_calories += snack["calories"]
            total_protein += snack["protein"]
            total_carbs += snack["carbs"]
            total_fat += snack["fat"]

        # Add fruits nutrition (using first fruit)
        if lunch["fruits"]:
            fruit = lunch["fruits"][0]
            total_calories += fruit["calories"]
            total_protein += fruit["protein"]
            total_carbs += fruit["carbs"]
            total_fat += fruit["fat"]

        meal_plan = {
            "patient_info": {
                "name": patient_data.get("name", ""),
                "bmi": patient_data.get("bmi", 0),
                "healthy_weight_range": patient_data.get("healthy_weight_range", (0, 0)),
                "water_intake": f"{patient_data.get('water_intake', 0)} liters per day"
            },
            "breakfast": breakfast,
            "lunch": lunch,
            "dinner": dinner,
            "meal_timing": {
                "breakfast": "1 hour after waking up",
                "lunch": "12:00 PM - 1:00 PM",
                "dinner": "7:00 PM - 8:00 PM"
            },
            "exercise_plan": self._generate_exercise_plan(patient_data),
            "supplements": self._generate_supplements(patient_data),
            "total_nutrition": {
                "calories": total_calories,
                "protein": total_protein,
                "carbs": total_carbs,
                "fat": total_fat
            }
        }
        
        return meal_plan

    def _generate_breakfast(self, patient_data: Dict) -> Dict:
        """
        Generate breakfast options based on PCOS type
        For insulin-resistant PCOS, prioritize protein-rich breakfast
        """
        pcos_type = patient_data.get("pcos_type", "")
        if "Rintangan Insulin" in pcos_type:
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
        Generate Mediterranean diet meal guidelines with detailed nutritional information
        """
        return {
            "protein": self.mediterranean_diet["protein"],
            "carbs": self.mediterranean_diet["carbs"],
            "vegetables": self.mediterranean_diet["vegetables"],
            "snacks": self.mediterranean_diet["snacks"],
            "fruits": self.mediterranean_diet["fruits"],
            "notes": "Cook with minimal oil, no sugar. Avoid fast food and MSG."
        }

    def _generate_exercise_plan(self, patient_data: Dict) -> Dict:
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

    def _generate_supplements(self, patient_data: Dict) -> Dict:
        """
        Generate supplement recommendations based on PCOS type
        """
        supplements = []
        pcos_type = patient_data.get("pcos_type", "")
        if "Rintangan Insulin" in pcos_type:
            supplements.append("Feminira (for 3 months) - Balance hormones and improve egg quality")
        if "Adrenal" in pcos_type:
            supplements.append("Consider stress-reducing herbal supplements")
        return {"recommendations": supplements} 