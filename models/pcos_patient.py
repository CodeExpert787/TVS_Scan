from enum import Enum
from typing import Optional, List
from pydantic import BaseModel

class PCOSType(str, Enum):
    """Enum for different types of PCOS."""
    INSULIN_RESISTANCE = "Rintangan Insulin"
    ADRENAL = "Adrenal"
    UNKNOWN = "Unknown"

class PatientData(BaseModel):
    """Model for PCOS patient data."""
    name: str
    age: int
    weight: float
    height: float
    bmi: float
    healthy_weight_range: tuple[float, float]
    water_intake: float
    pcos_type: str  # Changed from PCOSType to str
    waist_measurement: Optional[float]
    menstrual_cycle_length: Optional[int] = None
    symptoms: List[str] = []
    medical_history: Optional[str] = None
    medications: List[str] = []
    allergies: List[str] = []
    dietary_restrictions: List[str] = []
    activity_level: Optional[str] = None
    sleep_hours: Optional[float] = None
    stress_level: Optional[int] = None 