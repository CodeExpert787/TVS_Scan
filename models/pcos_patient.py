from enum import Enum
from typing import Optional, List
from pydantic import BaseModel

class PCOSType(str, Enum):
    """Enum for different types of PCOS."""
    TYPE_A = "Type A"
    TYPE_B = "Type B"
    TYPE_C = "Type C"
    TYPE_D = "Type D"

class PatientData(BaseModel):
    """Model for PCOS patient data."""
    age: int
    weight: float
    height: float
    bmi: float
    menstrual_cycle_length: Optional[int] = None
    symptoms: List[str] = []
    pcos_type: Optional[PCOSType] = None
    medical_history: Optional[str] = None
    medications: List[str] = []
    allergies: List[str] = []
    dietary_restrictions: List[str] = []
    activity_level: Optional[str] = None
    sleep_hours: Optional[float] = None
    stress_level: Optional[int] = None  # 1-10 scale 