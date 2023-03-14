from pydantic import BaseModel,Field

class CKD_class_values(BaseModel):
    Age:float
    Blood_Pressure:float
    Specific_Gravity:float
    Albumin:float
    Sugar:float
    Red_Blood_Cells:float
    Pus_Cell: object
    Pus_Cell_clumps: object
    Bacteria: object
    Blood_Glucose_Random: float
    Blood_Urea:float
    Serum_Creatinine:float
    Sodium: float
    Potassium:float
    Hemoglobin:float
    Packed_Cell_Volume :float = Field(..., alias="Packed _Cell_Volume")
    White_Blood_Cell_Count:float
    Red_Blood_Cell_Count:float
    Hypertension:object
    Diabetes_Mellitus:object
    Coronary_Artery_Disease : object
    Appetite : object
    Pedal_Edema : object
    Anemia : object
    Class : object