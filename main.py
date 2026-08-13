from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allows cross-origin requests from any HTML file
    allow_credentials=True,
    allow_methods=["*"],       # Allows POST, OPTIONS, GET, etc.
    allow_headers=["*"],
)

model = joblib.load('decision_tree_model.pkl')
    
    
#input schema
from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    Gender: str
    Age: int = Field(ge=18, le=100)
    Senior_Citizen: int = Field(ge=0, le=1)
    Marital_Status: str
    Region: str
    Tenure: int = Field(ge=0)
    Contract_Type: str
    Internet_Service: str
    Phone_Service: str
    Multiple_Lines: str
    Tech_Support: str
    Streaming_Service: str
    Payment_Method: str
    Monthly_Charges: float = Field(gt=0)
    Total_Charges: float = Field(ge=0)
    Data_Usage_GB: float = Field(ge=0)
    Call_Minutes: float = Field(ge=0)
    Support_Tickets: int = Field(ge=0)
    Satisfaction_Score: int = Field(ge=1, le=10)
    Last_Login_Days: int = Field(ge=0)
    Late_Payments: int = Field(ge=0)
    Auto_Pay: int = Field(ge=0, le=1)
    
@app.get('/')
def home():
    return {"message":"Loan Prediction API is running"}

@app.post("/predict")
def predict(data:PredictionInput):
    input_data = pd.DataFrame([{
        "Gender": data.Gender,
        "Age": data.Age,
        "Senior_Citizen": data.Senior_Citizen,
        "Marital_Status": data.Marital_Status,
        "Region": data.Region,
        "Tenure": data.Tenure,  
        "Contract_Type": data.Contract_Type,
        "Internet_Service": data.Internet_Service,
        "Phone_Service": data.Phone_Service,
        "Multiple_Lines": data.Multiple_Lines,
        "Tech_Support": data.Tech_Support,
        "Streaming_Service": data.Streaming_Service,
        "Payment_Method": data.Payment_Method,
        "Monthly_Charges": data.Monthly_Charges,
        "Total_Charges": data.Total_Charges,
        "Data_Usage_GB": data.Data_Usage_GB,
        "Call_Minutes": data.Call_Minutes,
        "Support_Tickets": data.Support_Tickets,
        "Satisfaction_Score": data.Satisfaction_Score,
        "Last_Login_Days": data.Last_Login_Days,
        "Late_Payments": data.Late_Payments,
        "Auto_Pay": data.Auto_Pay
    }])
    prediction = model.predict(input_data)
    return {
        "prediction": int(prediction[0])
    }
