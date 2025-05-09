# 1. Library imports
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from Customer import Customer
import numpy as np
import pickle
import pandas as pd

# 2. Create the app object
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the templates directory
app.mount("/static", StaticFiles(directory="templates"), name="static")

# Load the trained model
pickle_in = open("classifier.pkl","rb")
classifier = pickle.load(pickle_in)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return FileResponse('templates/index.html')

# 4. Expose the prediction functionality
@app.post('/predict')
def predict_customer(data: Customer):
    data = data.dict()
    CreditScore = data['CreditScore']
    Gender = data['Gender']
    Age = data['Age']
    Tenure = data['Tenure']
    Balance = data['Balance']
    NumOfProducts = data['NumOfProducts']
    HasCrCard = data['HasCrCard']
    IsActiveMember = data['IsActiveMember']
    EstimatedSalary = data['EstimatedSalary']

    prediction = classifier.predict([[CreditScore, Gender, Age, Tenure, Balance, 
                                   NumOfProducts, HasCrCard, IsActiveMember, 
                                   EstimatedSalary]])
    
    if prediction[0] > 0.5:
        prediction = "Customer will churn"
    else:
        prediction = "Customer will not churn"
        
    return {
        'prediction': prediction
    }