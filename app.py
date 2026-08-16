from pydantic import BaseModel, Field
from fastapi import FastAPI
from predict import predict

app = FastAPI(title='Crisis Report Triage API', version='1.0.0')

class MessageRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/predict')
def predict_message(request: MessageRequest):
    return predict(request.text)
