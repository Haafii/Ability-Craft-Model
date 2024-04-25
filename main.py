from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load your pre-trained model
clf = joblib.load('decision_tree_model.pkl')  # Replace 'decision_tree_model.pkl' with the actual filename

app = FastAPI()

class InputData(BaseModel):
    basic_etiquette: float
    memory_test: float
    speech_test: float

@app.post("/predict/")
async def predict(data: InputData):
    new_inputs = [[data.basic_etiquette, data.memory_test, data.speech_test]]
    predicted_class = clf.predict(new_inputs)
    return {"predicted_class": predicted_class[0]}

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
