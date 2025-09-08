import os
from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import uvicorn
import torch



class SymptomsInput(BaseModel):
    symptoms: str


app = FastAPI(
    title="Clinical AI Assistant V1",
    description="An API to get a possible diagnosis based on a list of symptoms using a local Hugging Face model"
)

api_router = APIRouter(prefix='/v1')

@api_router.on_event('startup')
async def load_model():
    """
    Initialize AI prod pipeline
    """
    global text_generator
    try:
        print("Loading Hugging model")
        text_generator = pipeline("text-generation", model="distilgpt2")
        print("Model loaded sucessfully")
    except Exception as e:
        raise HTTPException(
            status_code=550, detail=f"Failed to laod model:{e}"
        )

@api_router.post('/diagnose')
async def get_medical_diagnoses(symptoms_data: SymptomsInput):
    """
    Provide diagnoases based on the list of systems given, using the local ai model"
    """
    
    try:
        symptoms = symptoms_data.symptoms
        prompt = f"Given the symptoms: '{symptoms}', what are some possible common diagnoses? Answer concisely:"
        response = text_generator(
            prompt,
            max_length=50,
            num_return_sequences=1,
            temperature=0.7
        )

        # Extract and return the generated text
        diagnosis = response[0]['generated_text'].strip()

        # Clean up the output to remove the initial prompt text
        cleaned_diagnosis = diagnosis.replace(prompt, "").strip()

        return {"Input sysmptoms": symptoms, "Recommended Diagnosis": cleaned_diagnosis}
    
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )

app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)