import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


class PromptTemplate(Enum):
    BASIC_DIAGNOSIS = "basic_diagnosis"
    DIFFERENTIAL_DIAGNOSIS = "differential"
    TREATMENT_PLAN = "treatment"
    EMERGENCY_TRIAGE = "emergency"
    FOLLOW_UP = "followup"


@dataclass
class ConsultationRequest:
    symptoms: str
    age: Optional[int] = None
    gender: Optional[str] = None
    medical_history: Optional[str] = None
    current_medications: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 100


class AdvancedOpenRouterClient:
    def __init__(self, model):
        """
        Initialize the OpenRouterClient with the specified model.
        """
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.api_key = os.getenv("api_key")  # Access API key directly using os.getenv
        if not self.api_key:
            raise Exception(
                "API key not found. Please set it in Colab secrets with the name 'api_key'."
            )
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def build_system_prompt(self, template: PromptTemplate) -> str:
        """
        Build system prompt based on the selected template.
        """
        prompts = {
            PromptTemplate.BASIC_DIAGNOSIS: "You are a medical AI assistant. Provide concise diagnoses based on symptoms.",
            PromptTemplate.DIFFERENTIAL_DIAGNOSIS: "You are a medical AI assistant. Provide a list of possible diagnoses based on symptoms.",
            PromptTemplate.TREATMENT_PLAN: "You are a medical AI assistant. Suggest treatment plans based on diagnoses.",
            PromptTemplate.EMERGENCY_TRIAGE: "You are a medical AI assistant. Prioritize symptoms for emergency situations.",
            PromptTemplate.FOLLOW_UP: "You are a medical AI assistant. Suggest follow-up actions based on previous consultations.",
        }
        return prompts.get(template, prompts[PromptTemplate.BASIC_DIAGNOSIS])

    def generate_text_examples(self, template: PromptTemplate) -> List[Dict[str, str]]:
        """
        Provide example interactions based on the selected template.
        """
        examples = {
            PromptTemplate.BASIC_DIAGNOSIS: [
                {"role": "user", "content": "I have a headache and fever."},
                {
                    "role": "assistant",
                    "content": "You might have the flu or a common cold.",
                },
            ],
            PromptTemplate.DIFFERENTIAL_DIAGNOSIS: [
                {
                    "role": "user",
                    "content": "I have chest pain and shortness of breath.",
                },
                {
                    "role": "assistant",
                    "content": "Possible diagnoses include angina, heart attack, or pulmonary embolism.",
                },
            ],
            PromptTemplate.TREATMENT_PLAN: [
                {"role": "user", "content": "I was diagnosed with hypertension."},
                {
                    "role": "assistant",
                    "content": "A treatment plan may include lifestyle changes and medication.",
                },
            ],
            PromptTemplate.EMERGENCY_TRIAGE: [
                {"role": "user", "content": "I am experiencing severe abdominal pain."},
                {
                    "role": "assistant",
                    "content": "Seek immediate medical attention as this could be an emergency.",
                },
            ],
            PromptTemplate.FOLLOW_UP: [
                {
                    "role": "user",
                    "content": "I had a mild allergic reaction last week.",
                },
                {
                    "role": "assistant",
                    "content": "Follow up with your healthcare provider if symptoms persist or worsen.",
                },
            ],
        }
        return examples.get(template, examples[PromptTemplate.BASIC_DIAGNOSIS])

    def clinical_consultation(
        self, request: ConsultationRequest, template: PromptTemplate.BASIC_DIAGNOSIS
    ) -> dict:
        """
        Perform a clinical consultation based on patient data and selected template.
        """
        patient_symptoms = f"Patient Symptoms: {request.symptoms}"
        if request.age:
            patient_symptoms += f", Age: {request.age}"
        if request.gender:
            patient_symptoms += f", Gender: {request.gender}"
        if request.medical_history:
            patient_symptoms += f", Medical History: {request.medical_history}"
        if request.current_medications:
            patient_symptoms += f", Current Medications: {request.current_medications}"

        messages = [
            {"role": "system", "content": self.build_system_prompt(template)},
            *self.generate_text_examples(template),
            {"role": "user", "content": patient_symptoms},
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            diagnosis = response.json()["choices"][0]["message"]["content"]
            return {
                "Input symptoms": request.symptoms,
                "Recommended Diagnosis": diagnosis,
            }
        except requests.RequestException as e:
            print(f"Error occurred: {e}")
            return {"error": str(e)}


if __name__ == "__main__":
    client = AdvancedOpenRouterClient(model="openai/gpt-oss-20b:free")
    request = ConsultationRequest(
        symptoms="fever, cough, shortness of breath",
        age=30,
        gender="female",
        medical_history="asthma",
        current_medications="albuterol",
        max_tokens=100,
        temperature=0.7,
    )
    response = client.clinical_consultation(
        request, template=PromptTemplate.BASIC_DIAGNOSIS
    )
    print(response)
