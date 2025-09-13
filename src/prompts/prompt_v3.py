import os
from unittest import result

import requests
from dotenv import load_dotenv

load_dotenv()


class OpenRouterClient:
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

    def generate_text(
        self, prompt: str, max_tokens: int = 100, temperature: float = 0.7
    ):
        """
        Generate text using the OpenRouter API.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        response = requests.post(self.api_url, headers=headers, json=data)
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]

    def structured_consultation(self, patient_data: dict):
        """
        Perform a structured consultation based on patient data.
        """
        symptoms = patient_data.get("symptoms", "")
        age = patient_data.get("age", "")
        gender = patient_data.get("gender", "")

        prompt = f"Given the symptoms: '{symptoms}', age: '{age}', and gender: '{gender}', what are some possible common diagnoses? Answer concisely:"

        try:
            diagnosis = self.generate_text(prompt)  # Call generate_text on the instance

            return {"Input symptoms": symptoms, "Recommended Diagnosis": diagnosis}

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return {
                "Input symptoms": symptoms,
                "Recommended Diagnosis": "Error: Could not retrieve diagnosis.",
            }

    def Model_Testing():
        """
        Test the OpenRouterClient with sample patient data.
        """
        client = OpenRouterClient(model="openai/gpt-oss-20b:free")
        patient_data = {
            "symptoms": "fever, cough, shortness of breath",
            "age": 45,
            "gender": "male",
        }
        result = client.structured_consultation(patient_data)
        return result


if __name__ == "__main__":
    result = OpenRouterClient.Model_Testing()
    print(result)
