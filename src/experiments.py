import json
import os

from dotenv import load_dotenv

from src.prompts.advanced_prompting import (
    AdvancedOpenRouterClient,
    ConsultationRequest,
    PromptTemplate,
)

load_dotenv()


class PromptExperiment:
    def __init__(self, model: str):
        self.client = AdvancedOpenRouterClient(model)
        self.api_key = os.getenv("api_key")
        if not self.api_key:
            raise Exception(
                "API key not found. Please set it in Colab secrets with the name 'api_key'."
            )

    def temperature_experiment(self, patient_data: dict, temperatures: list):
        """
        Experiment with different temperature settings.
        """
        results = {}
        for temp in temperatures:
            request = ConsultationRequest(
                symptoms=patient_data.get("symptoms", ""),
                age=patient_data.get("age", ""),
                gender=patient_data.get("gender", ""),
                medical_history=patient_data.get("medical_history", ""),
                current_medications=patient_data.get("current_medications", ""),
                temperature=temp,
            )
            response = self.client.clinical_consultation(
                request, template=PromptTemplate.BASIC_DIAGNOSIS
            )
            results[temp] = response
        return results

    def max_tokens_experiment(self, patient_data: dict, max_tokens_list: list):
        """
        Experiment with different max token settings."""
        results = {}
        for max_tokens in max_tokens_list:
            request = ConsultationRequest(
                symptoms=patient_data.get("symptoms", ""),
                age=patient_data.get("age", ""),
                gender=patient_data.get("gender", ""),
                medical_history=patient_data.get("medical_history", ""),
                current_medications=patient_data.get("current_medications", ""),
                temperature=patient_data.get("temperature", 0.7),
                max_tokens=max_tokens,
            )
            response = self.client.clinical_consultation(
                request, template=PromptTemplate.BASIC_DIAGNOSIS
            )
            results[max_tokens] = response
        return results

    def template_experiment(self, patient_data: dict, templates: list):
        """
        Experiment with different prompt templates."""
        results = {}
        for template in templates:
            request = ConsultationRequest(
                symptoms=patient_data.get("symptoms", ""),
                age=patient_data.get("age", ""),
                gender=patient_data.get("gender", ""),
                medical_history=patient_data.get("medical_history", ""),
                current_medications=patient_data.get("current_medications", ""),
                temperature=patient_data.get("temperature", 0.7),
                max_tokens=patient_data.get("max_tokens", 1024),
            )
            response = self.client.clinical_consultation(request, template=template)
            results[template.name] = response
        return results

    def structured_vs_unstructured(self, patient_data: dict):
        """
        Compare structured vs unstructured prompts.
        """
        structured_request = ConsultationRequest(
            symptoms=patient_data.get("symptoms", ""),
            age=patient_data.get("age", ""),
            gender=patient_data.get("gender", ""),
            medical_history=patient_data.get("medical_history", ""),
            current_medications=patient_data.get("current_medications", ""),
            temperature=patient_data.get("temperature", 0.7),
            max_tokens=patient_data.get("max_tokens", 1024),
        )
        structured_response = self.client.clinical_consultation(
            structured_request, template=PromptTemplate.BASIC_DIAGNOSIS
        )
        unstructured_request = ConsultationRequest(
            symptoms=patient_data.get("symptoms", "")
        )
        unstructured_response = self.client.clinical_consultation(
            unstructured_request, template=PromptTemplate.BASIC_DIAGNOSIS
        )
        return {
            "structured": structured_response,
            "unstructured": unstructured_response,
        }


if __name__ == "__main__":
    experiment = PromptExperiment(model="openai/gpt-oss-20b:free")
    patient_data = {
        "symptoms": "fever, cough, shortness of breath",
        "age": 45,
        "gender": "male",
        "medical_history": "hypertension",
        "current_medications": "lisinopril",
        "temperature": 0.7,
        "max_tokens": 1024,
    }
    temperatures = [0.3, 0.7, 1.0]
    max_tokens_list = [50, 100, 200]
    templates = [
        PromptTemplate.BASIC_DIAGNOSIS,
        PromptTemplate.DIFFERENTIAL_DIAGNOSIS,
        PromptTemplate.EMERGENCY_TRIAGE,
        PromptTemplate.EMERGENCY_TRIAGE,
        PromptTemplate.TREATMENT_PLAN,
    ]

    # Run experiments
    temp_results = experiment.temperature_experiment(patient_data, temperatures)
    max_tokens_results = experiment.max_tokens_experiment(patient_data, max_tokens_list)
    template_results = experiment.template_experiment(patient_data, templates)
    structured_unstructured_results = experiment.structured_vs_unstructured(
        patient_data
    )

    # Print results
    print("Temperature Experiment Results:")
    for temp, result in temp_results.items():
        print(f"Temperature: {temp}, Result: {result}")

    print("\nMax Tokens Experiment Results:")
    for max_tokens, result in max_tokens_results.items():
        print(f"Max Tokens: {max_tokens}, Result: {result}")

    print("\nTemplate Experiment Results:")
    for template, result in template_results.items():
        print(f"Template: {template}, Result: {result}")

    print("\nStructured vs Unstructured Results:")
    for key, result in structured_unstructured_results.items():
        print(f"{key.capitalize()} Result: {result}")

    # save all results
    all_results = {
        "temperature_experiment": temp_results,
        "max_tokens_experiment": max_tokens_results,
        "template_experiment": template_results,
        "structured_vs_unstructured": structured_unstructured_results,
    }
    with open("experiment_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

    print("\nAll experiment results saved to 'experiment_results.json'")
