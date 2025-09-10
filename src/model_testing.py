from src.prompt_v2 import GPTOSSClient


def test_clinical_ai_assistant():
    client = GPTOSSClient()
    patient_data = {
        "symptoms": "fever, cough, shortness of breath",
        "age": 45,
        "gender": "male",
    }
    result = client.structured_consultation(patient_data)

    return result


if __name__ == "__main__":
    print(test_clinical_ai_assistant())
