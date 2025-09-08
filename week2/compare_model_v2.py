import pandas as pd
from transformers import pipeline
from dotenv import load_dotenv
import os


load_dotenv()

import requests


# Load HuggingFace
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

# # Prepare test cases
test_cases = [
    {"input": "fever + cough", "context": "Fever and cough are symptoms of flu or malaria."},
    {"input": "chest pain", "context": "Chest pain can indicate a heart attack or pneumonia."},
]

# HuggingFace predictions
# hf_outputs = [generator(f"Possible disease for: {s}", max_length=50)[0]["generated_text"] for s in symptoms_list]
hf_outputs = []
for case in test_cases:
    hf_answer = qa_model(question=f"what are the possible disease for {case['input']}", context=case["context"])["answer"]
    hf_outputs.append(hf_answer)

print(hf_outputs)

# GPT predictions via OpenRouter
try:
    gpt_outputs = [] 
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    print("openrouter key is ",OPENROUTER_API_KEY)
    for case in test_cases:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "openai/gpt-oss-20b:free",
            "messages": [
                {"role": "system", "content": "You are a medical assistant powered by RAG."},
                {"role": "user", "content": f"what is Possible disease for: {case["input"]}? Context: {case["context"]}"},
            ],
            "temperature": 0.2,  # 🔹 Control creativity level dynamically
            "max_tokens": 100
        }

        response = requests.post(url, headers=headers, json=payload)
        response_json = response.json()

        # ✅ Extract model's answer
        try:
            answer = response_json["choices"][0]["message"]["content"]
            print(f"answer is ",answer)
            gpt_outputs.append(answer)
        except (KeyError, IndexError):
            answer = f"Error from OpenRouter: {response_json}"
except:
    gpt_outputs = ["N/A"] * len(test_cases)

print(gpt_outputs)

print(len(hf_outputs), len(gpt_outputs))
# Create comparison dataframe
df = pd.DataFrame({
    "Symptoms": [x["input"] for x in test_cases],
    "HuggingFace_Predictions": hf_outputs,
    "GPT_Predictions": gpt_outputs
})

# df.to_csv("clean/model_comparison.csv", index=False)
print(df)


