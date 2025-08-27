from transformers import pipeline
import pandas as pd

# HuggingFace model
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Sample test cases
test_cases = [
    {"input": "fever + cough", "context": "Fever and cough are symptoms of flu or malaria."},
    {"input": "chest pain", "context": "Chest pain can indicate a heart attack or pneumonia."},
]

results = []
for case in test_cases:
    hf_answer = qa_model(question=case["input"], context=case["context"])["answer"]
    gpt_answer = "Likely flu or malaria" if "fever" in case["input"] else "Possibly heart issue"
    results.append({
        "Symptoms": case["input"],
        "HuggingFace": hf_answer,
        "GPT (Simulated)": gpt_answer
    })

df = pd.DataFrame(results)
df.to_csv("clean/model_comparison.csv", index=False)
print(df)
