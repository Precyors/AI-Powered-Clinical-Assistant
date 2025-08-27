from transformers import pipeline

# Load a lightweight medical QA model from HuggingFace
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

context = """
Fever and cough are common symptoms of influenza and sometimes malaria.
Headache and dizziness may indicate high blood pressure or dehydration.
"""
question = "What disease can cause fever and cough?"

result = qa_model(question=question, context=context)
print("🤖 HuggingFace Answer:", result["answer"])
