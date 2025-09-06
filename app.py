from config import settings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import openai


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """ You are an autonomous clinical assistant. 
        You must assist doctors in reasoning about diagnoses and treatment plans using medical knowledge. 
        Always cite your reasoning from retrieved sources and include a safety disclaimer. 
        Never make final clinical decisions."""),
        
        ("user", "Question: {input}"),
    ]
)

def generate_response(engine, temperature, max_tokens):
    llm = ChatGroq(api_key=settings.GROQ_API_KEY, model=engine, temperature=temperature, max_tokens=max_tokens)

    chain = prompt | llm

    return chain


if __name__ == "__main__":
    model= generate_response(engine="Gemma2-9b-It", temperature=0.7, max_tokens=1000)
    response = model.invoke({"input": "What is the treatment for hypertension?"})
    print(response.content)
    

