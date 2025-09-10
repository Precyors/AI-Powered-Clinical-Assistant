from transformers import AutoModelForCausalLM, AutoTokenizer


class GPTOSSClient:
    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        """Initialize the GPT-OSS client with the specified model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_text(
        self,
        prompt: str,
        max_length: int = 50,
        num_return_sequences: int = 1,
        temperature: float = 0.7,
    ):
        """Generate text based on the given prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]

    def structured_consultation(self, patient_data: dict) -> dict:
        """Strutured consultation based on patient data."""
        symptoms = patient_data.get("symptoms", "")
        age = patient_data.get("age", "")
        gender = patient_data.get("gender", "")
        prompt = f"Given the symptoms: '{symptoms}, {age} and {gender}', what are some possible common diagnoses? Answer concisely:"
        responses = self.generate_text(prompt)
        # Clean up the output to remove the initial prompt text
        cleaned_diagnosis = responses[0].replace(prompt, "").strip()
        return {"Input symptoms": symptoms, "Recommended Diagnosis": cleaned_diagnosis}
