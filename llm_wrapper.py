from ctransformers import AutoModelForCausalLM
from llama_index.llms.base import LLM
from typing import Optional, List, Mapping, Any

class CTransformersLLM(LLM):
    def __init__(self, model_path: str, max_new_tokens: int = 512, temperature: float = 0.1):
        self.model = AutoModelForCausalLM(model_path)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Génération de texte avec ctransformers
        output = self.model.generate(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            stop=stop
        )
        return output

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {"name": "ctransformers-llm"}
