from llama_cpp import Llama
from llama_index.llms.base import LLM
from typing import Optional, List, Mapping, Any

class LlamaCppLLM(LLM):
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 4,
        n_batch: int = 8,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ):
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=n_batch,
            use_mlock=True,  # optionnel : évite que le modèle soit swappé
            verbose=False,
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        output = self.model(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop,
        )
        return output["choices"][0]["text"]

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {"name": "llama-cpp"}
