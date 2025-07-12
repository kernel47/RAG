from time import time
from llm_wrapper_cpp import LlamaCppLLM

def test_response(prompt: str):
    print(f"🔍 Prompt: {prompt}")
    model_path = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

    llm = LlamaCppLLM(
        model_path=model_path,
        n_ctx=2048,
        n_threads=4,    # adapte à ton CPU
        n_batch=8,
        temperature=0.1,
        max_tokens=256
    )

    t0 = time()
    output = llm._call(prompt)
    t1 = time()
    print(f"🧠 Réponse ({t1 - t0:.2f}s):\n{output.strip()}\n")

if __name__ == "__main__":
    test_response("Explique le code d'erreur 96 dans NetBackup et les causes possibles")
    test_response("Quelles sont les étapes d’un processus de restauration d’un fichier NAS avec NetBackup ?")
