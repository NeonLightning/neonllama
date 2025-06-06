import requests
import time
import random
import requests.exceptions  
from tokenizers import Tokenizer

seed = random.randint(0, 99999999)

def fetch_ollama_models():
    url = "http://localhost:11434/api/tags"
    try:
        res = requests.get(url, timeout=2)
        res.raise_for_status()
        data = res.json()
        models = [m["model"] for m in data.get("models", [])]
        print(f"[Ollama] Fetched models: {models}")
        return models if models else [""]
    except Exception as e:
        print(f"[Ollama] Failed to fetch models: {e}")
        return [""]

OLLAMA_MODELS = fetch_ollama_models()

tokenizer = Tokenizer.from_pretrained("openai/clip-vit-base-patch32")

def estimate_tokens(text):
    return tokenizer.encode(text).ids

class OllamaPromptFromIdea:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (OLLAMA_MODELS, {"description": "Select the Ollama model to generate prompts with."}),
                "idea": ("STRING", {"multiline": True, "default": "futuristic cyberpunk city", "description": "Enter the core concept or theme for your prompt to ollama."}),
                "avoid": ("STRING", {"multiline": True, "default": "", "description": "Words or themes to exclude from the prompt.(non ollama prompting)"}),
                "max_tokens": ("INT", {"default": 77, "min": 10, "max": 304, "description": "Maximum token length for the generated prompt."}),
                "min_tokens": ("INT", {"default": 60, "min": 10, "max": 140, "description": "Minimum token length for the generated prompt."}),
                "max_attempts": ("INT", {"default": 30, "min": 1, "max": 150, "description": "Number of attempts to generate a prompt fitting token limits."}),
                "regen_on_each_use": ("BOOLEAN", {"default": True, "description": "Force regeneration on each node execution."}),
            }
        }
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("prompt", "avoid",)
    FUNCTION = "generate_prompt"
    CATEGORY = "Ollama"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs.get("regen_on_each_use", True):
            return float("NaN")
        return None
    
    def generate_prompt(self, model, idea, avoid, max_tokens, min_tokens, max_attempts, regen_on_each_use):
        if not avoid:
            avoid = ""
        token_min = min(min_tokens, max_tokens)
        token_expand_threshold = int(token_min * 0.75)
        last_output = None
        avoid_clause = f"\nABSOLUTELY avoid mentioning: {avoid.strip()}" if avoid.strip() else ""
        for attempt in range(1, max_attempts + 1):
            try:
                if last_output is None:
                    system_prompt = (
                        f"Convert the following idea into a richly descriptive, visually detailed image prompt for Stable Diffusion XL. "
                        f"Use short but expressive phrases, and allow natural connectors like 'with', 'and', or 'under'. "
                        f"Focus on concrete, vivid visual elements – not abstract concepts. "
                        f"Use multi-word descriptions where appropriate. "
                        f"Do not include full sentences, storytelling, or subjective opinions. "
                        f"Use only short descriptions. and don't describe feeling"
                        f"Avoid overly generic or disconnected terms. "
                        f"Target between {token_min} and {max_tokens} tokens. "
                        f"{avoid_clause}"
                        f"Reminder: You MUST preserve all core themes of the original idea. The original idea is: {idea} DO NOT CHANGE THE IDEA."
                        f"you MUST NOT ever talk about your thought process or explain how you generated the prompt."
                        f"\nIdea: {idea}\nPrompt:"
                    )
                else:
                    token_count = len(estimate_tokens(last_output))
                    if token_count > max_tokens:
                        system_prompt = (
                            f"The following prompt is too long (over {max_tokens} tokens). "
                            f"Revise it to be shorter but keep visual richness and specificity. "
                            f"Use compact phrases or brief expressions with light structure. "
                            f"Avoid long sentences or reinterpreting the concept. "
                            f"Use only short descriptions. and don't describe feeling"
                            f"{avoid_clause}"
                            f"Reminder: You MUST preserve all core themes of the original idea. The original idea is: {idea} DO NOT CHANGE THE IDEA."
                            f"you MUST NOT ever talk about your thought process or explain how you generated the prompt."
                            f"\nPrevious prompt: {last_output}\nShorter prompt:"
                        )
                    elif token_count < token_expand_threshold:
                        system_prompt = (
                            f"The following prompt is too short (under {token_expand_threshold} tokens). "
                            f"Expand it by adding specific, vivid imagery using short but rich phrases. "
                            f"Include unique textures, lighting effects, environments, and visual motifs. "
                            f"Light structure is allowed: use connectors like 'with', 'under', 'surrounded by', etc. "
                            f"Do not repeat phrases or rearrange words – add new coherent, visual material. "
                            f"Use only short descriptions. and don't describe feeling"
                            f"Avoid full sentences or storylines. "
                            f"{avoid_clause}"
                            f"Reminder: You MUST preserve all core themes of the original idea. The original idea is: {idea} DO NOT CHANGE THE IDEA."
                            f"you MUST NOT ever talk about your thought process or explain how you generated the prompt."
                            f"\nPrevious prompt: {last_output}\nExpanded prompt:"
                        )
                    else:
                        system_prompt = (
                            f"Revise the following prompt to improve clarity and vividness, while keeping all original ideas intact. "
                            f"You may slightly structure the phrases for better flow. "
                            f"Do not add new concepts or remove core elements. "
                            f"Use only short descriptions. and don't describe feeling"
                            f"Keep it between {token_min}–{max_tokens} tokens. "
                            f"{avoid_clause}"
                            f"Reminder: You MUST preserve all core themes of the original idea. The original idea is: {idea} DO NOT CHANGE THE IDEA."
                            f"you MUST NOT ever talk about your thought process or explain how you generated the prompt."
                            f"\nPrevious prompt: {last_output}\nRevised prompt:"
                        )
                seed = random.randint(0, 99999999)
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": model,
                        "prompt": system_prompt,
                        "stream": False,
                        "options": {
                            "seed": seed,
                            "temperature": 0.7
                        }
                    },
                    timeout=120,
                )
                response.raise_for_status()
                raw_result = response.json().get("response", "").strip()
                token_count = len(estimate_tokens(raw_result))
                print(f"Attempt {attempt} Ollama result: {raw_result}")
                print(f"→ Token count: {token_count} (target: {token_min}–{max_tokens})")
                if last_output is not None and raw_result.strip() == last_output.strip():
                    print("⚠️ Prompt identical to last attempt. Restarting generation from scratch...\n")
                    last_output = None
                    seed = random.randint(0, 99999999)
                    time.sleep(0.5)
                    continue
                if token_min <= token_count <= max_tokens:
                    print("✔️ Prompt accepted.")
                    return (raw_result, avoid)
                last_output = raw_result
                reason = "too long" if token_count > max_tokens else "too short"
                print(f"⚠️ Prompt {reason}. Retrying...\n")
                time.sleep(0.5)
            except requests.exceptions.Timeout:
                print(f"⚠️ Attempt {attempt} timed out. Retrying...\n")
                time.sleep(0.5)
                continue
            except Exception as e:
                error_msg = f"[Ollama Error] {str(e)}"
                print(error_msg)
                return (error_msg, "")
        print("❌ Max attempts reached. Returning fallback idea.")
        return (idea, avoid)

    def ui(self, inputs, outputs):
        prompt_str = outputs[0] if isinstance(outputs, (list, tuple)) and outputs else ""
        return {
            "prompt": f"🧠 Generated Prompt:\n{prompt_str}"
        }

NODE_CLASS_MAPPINGS = {
    "OllamaPromptFromIdea": OllamaPromptFromIdea,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaPromptFromIdea": "🧠 Ollama Prompt From Idea",
}
