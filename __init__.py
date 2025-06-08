import subprocess
import requests
import time
import requests.exceptions
from pathlib import Path
from tokenizers import Tokenizer

def fetch_ollama_models():
    url = "http://localhost:11434/api/tags"
    try:
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        data = res.json()
        models = [m["model"] for m in data.get("models", [])]
        print(f"[Ollama] Fetched models: {models}")
        return models if models else [""]
    except Exception as e:
        print(f"[Ollama] Failed to fetch models: {e}")
        return [""]

OLLAMA_MODELS = fetch_ollama_models()

tokenizer = Tokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

def estimate_tokens(text):
    return tokenizer.encode(text).ids

def clear_ollama_model():
    try:
        result = subprocess.run(["ollama", "ps"], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().splitlines()
        if len(lines) < 2:
            print("No model is currently loaded.")
            return False
        model_name = lines[1].split()[0]
        if not model_name:
            print("Could not determine model name.")
            return False
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "keep_alive": 0},
            timeout=10
        )
        res.raise_for_status()
        print(f"Unloaded model: {model_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running 'ollama ps': {e}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending unload request: {e}")
    return False

clear_ollama_model()

class OllamaPromptFromIdea:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (OLLAMA_MODELS, {"tooltip": "Select the Ollama model to generate prompts with."}),
                "idea": ("STRING", {"multiline": True, "default": "futuristic cyberpunk city", "tooltip": "Enter the core concept or theme for your prompt to ollama\nYou can have seperated ideas if you have a hard return.\nOnly use up to 3 lines though. to a maximum. of 231 tokens."}),
                "negative": ("STRING", {"multiline": True, "default": "", "tooltip": "Words or themes to exclude from the prompt.(non ollama prompting)"}),
                "max_tokens": ("INT", {"default": 75, "min": 10, "max": 231, "tooltip": "Maximum token length for the generated prompt."}),
                "min_tokens": ("INT", {"default": 50, "min": 10, "max": 230, "tooltip": "Minimum token length for the generated prompt."}),
                "max_attempts": ("INT", {"default": 30, "min": 1, "max": 200, "tooltip": "Number of attempts to generate a prompt fitting token limits."}),
                "regen_on_each_use": ("BOOLEAN", {"default": True, "tooltip": "Force regeneration on each node execution.(doesn't matter if just_use_idea is on)"}),
                "just_use_idea": ("BOOLEAN", {"default": True, "tooltip": "Skip Generating and just use idea as prompt."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("prompt", "negative", "idea")
    FUNCTION = "generate_prompt"
    CATEGORY = "Ollama"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def generate_prompt(self, model, idea, negative, max_tokens, min_tokens, max_attempts, regen_on_each_use, just_use_idea):
        if just_use_idea:
            return (idea, negative, idea)
        if not negative:
            negative = ""
        idea_list = [i.strip() for i in idea.strip().split("\n") if i.strip()]
        generated_prompts = []
        prompt_log_file = Path("ollama_prompt_log.txt")
        if not regen_on_each_use:
            try:
                with open(prompt_log_file, "r", encoding="utf-8") as f:
                    file_prompt = f.read().strip()
                    print("[Ollama] Loaded prompt from file instead of regenerating.")
                    return (file_prompt, negative, idea)
            except FileNotFoundError:
                print("[Ollama] Prompt file not found. Falling back to input ideas.")
                fallback_prompt = " BREAK ".join(idea_list)
                return (fallback_prompt, negative, idea)
            except Exception as e:
                print(f"[Ollama] Failed to load cached prompt file: {e}")
                fallback_prompt = " BREAK ".join(idea_list)
                return (fallback_prompt, negative, idea)
        for idx, sub_idea in enumerate(idea_list):
            print(f"\n🧠 Generating prompt for idea {idx + 1}: '{sub_idea}'")
            last_output = None
            used_phrases = []
            timeout_number = 0
            if negative.strip():
                used_phrases.append(negative.strip())
            for attempt in range(1, max_attempts + 1):
                try:
                    avoid_text = " | ".join(used_phrases)
                    avoid_clause = ""
                    if avoid_text.strip() and (negative.strip() or idx > 0):
                        avoid_clause = f"\nABSOLUTELY DO NOT use or repeat any of the following phrases or content {avoid_text}"
                    if last_output is None:
                        system_prompt = (
                            f"Convert the following idea into a richly descriptive, visually detailed image prompt for Stable Diffusion XL. "
                            f"Use short phrases, and allow natural connectors like 'with', 'and', or 'under'. "
                            f"Focus on concrete, vivid visual elements – not abstract concepts. "
                            f"Use multi-word descriptions only where needed. "
                            f"Do not include full sentences, storytelling, or subjective opinions. "
                            f"Use only short descriptions. and don't describe feeling. "
                            f"Use an appropriate amount of commas to separate ideas for a image prompt. no excessive ideas."
                            f"Target between {min_tokens} and {max_tokens} tokens and all on one line."
                            f"{avoid_clause}"
                            f"Reminder: You MUST preserve all core themes of the original idea. The original idea is: {sub_idea} DO NOT CHANGE THE IDEA."
                            f"you MUST NOT ever talk about your thought process or explain how you generated the prompt."
                            f"\nIdea: {sub_idea}\nPrompt:"
                        )
                    else:
                        token_count = len(estimate_tokens(last_output))
                        if token_count > max_tokens:
                            system_prompt = (
                                f"The following prompt is too long (over {max_tokens} tokens). "
                                f"Revise it to be shorter but keep visual richness and specificity and all on one line."
                                f"Use compact phrases or brief expressions with light structure. "
                                f"{avoid_clause}"
                                f"you MUST NOT ever talk about your thought process or explain how you generated the prompt."
                                f"Reminder: You MUST preserve all core themes of the original idea. The original idea is: {sub_idea} DO NOT CHANGE THE IDEA."
                                f"\nPrevious prompt: {last_output}\nShorter prompt:"
                            )
                        elif token_count < min_tokens:
                            system_prompt = (
                                f"The following prompt is too short (under {min_tokens} tokens). "
                                f"Expand it by adding specific, vivid imagery using short but rich phrases and all on one line."
                                f"{avoid_clause}"
                                f"you MUST NOT ever talk about your thought process or explain how you generated the prompt."
                                f"Reminder: You MUST preserve all core themes of the original idea. The original idea is: {sub_idea} DO NOT CHANGE THE IDEA."
                                f"\nPrevious prompt: {last_output}\nExpanded prompt:"
                            )
                        else:
                            system_prompt = (
                                f"Revise the following prompt to improve clarity and vividness, while keeping all original ideas intact and all on one line."
                                f"{avoid_clause}"
                                f"Reminder: You MUST preserve all core themes of the original idea. The original idea is: {sub_idea} DO NOT CHANGE THE IDEA."
                                f"\nPrevious prompt: {last_output}\nRevised prompt:"
                            )
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": model,
                            "prompt": system_prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.3
                            }
                        },
                        timeout=60,
                    )
                    response.raise_for_status()
                    raw_result = response.json().get("response", "").strip()
                    token_count = len(estimate_tokens(raw_result))
                    print(f"Idea: {idx + 1} Attempt: {attempt}/{max_attempts} Ollama result: {raw_result}")
                    print(f"→ Token count: {token_count} (target: {min_tokens}–{max_tokens})")
                    if last_output is not None and raw_result.strip() == last_output.strip():
                        print("⚠️ Prompt identical to last attempt. Restarting generation from scratch...\n")
                        last_output = None
                        timeout_number = 0
                        time.sleep(0.5)
                        continue
                    if min_tokens <= token_count <= max_tokens:
                        used_phrases.append(raw_result)
                        print("✔️ Prompt accepted.")
                        timeout_number = 0
                        clear_ollama_model()
                        generated_prompts.append(raw_result)
                        break
                    last_output = raw_result
                    print(f"⚠️ Prompt out of bounds. Retrying...\n")
                    timeout_number = 0
                    time.sleep(0.5)
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 500:
                        print(f"⚠️ Attempt {attempt}/{max_attempts} 500 error. Retrying...\n")
                        clear_ollama_model()
                        time.sleep(1)
                        continue                    
                except requests.exceptions.Timeout:
                    if timeout_number < 6:
                        print(f"⚠️ Attempt {attempt}/{max_attempts} timed out. Retrying...\n")
                        time.sleep(1)
                        timeout_number += 1
                        continue
                    else:
                        error_msg = f"Too many timeouts (5). Falling back to idea."
                        generated_prompts.append(sub_idea)
                        timeout_number = 0
                        break
                except Exception as e:
                    error_msg = f"[Ollama Error] {str(e)}"
                    print(error_msg)
                    return (error_msg, negative, idea)
            else:
                print(f"❌ Max attempts for idea '{sub_idea}' reached. Using original as fallback.")
                timeout_number = 0
                clear_ollama_model()
                generated_prompts.append(sub_idea)
        final_prompt = " BREAK ".join(generated_prompts)
        print(f"output of: {final_prompt}")
        try:
            with open(prompt_log_file, "w+", encoding="utf-8") as f:
                f.write(final_prompt)
        except Exception as log_error:
            print(f"⚠️ Failed to write prompt to file: {log_error}")
        return (final_prompt, negative, idea)

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