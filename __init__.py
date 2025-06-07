import requests
import time
import random
import requests.exceptions  
from tokenizers import Tokenizer
import os

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

tokenizer = Tokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

def estimate_tokens(text):
    return tokenizer.encode(text).ids

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
                "regen_on_each_use": ("BOOLEAN", {"default": True, "tooltip": "Force regeneration on each node execution."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("prompt", "negative", "idea")
    FUNCTION = "generate_prompt"
    CATEGORY = "Ollama"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
            return float("NaN")

    def generate_prompt(self, model, idea, negative, max_tokens, min_tokens, max_attempts, regen_on_each_use):
        if not negative:
            negative = ""
        token_min = min(min_tokens, max_tokens)
        token_expand_threshold = int(token_min * 0.75)

        idea_list = [i.strip() for i in idea.strip().split("\n") if i.strip()]
        generated_prompts = []

        prompt_log_file = "ollama_prompt_log.txt"

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
            if negative.strip():
                used_phrases.append(negative.strip())

            for attempt in range(1, max_attempts + 1):
                try:
                    avoid_text = " | ".join(used_phrases)
                    avoid_clause = ""
                    if avoid_text.strip() and (negative.strip() or idx > 0):
                        avoid_clause = f"\nABSOLUTELY avoid using or repeating any of the following phrases or content but keep them in mind: {avoid_text}"

                    if last_output is None:
                        system_prompt = (
                            f"Convert the following idea into a richly descriptive, visually detailed image prompt for Stable Diffusion XL. "
                            f"Use short phrases, and allow natural connectors like 'with', 'and', or 'under'. "
                            f"Focus on concrete, vivid visual elements – not abstract concepts. "
                            f"Use multi-word descriptions only where needed. "
                            f"Do not include full sentences, storytelling, or subjective opinions. "
                            f"Use only short descriptions. and don't describe feeling. "
                            f"Use an appropriate amount of commas to separate ideas for a image prompt. no excessive ideas."
                            f"Target between {token_min} and {max_tokens} tokens. "
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
                                f"Revise it to be shorter but keep visual richness and specificity. "
                                f"Use compact phrases or brief expressions with light structure. "
                                f"{avoid_clause}"
                                f"Reminder: You MUST preserve all core themes of the original idea. The original idea is: {sub_idea} DO NOT CHANGE THE IDEA."
                                f"\nPrevious prompt: {last_output}\nShorter prompt:"
                            )
                        elif token_count < token_expand_threshold:
                            system_prompt = (
                                f"The following prompt is too short (under {token_expand_threshold} tokens). "
                                f"Expand it by adding specific, vivid imagery using short but rich phrases. "
                                f"{avoid_clause}"
                                f"Reminder: You MUST preserve all core themes of the original idea. The original idea is: {sub_idea} DO NOT CHANGE THE IDEA."
                                f"\nPrevious prompt: {last_output}\nExpanded prompt:"
                            )
                        else:
                            system_prompt = (
                                f"Revise the following prompt to improve clarity and vividness, while keeping all original ideas intact. "
                                f"{avoid_clause}"
                                f"Reminder: You MUST preserve all core themes of the original idea. The original idea is: {sub_idea} DO NOT CHANGE THE IDEA."
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
                    print(f"Idea: {idx + 1} Attempt: {attempt}/{max_attempts} Ollama result: {raw_result}")
                    print(f"→ Token count: {token_count} (target: {token_min}–{max_tokens})")

                    if last_output is not None and raw_result.strip() == last_output.strip():
                        print("⚠️ Prompt identical to last attempt. Restarting generation from scratch...\n")
                        last_output = None
                        seed = random.randint(0, 99999999)
                        time.sleep(0.5)
                        continue

                    if token_min <= token_count <= max_tokens:
                        used_phrases.append(raw_result)
                        print("✔️ Prompt accepted.")
                        generated_prompts.append(raw_result)
                        break

                    last_output = raw_result
                    print(f"⚠️ Prompt out of bounds. Retrying...\n")
                    time.sleep(0.5)

                except requests.exceptions.Timeout:
                    print(f"⚠️ Attempt {attempt}/{max_attempts} timed out. Retrying...\n")
                    time.sleep(0.5)
                    continue
                except Exception as e:
                    error_msg = f"[Ollama Error] {str(e)}"
                    print(error_msg)
                    return (error_msg, negative, idea)

            else:
                print(f"❌ Max attempts for idea '{sub_idea}' reached. Using original as fallback.")
                generated_prompts.append(sub_idea)

        final_prompt = " BREAK ".join(generated_prompts)
        print(f"output of: {final_prompt}")

        # Save to file using w+ (overwrite)
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
