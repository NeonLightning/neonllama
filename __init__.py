import random
import subprocess
import requests
import time
import requests.exceptions
from pathlib import Path
from comfy.utils import ProgressBar
from tokenizers import Tokenizer
import lmstudio
import lmstudio as lms
import threading
import time

def fetch_all_llm_models():
    all_models = []
    ollama_url = "http://localhost:11434/api/tags"
    try:
        res = requests.get(ollama_url, timeout=5)
        res.raise_for_status()
        data = res.json()
        ollama_models = [m["model"] for m in data.get("models", [])]
        print(f"[Ollama] Fetched models: {ollama_models}")
        all_models.extend([f"ollama:{model}" for model in ollama_models])
    except requests.exceptions.ConnectionError:
        print("[Ollama] Failed to connect to Ollama. Is the server running on localhost:11434?")
    except Exception as e:
        print(f"[Ollama] Failed to fetch models: {e}")
    try:
        client = lms.Client()
        downloaded_lmstudio_models = client.list_downloaded_models()
        lmstudio_llm_model_keys = []
        if not downloaded_lmstudio_models:
            print("No downloaded models found in your LM Studio installation.")
        else:
            for model_obj in downloaded_lmstudio_models:
                if isinstance(model_obj, lms.DownloadedLlm):
                    lmstudio_llm_model_keys.append(model_obj.model_key)
        print(f"[LM Studio] Fetched LLM model keys: {lmstudio_llm_model_keys}")
        all_models.extend([f"lmstudio:{key}" for key in lmstudio_llm_model_keys])
    except requests.exceptions.ConnectionError:
        print("[LM Studio] Failed to connect to LM Studio. Is the server running on localhost:1234?")
    except Exception as e:
        print(f"[LM Studio] Failed to fetch LM Studio models: {e}")
    if not all_models:
        print("[Global Model Fetch] No models found from Ollama or LM Studio. Please check servers.")
        return ["No models found - ensure Ollama or LM Studio is running."]
    else:
        print(f"[Global Model Fetch] Combined available models: {all_models}")
        return all_models

def clear_ollama_model():
    try:
        result = subprocess.run(["ollama", "ps"], capture_output=True, text=True, check=False, timeout=5)
        if result.returncode != 0:
            print(f"[Ollama] 'ollama ps' command failed: {result.stderr.strip()}")
            return False
        lines = result.stdout.strip().splitlines()
        if len(lines) < 2 or "MODEL" not in lines[0]:
            print("[Ollama] No Ollama model is currently loaded or 'ollama ps' output is unexpected.")
            return False
        model_name = lines[1].split()[0]
        if not model_name:
            print("[Ollama] Could not determine Ollama model name from 'ollama ps' output.")
            return False
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "keep_alive": "0m"},
            timeout=10
        )
        res.raise_for_status()
        print(f"[Ollama] Unloaded model: {model_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Ollama] Error running 'ollama ps': {e}")
    except requests.exceptions.RequestException as e:
        print(f"[Ollama] Error sending unload request: {e}")
    except Exception as e:
        print(f"[Ollama] An unexpected error occurred during Ollama model clearing: {e}")
    return False

ALL_AVAILABLE_MODELS = fetch_all_llm_models()
try:
    lmstudio_model = lms.llm()
    lmstudio_model.unload()
    print("[LM Studio] Unloaded any previously running model at startup.")
except Exception as e:
    print(f"[LM Studio] No model to unload or error unloading at startup: {e}")
clear_ollama_model()
try:
    tokenizer = Tokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
except Exception as e:
    print(f"Warning: Could not load tokenizer. Prompt token estimation will not work: {e}")
    tokenizer = None

def estimate_tokens(text):
    if tokenizer:
        return len(tokenizer.encode(text).ids)
    else:
        return len(text.split())

class OllamaPromptFromIdea:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (ALL_AVAILABLE_MODELS, {"tooltip": "Select the LLM model (Ollama or LM Studio) to generate prompts with."}),
                "idea": ("STRING", {"multiline": True, "default": "futuristic cyberpunk city", "tooltip": "Enter the core concept or theme for your prompt.\nYou can have separated ideas if you have a hard return.\nOnly use up to 3 lines though, to a maximum of 231 tokens."}),
                "negative": ("STRING", {"multiline": True, "default": "", "tooltip": "Words or themes to exclude from the prompt (used by Stable Diffusion, not LLM).", "dynamicPrompts": False}),
                "max_tokens": ("INT", {"default": 75, "min": 10, "max": 1024, "tooltip": "Maximum token length for the generated prompt."}),
                "min_tokens": ("INT", {"default": 50, "min": 10, "max": 1024, "tooltip": "Minimum token length for the generated prompt."}),
                "max_attempts": ("INT", {"default": 30, "min": 1, "max": 200, "tooltip": "Number of attempts to generate a prompt fitting token limits."}),
                "regen_on_each_use": ("BOOLEAN", {"default": True, "tooltip": "Force regeneration on each node execution (doesn't matter if just_use_idea is on)."}),
                "just_use_idea": ("BOOLEAN", {"default": True, "tooltip": "Skip Generating and just use idea as prompt."}),
                "exclude_comma": ("BOOLEAN", {"default": False, "tooltip": "Disables commas and sentence removal suggesting."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("prompt", "negative", "idea")
    FUNCTION = "generate_prompt"
    CATEGORY = "LLM Prompts"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def generate_prompt(self, model, idea, negative, max_tokens, min_tokens, max_attempts, regen_on_each_use, just_use_idea, exclude_comma):
        PREDICTION_TIMEOUT = 120
        if just_use_idea:
            print("[LLM Prompt Node] 'Just Use Idea' is enabled. Skipping LLM generation.")
            return (idea, negative, idea)
        if not negative:
            negative = ""
        is_ollama_model = model.startswith("ollama:")
        is_lmstudio_model = model.startswith("lmstudio:")
        if not is_ollama_model and not is_lmstudio_model:
            print(f"[LLM Prompt Node] Error: Invalid model selection prefix for '{model}'. Falling back to idea.")
            return (idea, negative, idea)
        actual_model_name = model.split(":", 1)[1] if ":" in model else model
        print(f"[LLM Prompt Node] Using model: {actual_model_name} from {'Ollama' if is_ollama_model else 'LM Studio'}")
        idea_list = [i.strip() for i in idea.strip().split("\n") if i.strip()]
        total_attempts = len(idea_list) * max_attempts
        pbar = ProgressBar(total_attempts)
        if not idea_list:
            print("[LLM Prompt Node] No valid ideas provided. Returning empty prompt.")
            return ("", negative, "")
        generated_prompts = []
        prompt_log_file = Path("llm_generated_prompt_log.txt")
        if not regen_on_each_use:
            try:
                if prompt_log_file.exists():
                    with open(prompt_log_file, "r", encoding="utf-8") as f:
                        file_prompt = f.read().strip()
                    print("[LLM Prompt Node] Loaded prompt from file instead of regenerating.")
                    return (file_prompt, negative, idea)
                else:
                    print("[LLM Prompt Node] Prompt log file not found. Generating new prompt.")
            except Exception as e:
                print(f"[LLM Prompt Node] Failed to load cached prompt file: {e}. Generating new prompt.")
        lm_studio_llm_instance = None
        if is_lmstudio_model:
            try:
                lm_studio_client = lms.Client()
                lm_studio_llm_instance = lm_studio_client.llm.model(actual_model_name)
            except Exception as e:
                print(f"[LLM Prompt Node] Error initializing LM Studio client or model: {e}")
                return (idea, negative, idea)
        for idx, sub_idea in enumerate(idea_list):
            print(f"\n🧠 Generating prompt for idea {idx + 1}: '{sub_idea}'")
            last_output = None
            used_phrases = []
            timeout_number = 0
            if negative.strip():
                used_phrases.append(negative.strip())
            for attempt in range(1, max_attempts + 1):
                pbar.update_absolute((idx * max_attempts) + (attempt - 1))
                try:
                    avoid_text = " | ".join(used_phrases)
                    avoid_clause = ""
                    if avoid_text.strip():
                        avoid_clause = f"\nABSOLUTELY DO NOT use or repeat any of the following phrases or content: {avoid_text}." if avoid_text.strip() else ""
                    if exclude_comma:
                        system_message_content = (
                            f"You are a specialized prompt generator for Stable Diffusion XL. "
                            f"Your task is to convert a raw idea into a single-line, visually dense, and concrete image prompt. "
                            f"Use only short, descriptive fragments – no emotions, no abstract terms, no opinions. "
                            f"Focus strictly on visible elements: subject appearance, setting, lighting, objects, materials, and structure. "
                            f"Use connectors like 'with', 'under', 'surrounded by', but avoid excessive chaining. "
                            f"Sort ideas by visual importance, from main subject to secondary elements. "
                            f"The entire prompt must be between {min_tokens} and {max_tokens} tokens. "
                            f"{avoid_clause}"
                            f"DO NOT explain your reasoning. DO NOT change or reinterpret the original idea. Preserve it exactly: {sub_idea}"
                            f"Never use storytelling, feelings, or narrative context."
                            f"Your output must be suitable for direct input into an AI image generation model."
                            f"Give a new idea from any previous"
                        )
                    else:
                        system_message_content = (
                        f"You are a specialized prompt generator for Stable Diffusion XL. "
                        f"Your task is to convert a raw idea into a single-line, visually dense, and concrete image prompt. "
                        f"Use only short, descriptive fragments – no full sentences, no emotions, no abstract terms, no opinions. "
                        f"Focus strictly on visible elements: subject appearance, setting, lighting, objects, materials, and structure. "
                        f"Use connectors like 'with', 'under', 'surrounded by', but avoid excessive chaining. "
                        f"Sort ideas by visual importance, from main subject to secondary elements. "
                        f"Separate all ideas with commas. In order of importance."
                        f"The entire prompt must be between {min_tokens} and {max_tokens} tokens"
                        f"{avoid_clause}"
                        f"DO NOT explain your reasoning. DO NOT change or reinterpret the original idea. Preserve it exactly: {sub_idea}"
                        f"Never use storytelling, feelings, or narrative context."
                        f"Your output must be suitable for direct input into an AI image generation model."
                        f"Give a new idea from any previous"
                    )
                    user_message_content = f"Idea: {sub_idea}\nPrompt:"
                    if last_output is not None:
                        token_count_last_output = estimate_tokens(last_output)
                        if token_count_last_output > max_tokens:
                            user_message_content = (
                                f"\nOriginal prompt: {sub_idea}\n"
                                f"The following prompt is too long it needs to be slightly shorter."
                                f"Shorten it without removing detail. Compress phrases, remove redundancy. "
                                f"{avoid_clause}"
                                f"Never use storytelling, feelings, or narrative context."
                                f"Your output must be suitable for direct input into an AI image generation model."
                                f"Give a new idea from the previous"
                            )
                        elif token_count_last_output < min_tokens:
                            user_message_content = (
                                f"\nOriginal prompt: {sub_idea}\n"
                                f"The following prompt is too short it needs to be slightly longer.. "
                                f"Expand it with vivid, concrete visual details. Add setting, lighting, textures, or objects. "
                                f"Do NOT repeat. "
                                f"{avoid_clause}"
                                f"Never use storytelling, feelings, or narrative context."
                                f"Your output must be suitable for direct input into an AI image generation model."
                                f"Give a new idea from the previous"
                            )
                        else:
                            user_message_content = (
                                f"\nOriginal prompt: {sub_idea}\n"
                                f"Improve the following prompt for visual clarity and composition. "
                                f"Structure for better flow, but stay under {max_tokens} tokens. "
                                f"{avoid_clause}"
                                f"Never use storytelling, feelings, or narrative context."
                                f"Your output must be suitable for direct input into an AI image generation model."
                                f"Give a new idea from the previous"
                            )
                        system_message_content += avoid_clause
                    adaptive_temperature = 0.1 + (random.random() * (0.9 - 0.1))
                    raw_result = ""
                    if is_ollama_model:
                        api_url = "http://localhost:11434/api/generate"
                        random_seed = random.randint(0, 1000000000)
                        payload = {
                            "model": actual_model_name,
                            "prompt": system_message_content + "\n" + user_message_content,
                            "stream": False,
                            "options": {
                                "temperature": adaptive_temperature,
                                "seed": random_seed,
                            }
                        }
                        response = requests.post(
                            api_url,
                            json=payload,
                            timeout=PREDICTION_TIMEOUT
                        )
                        response.raise_for_status()
                        raw_result = response.json().get("response", "").strip()
                    elif is_lmstudio_model:
                        if lm_studio_llm_instance is None:
                            raise ConnectionError("LM Studio LLM instance not initialized for API call.")
                        chat = lms.Chat(system_message_content)
                        chat.add_user_message(user_message_content)
                        stream = lm_studio_llm_instance.respond_stream(chat, config={"repeatPenalty":1.3, "temperature":adaptive_temperature,}, on_message=chat.append)
                        cancelled = False
                        def timeout_handler():
                            nonlocal cancelled
                            cancelled = True
                            stream.cancel()
                        timer = threading.Timer(PREDICTION_TIMEOUT, timeout_handler)
                        timer.start()
                        try:
                            for _ in stream:
                                if cancelled:
                                    break
                            if not cancelled:
                                result_obj = stream.result()
                                if hasattr(result_obj, "text"):
                                    raw_result = result_obj.text.strip()
                                else:
                                    raw_result = str(result_obj).strip()
                            else:
                                raise TimeoutError("LM Studio response generation cancelled due to timeout.")
                        finally:
                            timer.cancel()
                    token_count = estimate_tokens(raw_result)
                    print(f"Idea: {idx + 1} Attempt: {attempt}/{max_attempts} LLM result: {raw_result}")
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
                        if is_ollama_model:
                            clear_ollama_model()
                        elif is_lmstudio_model and lm_studio_llm_instance:
                            model = lms.llm()
                            model.unload()
                        generated_prompts.append(raw_result)
                        break
                    last_output = raw_result
                    print(f"⚠️ Prompt out of bounds. Retrying...\n")
                    timeout_number = 0
                    time.sleep(0.5)
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 500:
                        print(f"⚠️ Attempt {attempt}/{max_attempts} HTTP 500 error. Retrying...\n")
                        if is_ollama_model:
                            clear_ollama_model()
                        time.sleep(1)
                        continue
                    else:
                        error_msg = f"[LLM API Error] HTTP Error {e.response.status_code}: {e.response.text}"
                        print(error_msg)
                        return (error_msg, negative, idea)
                except requests.exceptions.Timeout:
                    if timeout_number < 5:
                        print(f"⚠️ Attempt {attempt}/{max_attempts} timed out. Retrying...\n")
                        time.sleep(1)
                        timeout_number += 1
                        continue
                    else:
                        error_msg = f"Too many timeouts (5). Falling back to idea."
                        print(f"❌ {error_msg}")
                        generated_prompts.append(sub_idea)
                        timeout_number = 0
                        if is_ollama_model:
                            clear_ollama_model()
                        elif is_lmstudio_model and lm_studio_llm_instance:
                            model = lms.llm()
                            model.unload()
                        break
                except TimeoutError as e:
                    if timeout_number < 5:
                        print(f"⚠️ Attempt {attempt}/{max_attempts} LM Studio stream timeout. Retrying...\n")
                        time.sleep(1)
                        timeout_number += 1
                        continue
                    else:
                        error_msg = f"Too many LM Studio timeouts (5). Falling back to idea."
                        print(f"❌ {error_msg}")
                        generated_prompts.append(sub_idea)
                        timeout_number = 0
                        if is_lmstudio_model and lm_studio_llm_instance:
                            model = lms.llm()
                            model.unload()
                        break
                except Exception as e:
                    error_msg = f"[LLM Error] {str(e)}"
                    print(error_msg)
                    if is_ollama_model:
                        clear_ollama_model()
                    elif is_lmstudio_model and lm_studio_llm_instance:
                        model = lms.llm()
                        model.unload()
                    return (error_msg, negative, idea)
            else:
                print(f"❌ Max attempts for idea '{sub_idea}' reached. Using original as fallback.")
                timeout_number = 0
                if is_ollama_model:
                    clear_ollama_model()
                elif is_lmstudio_model and lm_studio_llm_instance:
                    model = lms.llm()
                    model.unload()
                generated_prompts.append(sub_idea)
        final_prompt = " BREAK ".join(generated_prompts)
        print(f"\nFinal Generated Prompt: {final_prompt}")
        pbar.update_absolute(max_attempts)
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
    "OllamaPromptFromIdea": "🧠 Ollama & LM Studio Prompt From Idea",
}