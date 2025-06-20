import random
import subprocess
import requests
import time
import requests.exceptions
from pathlib import Path
from comfy.utils import ProgressBar
import threading
import traceback

LMSTUDIO_AVAILABLE = False
lms = None
try:
    import lmstudio as lms
    from lmstudio.sync_api import Client
    LMSTUDIO_AVAILABLE = True
except ImportError:
    print("⚠️ LM Studio SDK not installed. LM Studio functionality disabled.")
ALL_AVAILABLE_MODELS = ["Loading models..."]
tokenizer = None
MODELS_LOADED = False

def initialize_in_background():
    global ALL_AVAILABLE_MODELS, tokenizer, MODELS_LOADED
    models = []
    try:
        res = requests.get("http://localhost:11434/api/tags", timeout=1.5)
        if res.status_code == 200:
            ollama_models = [m["model"] for m in res.json().get("models", [])]
            models.extend([f"ollama:{m}" for m in ollama_models])
            print(f"✅ Found {len(ollama_models)} Ollama models")
    except Exception as e:
        print(f"⚠️ Ollama model fetch skipped: {str(e)}")
    if LMSTUDIO_AVAILABLE and lms is not None:
        try:
            client = Client()
            result = []
            def fetch_models():
                try:
                    nonlocal result
                    downloaded_models = client.list_downloaded_models() or []
                    result = [
                        m.model_key for m in downloaded_models 
                        if isinstance(m, lms.DownloadedLlm)
                    ]
                except Exception as e:
                    print(f"LM Studio fetch error: {str(e)}")
                    result = []
            t = threading.Thread(target=fetch_models)
            t.daemon = True
            t.start()
            t.join(timeout=2.0)            
            if t.is_alive():
                print("⚠️ LM Studio model fetch timed out after 2 seconds")
            else:
                models.extend([f"lmstudio:{key}" for key in result])
                print(f"✅ Found {len(result)} LM Studio models")
        except Exception as e:
            print(f"⚠️ LM Studio model fetch failed: {str(e)}")
            traceback.print_exc()
    if models:
        ALL_AVAILABLE_MODELS = models
    else:
        ALL_AVAILABLE_MODELS = ["No models available - check Ollama/LM Studio"]
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
        print("✅ Tokenizer loaded successfully")
    except Exception as e:
        print(f"⚠️ Tokenizer load failed: {str(e)}")
        tokenizer = None
    MODELS_LOADED = True

threading.Thread(target=initialize_in_background, daemon=True).start()

class OllamaPromptFromIdea:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        llm_seed = random.randint(0, 1000000000)
        llm_temp = random.uniform(0.1, 1.0)
        return {
            "required": {
                "model": (ALL_AVAILABLE_MODELS, {"tooltip": "Select the LLM model (Ollama or LM Studio) to generate prompts with."}),
                "idea": ("STRING", {"multiline": True, "default": "futuristic cyberpunk city", "tooltip": "Enter the core concept or theme for your prompt.\nYou can have separated ideas if you have a hard return.\nOnly use up to 3 lines though, to a maximum of 231 tokens."}),
                "negative": ("STRING", {"multiline": True, "default": "", "tooltip": "Words or themes to exclude from the prompt (used by Stable Diffusion, not LLM).", "dynamicPrompts": False}),
                "max_tokens": ("INT", {"default": 75, "min": 10, "max": 1024, "tooltip": "Maximum token length for the generated prompt."}),
                "min_tokens": ("INT", {"default": 50, "min": 10, "max": 1024, "tooltip": "Minimum token length for the generated prompt."}),
                "max_attempts": ("INT", {"default": 30, "min": 1, "max": 200, "tooltip": "Number of attempts to generate a prompt fitting token limits."}),
                "regen_on_each_use": ("BOOLEAN", {"default": True, "tooltip": "Force regeneration on each node execution (doesn't matter if just_use_idea is on)."}),
                "just_use_idea": ("BOOLEAN", {"default": False, "tooltip": "Skip Generating and just use idea as prompt."}),
                "exclude_comma": ("BOOLEAN", {"default": False, "tooltip": "Disables commas and sentence removal suggesting."}),
                "randomize_seed": ("BOOLEAN", {"default": True, "tooltip": "Use a random seed on each generation."}),
                "llm_seed": ("INT", {"default": llm_seed, "min": 0, "max": 999999999, "tooltip": "Fixed seed (only used if randomize_seed is off)."}),
                "randomize_temp": ("BOOLEAN", {"default": True, "tooltip": "Use a random temperature on each generation."}),
                "llm_temp": ("FLOAT", {"default": llm_temp, "min": 0.1, "max": 1, "tooltip": "Fixed temperature (only used if randomize_temp is off)."}),
                "keepllm": ("BOOLEAN", {"default": False, "tooltip": "Keep LLM model in memory."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("prompt", "negative", "idea")
    FUNCTION = "generate_prompt"
    CATEGORY = "LLM Prompts"

    def generate_prompt(self, model, idea, negative, max_tokens, min_tokens, max_attempts, regen_on_each_use, just_use_idea, exclude_comma, randomize_seed, llm_seed, randomize_temp, llm_temp, keepllm):
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
                            f"You are a specialized prompt generator for Stable Diffusion. "
                            f"Your task is to convert a raw idea into a single-line, visually dense, and concrete image prompt. "
                            f"Use only short, descriptive fragments – no emotions, no abstract terms, no opinions. "
                            f"Focus strictly on visible elements: subject appearance, setting, lighting, objects, materials, and structure. "
                            f"Use connectors like 'with', 'under', 'surrounded by', but avoid excessive chaining. "
                            f"Sort ideas by visual importance, from main subject to secondary elements. "
                            f"The entire prompt must be between {min_tokens} and {max_tokens} tokens. "
                            f"{avoid_clause}"
                            f"DO NOT explain your reasoning or describe what you are doing. DO NOT reinterpret the original idea. Preserve it: {sub_idea}"
                            f"Never use storytelling, feelings, or narrative context."
                            f"Your output must be suitable for direct input into an AI image generation model."
                            f"Give a new idea from any previous"
                        )
                    else:
                        system_message_content = (
                        f"You are a specialized prompt generator for Stable Diffusion. "
                        f"Your task is to convert a raw idea into a single-line, visually dense, and concrete image prompt. "
                        f"Use only short, descriptive fragments – no full sentences, no emotions, no abstract terms, no opinions. "
                        f"Focus strictly on visible elements: subject appearance, setting, lighting, objects, materials, and structure. "
                        f"Use connectors like 'with', 'under', 'surrounded by', but avoid excessive chaining. "
                        f"Sort ideas by visual importance, from main subject to secondary elements. "
                        f"Separate all ideas with commas. In order of importance."
                        f"The entire prompt must be between {min_tokens} and {max_tokens} tokens"
                        f"{avoid_clause}"
                        f"DO NOT explain your reasoning or describe what you are doing. DO NOT reinterpret the original idea. Preserve it: {sub_idea}"
                        f"Never use storytelling, feelings, or narrative context."
                        f"Your output must be suitable for direct input into an AI image generation model."
                        f"Give a new idea from any previous"
                    )
                    user_message_content = f"Idea: {sub_idea}\nPrompt:"
                    if last_output is not None:
                        token_count_last_output = estimate_tokens(last_output)
                        if token_count_last_output > max_tokens:
                            user_message_content = (
                                f"DO NOT explain your reasoning or describe what you are doing."
                                f"The following prompt is too long it needs to be slightly shorter."
                                f"Shorten it without removing detail. Compress phrases, remove redundancy. "
                                f"{avoid_clause}"
                                f"Never use storytelling, feelings, or narrative context."
                                f"Your output must be suitable for direct input into an AI image generation model."
                                f"Give a new idea from the previous"
                                f"{last_output}"
                            )
                        elif token_count_last_output < min_tokens:
                            user_message_content = (
                                f"DO NOT explain your reasoning or describe what you are doing."
                                f"The following prompt is too short it needs to be slightly longer.. "
                                f"Expand it with vivid, concrete visual details. Add setting, lighting, textures, or objects. "
                                f"Do NOT repeat. "
                                f"{avoid_clause}"
                                f"Never use storytelling, feelings, or narrative context."
                                f"Your output must be suitable for direct input into an AI image generation model."
                                f"Give a new idea from the previous"
                                f"{last_output}"
                            )
                        else:
                            user_message_content = (
                                f"DO NOT explain your reasoning or describe what you are doing."
                                f"Improve the following prompt for visual clarity and composition. "
                                f"Structure for better flow, but stay under {max_tokens} tokens. "
                                f"{avoid_clause}"
                                f"Never use storytelling, feelings, or narrative context."
                                f"Your output must be suitable for direct input into an AI image generation model."
                                f"Give a new idea from the previous"
                                f"{last_output}"
                            )
                        system_message_content += avoid_clause
                    if randomize_temp:
                        llm_temp = random.uniform(0.1, 1.0)
                    raw_result = ""
                    if is_ollama_model:
                        api_url = "http://localhost:11434/api/generate"
                        if randomize_seed:
                            llm_seed = random.randint(0, 1000000000)
                        payload = {
                            "model": actual_model_name,
                            "prompt": system_message_content + "\n" + user_message_content,
                            "stream": False,
                            "options": {
                                "temperature": llm_temp,
                                "seed": llm_seed,
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
                        chat = lms.Chat()
                        chat.add_user_message(system_message_content + "\n" + user_message_content)
                        stream = lm_studio_llm_instance.respond_stream(chat, config={"repeatPenalty":1.1, "temperature":llm_temp, "seed": llm_seed,}, on_message=chat.append)
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
                        if keepllm == False:
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
                        if keepllm == False:
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
                        if keepllm == False:
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
                        if keepllm == False:
                            if is_lmstudio_model and lm_studio_llm_instance:
                                model = lms.llm()
                                model.unload()
                        break
                except Exception as e:
                    error_msg = f"[LLM Error] {str(e)}"
                    print(error_msg)
                    if keepllm == False:
                        if is_ollama_model:
                            clear_ollama_model()
                        elif is_lmstudio_model and lm_studio_llm_instance:
                            model = lms.llm()
                            model.unload()
                    return (error_msg, negative, idea)
            else:
                print(f"❌ Max attempts for idea '{sub_idea}' reached. Using original as fallback.")
                timeout_number = 0
                if keepllm == False:
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