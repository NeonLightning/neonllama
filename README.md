# 🧠 NeonLLama ComfyUI Extension

**NeonLLama** is a custom ComfyUI node that transforms one or more **idea lines** into vivid, richly detailed prompts using a local [Ollama](https://ollama.com) LLM. It also supports **avoid** content, which the model takes into account to steer generation — and is returned as a **negative prompt**.

---

## 🚀 Features

- 🧠 Generates a structured, richly visual **positive prompt** from a simple idea.
- ✂️ Supports **multi-line idea inputs** — each line becomes its own generation.
- ⛔ Accepts an "avoid" list to influence prompt generation by avoiding unwanted terms.
- 🎯 Outputs the avoid list unmodified as the **negative prompt** (for Stable Diffusion, etc.).
- 🧮 Accurate token control using `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` tokenizer.
- 🔁 Retries intelligently until token limits are respected.
- ⚙️ Highly configurable: model choice, token ranges, max attempts, regen flag, and more.
- 🖨️ Returns all prompts in order and allows re-run on every use.

---

## 🧩 How It Works

1. You input one or more **ideas** (each on a new line).
2. You can optionally add **avoid terms** — things to steer the model away from.
3. The model uses your idea(s) to generate descriptive prompts:
   - **Positive Prompt**: Visually rich, structured, short-phrase based prompt.
   - **Negative Prompt**: Your avoid terms, returned exactly as you wrote them.
4. Prompt generation is token-aware and dynamically revised if too long or too short.
5. Each idea line gets its own generated prompt; all are merged and returned.

---

## 📤 Outputs

| Output | Type   | Description                                                                 |
|--------|--------|-----------------------------------------------------------------------------|
| `prompt` | STRING | Generated **positive prompt(s)**, joined with `BREAK` separators.         |
| `negative` | STRING | The **avoid list**, passed through directly as the negative prompt.      |
| `idea` | STRING | The original idea(s) input, for traceability or UI purposes.                |

---

## 📥 Inputs / Configuration Fields

| Name               | Type     | Description                                                                 |
|--------------------|----------|-----------------------------------------------------------------------------|
| `model`            | Dropdown | Select which locally running Ollama model to use.                          |
| `idea`             | Text     | The concept or image prompt idea. Up to 3 lines, one idea per line.        |
| `negative`         | Text     | Words, elements, or themes to avoid. Used during gen + passed as negative. |
| `max_tokens`       | Int      | Max token limit (default: 75, max: 231).                                   |
| `min_tokens`       | Int      | Minimum token requirement (default: 50).                                   |
| `max_attempts`     | Int      | Number of tries to get an acceptable prompt per idea.                      |
| `regen_on_each_use`| Bool     | Forces re-generation on each run, even if inputs didn’t change.            |

---

## 🧪 Example

**Inputs:**

```text
idea:
    haunted subway station with broken lights

avoid:
    blood, gore, screaming
```

**Outputs:**

```text
prompt:
    dark abandoned subway, flickering fluorescent lights, cracked tiled walls, shadowy corners, old train cars, graffiti-covered pillars, dim green glow, debris scattered floor

negative:
    blood, gore, screaming
```

---

## 🔁 Token-Aware Generation

- Uses CLIP tokenizer (`laion/CLIP-ViT-bigG-14-laion2B-39B-b160k`) to measure tokens.
- Tries up to `max_attempts` times to reach your `min_tokens` and not exceed `max_tokens`.
- If prompt is too short, it adds detail; too long, it trims or rephrases.
- Avoid terms are included in a non-generative system message and tracked between retries.

---

## 🧠 Smart Prompt Structuring

The generation system instructs the LLM to:

- Use **short phrases**, not sentences.
- **Avoid storytelling**, abstract ideas, and emotional language.
- Preserve the **core visual themes** of your original idea exactly.
- Avoid explanation or meta-commentary.
- Support basic structure with connectors like `with`, `under`, `surrounded by`.

---

## 🛠️ Requirements

- [Ollama](https://ollama.com) installed and running locally (on `localhost:11434`).
- ComfyUI environment.
- Python dependencies: `requests`, `tokenizers`.

---

## 📄 License

MIT License

---

## ❤️ Credits

Built by Neon Lightning ⚡