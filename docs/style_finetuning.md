# Fine-Tuning for Style and Persona

Augmentoolkit is not just for factual recall; it is a powerful tool for capturing and reproducing specific writing styles, character voices, and narrative tones. This guide explains how to best leverage the different pipelines and features—including the new automatic anonymization tool—to create high-quality style-focused datasets.

## Core Pipelines for Style

### 1. RPToolkit (The Primary Style Pipeline)
The **RPToolkit** is the most direct way to capture style. It is designed to take source material (novels, scripts, chat logs) and generate multi-turn roleplay data that mimics the input's tone and character dynamics.

*   **How it works:** It extracts stylistic features, determines emotional undertones, and generates "scene cards" that guide the LLM to write in the specific style of the source text.
*   **Best for:** Character voices, narrative prose, dialogue-heavy interactions, and genre-specific tones (e.g., Lovecraftian horror, hard-boiled noir).
*   **Key Setting:** Ensure `include_chunk_in_prompt` is set to `True` in your config to give the generating LLM direct access to the stylistic nuances of the original text.

### 2. Representation Variation (For Pretraining)
If you want the model to deeply "inhale" a style before fine-tuning, use **Representation Variation**.

*   **How it works:** It takes source text and rewrites it into many different formats (e.g., transforming a narrative paragraph into a dry technical list or a vibrant blog post).
*   **Best for:** Training a model to understand a domain's concepts through the lens of a specific style during the continued pretraining phase.
*   **Outcome:** A model that is robust and doesn't "break character" when the format of the request changes.

### 3. Generic Data Rephrase (For Global Alignment)
To ensure your model doesn't lose its unique style when asked general questions, you must use **Generic Data Rephrase**.

*   **How it works:** It adds synthetic "Thought Processes" (Chain-of-Thought) to standard instruction datasets.
*   **Why it matters:** By making generic data follow the same structural format as your style-specific data, the model learns a unified way of responding, preventing it from reverting to a "default" personality during non-styled tasks.

---

## Safe Training with Anonymization

When training on personal letters, private chat logs, or proprietary documents to capture a specific "voice," privacy is paramount. Augmentoolkit now includes an **Automatic Anonymization** tool.

### Enabling Anonymization
You can enable this in any pipeline's configuration by adding:

```yaml
anonymize: True
```

### How it Works
The tool uses **Microsoft Presidio** (with an NLTK fallback) to detect and permanently replace:
*   **Names of People:** Replaced with `[PERSON]`.
*   **Cities and Locations:** Replaced with `[CITY]`.

This allows the model to learn the *patterns* of the writing—the rhythm of the sentences, the choice of vocabulary, and the emotional arcs—without ever seeing or memorizing sensitive real-world entities.

---

## Recommended Workflow for Style Fine-Tuning

1.  **Gather Source Texts:** Collect 50-500KB of high-quality text that represents the target style.
2.  **Anonymize & Pretrain (Optional):** Run the **Representation Variation** pipeline with `anonymize: True` to create a continued pretraining set.
3.  **Generate SFT Data:** Run **RPToolkit** with `anonymize: True` and `include_chunk_in_prompt: True`. This creates the "Instruct" portion of your dataset.
4.  **Align Generic Data:** Use **Generic Data Rephrase** on a small subset of a standard dataset (like OpenHermes) to ensure the model maintains its "thoughtful" structure across all tasks.
5.  **Train:** Use a training framework like Axolotl to fine-tune your base model on the resulting data.

## Tips for Success

*   **Model Selection:** For generating style data, use strong general-purpose models like **Llama 3.1 70B** or **DeepSeek-V3**. Avoid using "reasoning" models (like R1 or QwQ) for the generation steps themselves, as they can sometimes over-analyze the task instead of simply mimicking the style.
*   **Prompt Customization:** Don't hesitate to modify the prompts in the `prompts/` folder to emphasize specific stylistic elements you want the pipeline to focus on (e.g., "Use more sensory details" or "Keep sentences short and punchy").
*   **Review Debug Logs:** Check the `debug_outputs/` folder frequently. If the LLM is failing to capture the style, you may need to adjust your source text selection or your system prompts.


## Advanced: Scene Architectures (Structure -> Style)

If your goal is to train a model that takes a "Scene Architecture" (a structural plan including context, characters, and plot beats) and turns it into a finished piece of creative writing in a specific style, you can use the **Meta Datagen** feature of the RPToolkit.

### The "Architecture -> Story" Workflow

By default, RPToolkit generates a finished story from a "Scene Card." This Scene Card *is* the architecture. To turn this into a training dataset, you need to capture the inputs and outputs of these intermediate steps.

1.  **Enable Meta Datagen:** In your `config.yaml` for RPToolkit, enable meta-datagen:
    ```yaml
    do_meta_datagen: True
    meta_datagen_keys:
      - "scene_card_generation"
      - "story_generation"
    ```
2.  **Extract the Pairs:** This will cause Augmentoolkit to save the individual steps into `output_dir/meta_datagen/`. You will get:
    *   **Input:** The Scene Card (the "Architecture").
    *   **Output:** The generated Creative Writing (the "Finished Product").
3.  **Maintain Consistency:** Because you have enabled `anonymize: True`, both the architecture (names, cities) and the final prose will be anonymized consistently using the same placeholders, ensuring the model learns to associate the structural `[PERSON]` with the narrative `[PERSON]`.

### Multi-Dimensional Labeling
You can further enrich the "architecture" by adding custom keys to `to_include_features`. For example:
```yaml
to_include_features:
  - "Pacing"
  - "Sensory Details"
  - "Narrative Perspective"
  - "Thematic Undertones"
```
The RPToolkit will extract these from the source text and include them in the scene card, effectively creating a multi-dimensional structural prompt for your style-tuned model.
