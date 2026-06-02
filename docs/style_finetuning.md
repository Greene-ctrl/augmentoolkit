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

---

## Expanded Feature Sets: Architecture → Creative Writing by Format

The following sections define production-ready `to_include_features` sets for three creative writing formats: **Book Chapter** (priority), **Theater Scene**, and **Film/Screen Scene**. Each feature can be extracted from a finished text by an annotation LLM, then used as the labeled input side of a training pair — the original text becomes the output side.

### How This Extends the Architecture → Story Workflow

```
Finished Chapter (source text)
        │
        ▼  [LLM annotation pass — fills features below]
Labeled Architecture  ←  beat fields + to_include_features labels
        │
        ▼
Training pair:  Input  = labeled architecture
                Output = original finished prose (anonymized)
```

Run the annotation pass with `anonymize: True` so `[PERSON]` and `[CITY]` are consistent across both sides of every pair.

---

### Book Chapter Features (Priority)

#### Structure

| Feature | Values | What to extract |
|---------|--------|-----------------|
| `scene_count` | integer | Count distinct scene breaks |
| `beat_count` | integer | Count dramatic turns (goal shifts) |
| `chapter_length_category` | short (<3K) / medium (3–7K) / long (>7K) / epic (>12K) | Word count |
| `opening_beat_type` | action / dialogue / description / reflection / in_medias_res / image | First 1–3 paragraphs |
| `closing_beat_type` | hard_cut / echo_close / cliffhanger / image_landing / reflection_landing / revelation | Final 1–2 paragraphs |
| `structural_arc_shape` | linear / V_down_up / plateau_drop / ramp / descend / spike | Emotional trajectory |
| `chapter_function` | setup / escalation / crisis / pivot / climax / resolution / breather | Role in story |
| `time_span` | minutes / hours / days / longer | Diegetic time elapsed |
| `narrative_velocity` | compressed / real_time / stretched / mixed | Story-time vs. page-time ratio |

#### POV and Narrative Voice

| Feature | Values | Extraction note |
|---------|--------|-----------------|
| `pov_mode` | first_person / close_third / deep_third / distant_third / omniscient | Person + distance |
| `pov_distance` | intimate / moderate / distant | How far inside the character's head |
| `narrative_tense` | past / present | Verb tense |
| `interior_monologue_density` | none / sparse / moderate / saturated | Proportion of inner thought |
| `pov_unreliability` | none / mild / strong + description | Does POV character misread events? |
| `wound_filter_active` | yes / no + description | Is a core wound distorting perception? |
| `narrator_stance` | involved / ironic / sympathetic / clinical / wry | Tonal relationship to events |
| `free_indirect_discourse` | absent / occasional / dominant | "He knew she was lying" — between thought and narration |

#### Prose Style

| Feature | Values | Extraction note |
|---------|--------|-----------------|
| `sentence_rhythm` | short_punchy / long_periodic / mixed / fragment_heavy | Avg sentence length + variance |
| `sentence_average_length` | 5–10 / 10–15 / 15–20 / 20+ words | Count |
| `paragraph_rhythm` | tight_dense / medium_balanced / long_flowing / variable | Avg paragraph length |
| `paragraph_opening_style` | action_first / description_first / dialogue_first / reflection_first | First word type per paragraph |
| `white_space_strategy` | dense_blocks / airy_short / single_sentence_breaks / mixed | Structural use of blank space |
| `information_position` | front_loaded / back_loaded / balanced | Where key info sits in sentences |
| `direct_address` | never / rare / chapter_close_only / recurring | Narrator speaks to reader |

#### Dialogue

| Feature | Values | Extraction note |
|---------|--------|-----------------|
| `dialogue_ratio` | low (<25%) / medium (25–50%) / high (50–70%) / dominant (>70%) | Word-count proportion |
| `dialogue_style` | direct_speech_only / interleaved_with_action / interleaved_with_thought / layered | How dialogue is embedded |
| `speech_tag_style` | plain_said / varied_descriptive / action_beat_only / mixed | Attribution method |
| `subtext_ratio` | low / medium / high / total | Proportion with a second meaning |
| `silence_as_dialogue` | yes / no + description | Strategic non-responses |
| `voice_differentiation` | low / medium / high / extreme | How distinct each character's voice is |
| `dialogue_subtext_method` | deflection / counterquestion / subject_change / understatement / silence | Dominant technique |

#### Sensory and Imagery

| Feature | Values | Extraction note |
|---------|--------|-----------------|
| `dominant_sensory_modality` | visual / auditory / tactile / olfactory / gustatory / proprioceptive | Primary sense |
| `sensory_distribution` | vision_dominant / balanced_2 / balanced_3 / all_five | Which senses are active |
| `imagery_type` | concrete / symbolic / mythological / natural / architectural / bodily | Dominant image category |
| `color_register` | warm / cool / neutral / high_contrast / desaturated / vivid_specific | Color palette |
| `spatial_register` | claustrophobic / intimate / expansive / grand / ambiguous | How space is framed |
| `light_register` | bright / dim / torchlit / shadowplay / overexposed / neutral | Dominant light quality |
| `tactile_density` | absent / light / moderate / heavy | Touch/texture/weight in prose |
| `sensory_anchor_count` | integer per scene | Number of distinct anchors |

#### Figurative Language

| Feature | Values | Extraction note |
|---------|--------|-----------------|
| `figurative_density` | none / sparse (1/page) / moderate (2–4) / dense (5+) | Per-page count |
| `figurative_types` | list: simile / metaphor / personification / transferred_epithet / synecdoche / zeugma | Devices present |
| `comparison_style` | explicit_simile / embedded_metaphor / synesthetic / transferred_epithet | Dominant form |
| `sound_devices_used` | list: alliteration / assonance / consonance / anaphora / epistrophe / none | Phonetic patterning |
| `image_consistency` | isolated_images / consistent_field / returning_motif / transformed_motif | Whether images form a pattern |
| `abstraction_level` | concrete_only / concrete_with_abstract / abstraction_dominant | Tangible vs. conceptual balance |

#### Pacing

| Feature | Values | Extraction note |
|---------|--------|-----------------|
| `pacing_profile` | per-scene: slow / medium / fast + 1-sentence rationale | Scene-by-scene rhythm |
| `tension_arc` | ramp_and_hold / spike_release / slow_build / plateau / drop_and_recover | Tension shape |
| `time_compression_events` | list of summary moments | "Two weeks passed." — where and why |
| `slow_motion_moments` | list of stretched beats | Where a second takes a paragraph |
| `scene_transition_style` | hard_cut / white_space / bridge_line / echo_transition / summary_skip | Dominant method |
| `chapter_entry_velocity` | immediate_action / slow_establishing / mid_scene_drop | How fast reader is placed in scene |
| `chapter_exit_momentum` | complete_stop / forward_pull / echo_landing / question_left_open | Reader's state at close |

#### Thematic and Emotional

| Feature | Values | Extraction note |
|---------|--------|-----------------|
| `theme_embodiment_method` | stated / objectified / enacted / withheld / contradicted_by_action | How theme appears |
| `emotional_arc_shape` | per POV: flat / ascending / descending / V / spike / oscillating | Trajectory |
| `irony_type` | dramatic / verbal / situational / none | Type present |
| `dramatic_irony_level` | reader_ahead / reader_even / reader_behind | Information asymmetry |
| `tonal_register` | tragic / comic / wry / elegiac / tense / tender / cold / mythic | Dominant tone |
| `tonal_shifts` | list: beat + what tone shifts to | Tonal arc |
| `subtext_layer_count` | integer (0–4) | Concurrent subtexts |
| `power_register` | ascending / descending / contested / stable | POV character's power trajectory |

#### Chapter Motif and Echoes

| Feature | Values | Extraction note |
|---------|--------|-----------------|
| `chapter_motif` | concrete object/image established and returned | Opening image + transformation at close |
| `motif_appearance_count` | integer | How many times it recurs |
| `motif_transformation` | same / degraded / elevated / inverted | How motif changes |
| `callback_density` | sparse / moderate / dense | Calls back to prior chapters |
| `callback_list` | list: what is echoed + where it originated | Prior-text references |
| `planted_seed_count` | integer | New foreshadowing moments |
| `foreshadowing_method` | object / dialogue / action / omission / image / juxtaposition | How seeds are planted |

#### Historical / Setting Register

| Feature | Values | Extraction note |
|---------|--------|-----------------|
| `period_register` | light / medium / heavy | How strongly period language is felt |
| `worldbuilding_density` | sparse / integrated / expository / immersive | How setting is delivered |
| `cultural_specificity` | generic / culturally_marked / period_specific / highly_granular | Tie to culture/era |
| `anachronism_risk` | none / low / flagged_items | Modern vocabulary that shouldn't appear |
| `register_shifts` | yes / no + description | Social class / power markers in language |

#### Suggested `to_include_features` — Book Chapter (full set)

```yaml
to_include_features:
  # Prose style
  - "Sentence Rhythm"
  - "Paragraph Rhythm"
  - "White Space Strategy"
  - "Information Position in Sentence"
  # POV
  - "Narrative Perspective"
  - "POV Distance"
  - "Interior Monologue Density"
  - "Free Indirect Discourse"
  - "POV Unreliability"
  # Dialogue
  - "Dialogue Ratio"
  - "Dialogue Style"
  - "Subtext Ratio"
  - "Voice Differentiation"
  # Sensory
  - "Dominant Sensory Modality"
  - "Sensory Distribution"
  - "Color Register"
  - "Spatial Register"
  - "Light Register"
  # Figurative
  - "Figurative Language Density"
  - "Figurative Language Types"
  - "Comparison Style"
  - "Sound Devices Used"
  # Pacing
  - "Pacing Profile"
  - "Tension Arc"
  - "Scene Transition Style"
  - "Chapter Entry Velocity"
  - "Chapter Exit Momentum"
  # Thematic / emotional
  - "Thematic Undertones"
  - "Theme Embodiment Method"
  - "Emotional Arc Shape"
  - "Tonal Register"
  - "Dramatic Irony Level"
  # Motif
  - "Chapter Motif"
  - "Motif Transformation"
  - "Callback Density"
  - "Foreshadowing Method"
  # Setting
  - "Period Register"
  - "Worldbuilding Density"
  - "Cultural Specificity"
```

#### Minimum viable set for initial training (18 features)

Start here if generating a small dataset first. These cover the highest-signal dimensions — the ones that most strongly determine what the prose sounds and feels like.

```yaml
to_include_features:
  - "Sentence Rhythm"
  - "Paragraph Rhythm"
  - "Narrative Perspective"
  - "POV Distance"
  - "Interior Monologue Density"
  - "Dialogue Ratio"
  - "Subtext Ratio"
  - "Dominant Sensory Modality"
  - "Color Register"
  - "Figurative Language Density"
  - "Figurative Language Types"
  - "Comparison Style"
  - "Pacing Profile"
  - "Tonal Register"
  - "Theme Embodiment Method"
  - "Emotional Arc Shape"
  - "Chapter Motif"
  - "Period Register"
```

---

### Theater Scene Features

Theater scenes share the base beat fields. These labels complete the scene card for the stage format.

#### Theatrical Convention and Structure

| Feature | Values | Extraction note |
|---------|--------|-----------------|
| `theatrical_convention` | direct_address / aside / soliloquy / tableau / dumb_show / chorus / naturalistic | Special devices used |
| `verse_prose_ratio` | all_prose / prose_dominant / mixed / verse_dominant / all_verse | Dialogue in verse vs. prose |
| `line_length_distribution` | stichomythia / long_speeches / mixed / couplets | Speech length patterns |
| `act_position` | opening / rising / crisis / climax / falling / close | Structural position |
| `scene_length_estimate` | short (<10m) / medium (10–20m) / long (>20m) | Playing time |

#### Physical and Spatial (Stage)

| Feature | Values | Extraction note |
|---------|--------|-----------------|
| `proxemic_pattern` | close_throughout / open_throughout / approach_retreat / static_then_collapse / circling | Character movement relative to each other |
| `blocking_complexity` | minimal_static / simple_crosses / moderate / choreographed | Moves and their significance |
| `stage_picture_count` | integer | Distinct tableau moments |
| `physical_subtext_density` | low / medium / high | Moves that externalize internal state |
| `elevation_use` | flat_stage / height_differential / level_shifts | Power shown through height |
| `object_handling` | none / incidental / symbolic / central | Props and their dramatic function |

#### Dialogue and Silence (Stage)

| Feature | Values | Extraction note |
|---------|--------|-----------------|
| `silence_beat_count` | integer | Strategic pauses/silences |
| `silence_placement` | mid_exchange / end_of_speech / entry / exit | Where silences fall |
| `subtext_to_text_ratio` | low / medium / high / total | Surface vs. real conversation |
| `interruption_pattern` | none / frequent / structured | How characters cut each other off |
| `overlap_or_relay` | yes / no | Characters finishing each other's thoughts |
| `speech_rhythm` | even_paced / staccato / building / decelerating | Tempo of dialogue |

#### Technical Theater

| Feature | Values | Extraction note |
|---------|--------|-----------------|
| `lighting_register` | naturalistic / high_contrast / warm_cool_shift / expressionistic / symbolic | Quality and function |
| `lighting_cue_density` | none / sparse / moderate / frequent | Cues per scene |
| `sound_design_role` | absent / atmospheric_only / punctuating / continuous_underscore | Function of sound |
| `set_complexity` | bare_stage / representational / naturalistic / stylized | What the stage looks like |
| `audience_information_state` | audience_ahead / audience_even / audience_behind | Dramatic irony level |

#### Suggested `to_include_features` — Theater Scene

```yaml
to_include_features:
  - "Theatrical Convention"
  - "Verse Prose Ratio"
  - "Proxemic Pattern"
  - "Stage Picture Count"
  - "Physical Subtext Density"
  - "Silence Beat Count"
  - "Silence Placement"
  - "Subtext to Text Ratio"
  - "Lighting Register"
  - "Sound Design Role"
  - "Audience Information State"
  - "Tonal Register"
  - "Pacing Profile"
  - "Blocking Complexity"
  - "Object Handling"
  - "Emotional Arc Shape"
  - "Voice Differentiation"
```

---

### Film / Screen Scene Features

Film scenes share the base beat fields and the existing `camera_beats` / `sound_design` blocks. These labels add the multi-dimensional layer on top.

#### Visual Grammar

| Feature | Values | Extraction note |
|---------|--------|-----------------|
| `visual_grammar_style` | classical_hollywood / handheld_vérité / formalist / impressionist / surveillance / genre_specific | Overall visual language |
| `shot_size_distribution` | % ECU / CU / MCU / MS / WS / EWS across scene | Count per size |
| `movement_profile` | static_dominant / fluid / agitated / unmotivated / mixed | Camera movement character |
| `depth_of_field_style` | shallow_intimate / deep_contextual / variable_tracking | What is in/out of focus and why |
| `visual_subtext_density` | low / medium / high | Images carrying meaning without dialogue |

#### Editing Rhythm

| Feature | Values | Extraction note |
|---------|--------|-----------------|
| `cut_frequency` | fast (<3s) / medium (3–8s) / slow (>8s) / mixed | Average shot length |
| `editing_style` | continuity / parallel / rhythmic_montage / associative / elliptical | Cut logic |
| `reaction_shot_density` | low / medium / high | How often we cut to reactions |
| `time_manipulation` | real_time / compressed / expanded / non_linear / flash_cut | Temporal treatment |
| `match_cut_use` | none / occasional / structural | Thematic or graphic cuts |
| `chapter_in_out_method` | cold_open / establishing_wide / close_in / match_from_prior / smash_cut | Entry/exit method |

#### Sound Design (Film — extends existing `sound_design` block)

| Feature | Values | Extraction note |
|---------|--------|-----------------|
| `score_genre` | none / orchestral / electronic / diegetic_source / silence_as_score | Type of score |
| `dialogue_mix_style` | clean_present / overlapping / off_screen_dominant / mumbled_naturalistic | How dialogue is delivered |
| `sound_motif` | free text | Recurring sound element with symbolic function |
| `silence_function` | absent / punctuation / horror / reverence / isolation / grief | What silence does |

#### Performance and Color

| Feature | Values | Extraction note |
|---------|--------|-----------------|
| `performance_register` | restrained_micro / naturalistic / heightened / stylized | Acting style implied |
| `silent_performance_moments` | list of non-dialogue acting beats | Non-verbal scenes carrying narrative |
| `color_temperature` | warm / cool / neutral / shift_warm_to_cool / shift_reverse | Dominant color temp |
| `color_grading_register` | saturated / desaturated / high_contrast / naturalistic / tinted | Grading style |
| `visual_metaphor` | free text | Recurring image with symbolic function |
| `light_source_type` | natural / practical / artificial / mixed / expressionistic | Lighting sourcing |
| `shadow_use` | absent / atmospheric / symbolic / expressionistic | Role of shadow |

#### Visual Storytelling

| Feature | Values | Extraction note |
|---------|--------|-----------------|
| `visual_storytelling_method` | image_tells_all / image_complements_dialogue / image_contradicts_dialogue / dialogue_dominant | Image-word relationship |
| `off_screen_space` | ignored / implied / active / dominant | What happens outside frame |
| `lens_choice_meaning` | intimacy / surveillance / isolation / grandeur / alienation / neutral | What the lens communicates |
| `subtext_through_mise_en_scene` | list of staging/set/prop choices carrying subtext | Environmental subtext |

#### Suggested `to_include_features` — Film Scene

```yaml
to_include_features:
  - "Visual Grammar Style"
  - "Shot Size Distribution"
  - "Movement Profile"
  - "Depth of Field Style"
  - "Visual Subtext Density"
  - "Cut Frequency"
  - "Editing Style"
  - "Reaction Shot Density"
  - "Time Manipulation"
  - "Score Genre"
  - "Silence Function"
  - "Performance Register"
  - "Silent Performance Moments"
  - "Color Temperature"
  - "Visual Metaphor"
  - "Light Source Type"
  - "Shadow Use"
  - "Visual Storytelling Method"
  - "Lens Choice Meaning"
  - "Off Screen Space"
  # Shared with other formats
  - "Tonal Register"
  - "Subtext to Text Ratio"
  - "Dramatic Irony Level"
  - "Pacing Profile"
  - "Emotional Arc Shape"
```

---

### Shared Features (All Formats)

These apply across book, theater, and film and should appear in every architecture regardless of format.

| Feature | Values | Notes |
|---------|--------|-------|
| `tonal_register` | tragic / comic / wry / elegiac / tense / tender / cold / mythic | Dominant tone |
| `dramatic_irony_level` | reader_ahead / reader_even / reader_behind | Information asymmetry |
| `pacing_profile` | per-scene: slow / medium / fast + rationale | Scene-by-scene rhythm |
| `emotional_arc_shape` | per-character: flat / ascending / descending / V / spike / oscillating | |
| `subtext_to_text_ratio` | low / medium / high / total | Surface vs. real event |
| `sensory_anchor_count` | integer per scene | Distinct anchors |
| `dominant_sensory_modality` | visual / auditory / tactile / olfactory / gustatory | Primary sense |
| `tonal_shifts` | list: moment + what tone shifts to | Tonal arc |
| `power_register` | ascending / descending / contested / stable | Power trajectory |
| `callback_density` | sparse / moderate / dense | Callbacks to prior material |

---

### Annotation Prompt Pattern

Use this prompt to extract architecture labels from a finished text:

```
You are annotating a finished [book chapter / theater scene / film scene] to create
a structured architecture for LLM training.

Read the following text, then fill every field in the architecture template.

Rules:
- Extract only what is present in the text. Do not invent.
- Use exact values from the allowed sets where provided.
- Keep free-text entries under 30 words.
- Mark ambiguous fields with: (ambiguous: [reason])
- Replace all character names with [PERSON].
- Replace all real location names with [CITY] or [REGION].

[architecture template]

[source text]
```

The annotated architecture becomes the **input**. The original text (anonymized) becomes the **output**. Together they form one training pair.
