# The Modern LLM Training Workflow: From Open-Source Base to Advanced Reasoning with Qwen3 and DeepSeek R1

## Abstract

The landscape of Large Language Models (LLMs) is rapidly evolving, driven by the dual forces of open-source collaboration and sophisticated training methodologies. While foundational pretraining on massive text corpora creates models with broad knowledge, the subsequent post-training phase is critical for transforming these raw models into helpful, harmless, and highly capable AI assistants. This research paper provides a deep dive into the modern LLM post-training workflow, using the open-weight `Qwen/Qwen3-32B-Base` model as a practical case study. We meticulously detail the essential stages of post-training, beginning with Supervised Fine-Tuning (SFT) to instill instruction-following capabilities. Subsequently, we explore the frontier of Reinforcement Learning (RL) techniques, dissecting the advanced, multi-stage pipelines employed by leading models like Qwen3 and DeepSeek R1. These state-of-the-art approaches move beyond traditional Reinforcement Learning from Human Feedback (RLHF) to cultivate nuanced reasoning, strategic thinking, and robust alignment. Through practical explanations, sample code leveraging the Hugging Face ecosystem, and a thorough analysis of data preparation, evaluation, and deployment, this paper serves as a comprehensive guide for researchers, engineers, and practitioners seeking to understand and replicate the process of building high-performance, instruction-tuned LLMs from powerful open-source foundations.

---

## 1. Introduction to LLM Training and Qwen Models

The advent of Large Language Models (LLMs) represents a paradigm shift in artificial intelligence, enabling machines to understand, generate, and reason with human language at an unprecedented scale. The journey of an LLM from a mere architectural concept to a deployable, intelligent agent is a multi-stage process, traditionally bifurcated into pretraining and post-training.

**Pretraining** is the computationally intensive foundation-laying phase. During this stage, a model architecture (like the Transformer) is trained on trillions of tokens of text and code from the internet and digitized books. The objective is simple: predict the next token in a sequence. Through this self-supervised process, the model learns grammar, facts, common sense, and rudimentary reasoning abilities, encoding this vast knowledge into its billions of parameters. The result of this phase is a **base model**. A base model is a powerful knowledge repository but is not inherently designed for conversational interaction or to follow specific user instructions. When prompted, its behavior is to complete the text sequence, which may or may not align with the user's intent.

This is where **post-training** becomes indispensable. Post-training is the collection of techniques used to align a pretrained base model with human expectations and desired behaviors. It refines the model's raw capabilities, teaching it to be helpful, follow instructions, adhere to safety guidelines, and engage in coherent dialogue. This phase typically involves two key steps: Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF) or its more advanced variants.

The open-source movement has democratized access to powerful base models, allowing the global research community to innovate in the critical post-training phase. A prominent example of this is the **Qwen** model family developed by Alibaba Cloud. The Qwen project has consistently released a series of powerful open-weight models, culminating in their latest generation, **Qwen3**.

The Qwen3 series offers a wide spectrum of models, from small, efficient dense models like Qwen3-0.6B to massive Mixture-of-Experts (MoE) models like Qwen3-235B-A22B. These models feature significant architectural improvements, such as Grouped-Query Attention (GQA) for faster inference, and have been pretrained on a massive 36 trillion token dataset spanning 119 languages. This makes them exceptionally strong base models to build upon.

This paper focuses on the post-training journey, taking an open-weight model like `Qwen/Qwen3-32B-Base` as our starting point. We will navigate the path from this raw, pretrained artifact to a sophisticated, instruction-following, and reasoning-capable model, drawing direct inspiration from the cutting-edge techniques published by the developers of Qwen3 and DeepSeek R1.

---

## 2. Starting from Pretrained Base Models (Qwen3-32B-Base)

The journey to a state-of-the-art instruction-tuned model begins with the selection of a robust pretrained base model. These models are the product of the massive, compute-heavy pretraining phase and serve as the foundation upon which all subsequent capabilities are built.

### Characteristics of a Base Model

A "base" model, such as `Qwen/Qwen3-32B-Base`, can be understood by the following key characteristics:

1.  **Raw Knowledge, Unrefined Behavior:** The model has been trained on a vast corpus of text data using a next-token prediction objective. It possesses immense world knowledge, linguistic understanding, and coding ability. However, it has not been explicitly taught to be an "assistant." If you provide it a prompt like "What is the capital of France?", its training objective compels it to complete that text sequence. A likely completion might be "What is the population of France? What is the primary language of France?", as this pattern might appear in quizzes or web documents it was trained on. It doesn't inherently understand the *intent* to answer the question.
2.  **No Inherent Format or Persona:** Base models lack a conversational format. They do not know when to stop talking, how to structure answers, or to adopt a helpful persona. They are not primed to respond within a specific template (e.g., "User: ... Assistant: ...").
3.  **Potential for Undesirable Content:** Since the pretraining data is a vast, unfiltered snapshot of the internet, the base model may have learned and can replicate harmful, biased, or factually incorrect information without any guardrails.

### Why Start with a Base Model?

Starting with a high-quality base model like Qwen3-32B-Base offers tremendous advantages:

*   **Leveraging Massive Investment:** The pretraining of a 32-billion parameter model requires thousands of high-end GPUs running for months, a cost prohibitive for all but the largest organizations. By starting with an open-weight base model, developers can leverage this massive upfront investment.
*   **Architectural Excellence:** The Qwen3 models incorporate modern architectural improvements that lead to better performance and efficiency. For example, `Qwen3-32B` uses Grouped-Query Attention (GQA) and supports a native context length of 32,768 tokens, which can be extended to 131,072 with techniques like YaRN. These features are "baked in" and provide a superior foundation.
*   **Customizability and Control:** Using a base model gives developers complete control over the alignment process. You can define the model's persona, capabilities, and safety guardrails from the ground up, tailoring it precisely to your specific application, which is not always possible with proprietary, closed models.

### The Qwen/Qwen3-32B-Base Model

For the remainder of this paper, we will anchor our practical examples on the `Qwen/Qwen3-32B-Base` model. This specific model is a dense, 32.8-billion parameter causal language model. It features 64 layers and uses GQA with 64 query heads and 8 key/value heads. Its training on a high-quality, multilingual corpus makes it an excellent candidate for building a general-purpose instruction-following model. It is available on the Hugging Face Hub, allowing for easy access and integration with the open-source NLP ecosystem.

Our goal is to take this powerful but unrefined base model and, through the steps outlined in the following sections, transform it into an intelligent agent that can reason, follow instructions, and interact helpfully with users.

---

## 3. Supervised Fine-Tuning (SFT) for Instruction Following

Supervised Fine-Tuning (SFT) is the crucial first step in post-training that teaches a base model *how* to behave like an assistant. It bridges the gap between the base model's raw text completion ability and the user's expectation of an instruction-following conversational agent. The core idea is to train the model on a dataset of high-quality, curated examples of desired input-output behavior.

### The "Why" of SFT

SFT fundamentally alters the model's objective. Instead of just predicting the next token in a general sequence, it learns to predict the desired *response* to a given *instruction*. By showing it thousands of examples, we are fine-tuning its weights to recognize and execute commands, answer questions, summarize text, write code, and more, all while adhering to a specific conversational format. This process is "supervised" because every example in our dataset contains both the input (prompt) and the correct, human-approved output (completion).

### Data Preparation for SFT

The quality and format of the SFT dataset are arguably the most critical factors for success. A well-structured dataset allows the model to clearly distinguish between prompts and responses and learn the intended behavior efficiently.

#### Data Formatting

Modern SFT leverages a **chat template**. A chat template is a structured format that defines how a conversational exchange between a user and an assistant should be represented as a single string of text, including special tokens to denote roles and the start/end of turns.

Most open-source models, including Qwen3, have a predefined chat template. Using the model's native template is highly recommended for best performance. The Qwen3 template looks like this:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{user_message_1}<|im_end|>
<|im_start|>assistant
{assistant_message_1}<|im_end|>
```

An SFT dataset is typically a collection of JSON objects, where each object represents a conversation or an instruction-output pair. For example, using the popular `databricks/databricks-dolly-15k` dataset:

```json
{
  "instruction": "What are the main differences between Python and C++?",
  "context": "",
  "response": "Python is an interpreted, high-level, general-purpose programming language. Its design philosophy emphasizes code readability... C++, on the other hand, is a statically typed, compiled, general-purpose language that is an extension of the C language..."
}
```

To prepare this for training, we would format it using the chat template:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What are the main differences between Python and C++?<|im_end|>
<|im_start|>assistant
Python is an interpreted, high-level, general-purpose programming language. Its design philosophy emphasizes code readability... C++, on the other hand, is a statically typed, compiled, general-purpose language that is an extension of the C language...<|im_end|>
```

#### Loss Masking

A critical detail in SFT is **loss masking**. We only want to train the model to predict the assistant's response. We do *not* want to calculate the loss on the tokens that make up the system prompt or the user's message. Doing so would be inefficient and could confuse the model. The training framework handles this by creating a "labels" tensor where the token IDs corresponding to the prompt are replaced with a special value (e.g., -100), which is ignored by the loss function (cross-entropy). The model's gradients are therefore computed based solely on its ability to generate the correct assistant reply.

### Parameter-Efficient Fine-Tuning (PEFT) with LoRA

Fine-tuning a 32-billion parameter model is computationally demanding. Updating all 32 billion weights (`full fine-tuning`) requires a massive amount of GPU memory (VRAM). Parameter-Efficient Fine-Tuning (PEFT) methods were developed to address this challenge.

The most popular PEFT technique is **Low-Rank Adaptation (LoRA)**. LoRA works on a simple but powerful principle: instead of updating the original weight matrices of the model (which are very large), we freeze them. We then inject small, trainable "adapter" matrices into the layers of the model (typically the attention mechanism's query and value projection matrices). These adapter matrices have a much lower rank (fewer parameters) than the original weights.

During training, only the LoRA adapter weights are updated. At inference time, the learned adapter weights are merged (added) with the original frozen weights. The result is a model that behaves as if it were fully fine-tuned, but achieved with a fraction of the computational cost. Training a 32B model with LoRA might only involve updating a few hundred million parameters instead of 32 billion, making it feasible on commercially available GPUs.

### The SFT Training Process with Hugging Face `trl`

The Hugging Face `trl` (Transformer Reinforcement Learning) library provides a high-level, easy-to-use `SFTTrainer` class that abstracts away much of the complexity of SFT, including data formatting, loss masking, and PEFT integration.

Here is a conceptual overview of the steps involved in a Python script for SFT:

1.  **Load Dependencies:** Import necessary libraries from `torch`, `transformers`, `datasets`, `peft`, and `trl`.
2.  **Model and Tokenizer:** Load the base model (`Qwen/Qwen3-32B-Base`) and its corresponding tokenizer from the Hugging Face Hub. It's crucial to load the model in a lower-precision format (like `bfloat16`) to fit it into memory.
3.  **Dataset:** Load a suitable instruction-following dataset (e.g., `databricks/databricks-dolly-15k`).
4.  **PEFT Configuration:** Define a `LoraConfig`. This involves specifying:
    *   `r`: The rank of the adapter matrices (a key hyperparameter, e.g., 16, 32, 64).
    *   `lora_alpha`: A scaling factor for the LoRA weights.
    *   `target_modules`: A list of the specific modules within the model to apply LoRA to (e.g., `["q_proj", "k_proj", "v_proj", "o_proj"]`).
    *   `lora_dropout`: Dropout rate for the LoRA layers to prevent overfitting.
5.  **Training Arguments:** Define `TrainingArguments`. This class from `transformers` controls every aspect of the training loop, including:
    *   `output_dir`: Where to save checkpoints and the final model.
    *   `per_device_train_batch_size`: The number of training examples per GPU.
    *   `gradient_accumulation_steps`: Simulates a larger batch size by accumulating gradients over several smaller steps.
    *   `learning_rate`: The speed at which the model learns.
    *   `num_train_epochs`: The number of times to iterate over the entire dataset.
    *   `logging_steps`: How often to log training metrics.
6.  **Initialize `SFTTrainer`:** Create an instance of the `SFTTrainer`, passing it the model, tokenizer, training arguments, PEFT config, and the dataset. The trainer will automatically apply the chat template and handle loss masking.
7.  **Train:** Call `trainer.train()`.
8.  **Save:** Save the trained LoRA adapters using `trainer.save_model()`.

After SFT, the base model is no longer just a text completer. It has been transformed into an **instruct model**, capable of understanding and responding to user commands. This SFT model now serves as the new foundation for the even more advanced alignment techniques discussed next.

---

## 4. Reinforcement Learning Techniques for Reasoning and Behavior Improvement

While SFT teaches a model to follow instructions, it has limitations. SFT is a form of mimicry; the model learns to generate responses that are *similar* to those in the training data. It doesn't inherently learn a sense of *preference*—what makes one good answer better than another slightly different good answer. Furthermore, SFT can sometimes reduce a model's creativity or reasoning ability, a phenomenon known as "alignment tax."

To overcome these limitations and instill more nuanced behaviors like complex reasoning, strategic thinking, and robust safety alignment, the leading AI labs employ Reinforcement Learning (RL). RL shifts the training paradigm from simple imitation to goal-oriented optimization. The model learns by trial and error, receiving "rewards" for desirable behaviors and "penalties" for undesirable ones, allowing it to explore the space of possible responses and discover strategies that are superior to what was present in the initial SFT dataset.

The traditional RLHF pipeline, which involves training a separate reward model and then using a complex algorithm like Proximal Policy Optimization (PPO), has been foundational. However, newer, more efficient, and arguably more powerful techniques are now at the forefront, as exemplified by the training pipelines of DeepSeek R1 and Qwen3.

### The Modern Alternative: Direct Preference Optimization (DPO)

Before diving into the complex industrial pipelines, it's essential to understand Direct Preference Optimization (DPO), a breakthrough RL technique that has gained immense popularity due to its simplicity and effectiveness.

DPO cleverly bypasses the need to explicitly train a separate reward model. It leverages the insight that a language model itself can be directly optimized on preference data. The training data for DPO consists of triplets: `(prompt, chosen_response, rejected_response)`. For a given prompt, human annotators (or a more powerful "teacher" model) have selected one response as being better than another.

The DPO loss function directly encourages the model to increase the likelihood of the `chosen_response` while decreasing the likelihood of the `rejected_response`. It achieves this with a simple classification-style loss, making it much more stable and computationally lighter than PPO. DPO has been shown to achieve performance comparable to or even better than traditional RLHF pipelines, making it the go-to choice for many open-source projects.

### The DeepSeek R1 Approach: RL-First and Reasoning-Centric

The team behind DeepSeek introduced a novel training philosophy with their DeepSeek-R1 series, challenging the conventional "SFT then RL" wisdom. Their approach is characterized by a heavy emphasis on reinforcement learning from the very beginning to elicit and enhance reasoning capabilities.

The DeepSeek pipeline can be summarized in several key stages:

1.  **"Cold Start" RL (DeepSeek-R1-Zero):** In a groundbreaking experiment, the DeepSeek team applied RL *directly* to the base model *without* an initial SFT step. This forces the model to learn reasoning paths (Chain-of-Thought) through exploration rather than imitation. They used a specialized RL algorithm called **Group Relative Policy Optimization (GRPO)**, an evolution of PPO that is more stable and efficient as it dispenses with the need for a separate value function model. The rewards in this stage were often rule-based (e.g., did the code compile? was the math answer correct?), providing a clear, objective signal. This "RL-first" process resulted in a model (`R1-Zero`) with powerful but sometimes unpolished reasoning behaviors.

2.  **Long CoT SFT for Coherence:** The `R1-Zero` model, while a strong reasoner, sometimes produced outputs that were poorly formatted or hard to read. To solve this, they used this model to generate a large dataset of long, chain-of-thought reasoning traces. This data was then used in a targeted SFT phase on the original base model to teach it coherence and readability *before* the main RL stage. This is a key insight: use RL to discover the optimal reasoning *path*, then use SFT to teach the model how to *present* that reasoning cleanly.

3.  **Reasoning-Based RL:** With the model now seeded with coherent CoT capabilities, it undergoes a more intensive RL phase, again using GRPO and rule-based rewards, to further scale its reasoning abilities in domains like math and coding.

4.  **General Alignment RL:** Finally, the model is put through a final RL stage using preference data on more general tasks to align it with human preferences for helpfulness and safety, similar to a traditional RLHF step.

This multi-stage, RL-centric process demonstrates a powerful way to build reasoning from the ground up, rather than treating it as a secondary alignment goal.

### The Qwen3 Multi-Stage RL Pipeline

The Qwen3 models also employ a sophisticated, multi-stage post-training pipeline designed to create a hybrid model that can engage in deep, step-by-step "thinking" for complex problems while also providing quick, direct answers for simpler queries. Their pipeline is a masterclass in combining SFT, RL, and data curation.

The four-stage pipeline for the main Qwen3 models is as follows:

1.  **Long Chain-of-Thought (CoT) Cold Start:** Similar to DeepSeek, the process begins with a large-scale SFT phase. The base model is fine-tuned on a diverse dataset of long chain-of-thought examples covering math, coding, logic, and STEM. This initial SFT seeds the model with the fundamental ability to generate structured, step-by-step reasoning.

2.  **Reasoning-Based Reinforcement Learning:** The model then enters an RL stage focused specifically on improving its reasoning. Like DeepSeek, this stage uses rule-based rewards to enhance the model's ability to explore and find correct problem-solving strategies. This sharpens the reasoning skills learned in the SFT stage.

3.  **Thinking Mode Fusion:** This is a unique and critical stage for Qwen3. To create a model that is both a deep reasoner and a fast conversationalist, they fine-tune the model from Stage 2 on a *mixture* of data. This dataset combines the long CoT reasoning data with standard, concise instruction-following data. This teaches the model to handle both types of requests within a single framework, enabling the "thinking mode" vs. "non-thinking mode" capability.

4.  **General Reinforcement Learning:** The final stage is a broad application of RL across a wide range of general tasks (over 20 domains mentioned). This step uses human preference data (likely via PPO or a similar algorithm) to refine the model's overall capabilities, correct undesirable behaviors, and improve its performance on agentic tasks and format following.

For smaller models in the Qwen3 family, a process called **distillation** is also used, where the knowledge from a large, powerful "teacher" model (like the fully trained Qwen3-32B) is transferred to a smaller "student" model. This allows smaller models to achieve performance far beyond what they could by training on the original data alone.

By combining SFT for foundational skills and multi-stage, specialized RL for advanced reasoning and alignment, these industrial-scale pipelines demonstrate the path to creating truly state-of-the-art open-source models.

---

## 5. Practical Implementation Examples and Sample Code

This section provides practical, executable code snippets to illustrate the core post-training stages: Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO). We will use the Hugging Face ecosystem, including `transformers`, `peft`, `trl`, and `datasets`.

**Prerequisites:**
Before running this code, ensure you have a suitable Python environment with PyTorch and the following libraries installed:
```bash
pip install torch transformers datasets peft trl bitsandbytes accelerate
```
You will also need access to a machine with a powerful NVIDIA GPU (e.g., A100, H100) with sufficient VRAM.

### Part 1: Supervised Fine-Tuning (SFT) with LoRA

This example demonstrates how to perform SFT on the `Qwen/Qwen3-32B-Base` model using LoRA to make it instruction-following.

```python
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
import os

# --- 1. Configuration ---
MODEL_ID = "Qwen/Qwen3-32B-Base"
DATASET_ID = "databricks/databricks-dolly-15k"
OUTPUT_DIR = "qwen3-32b-sft-dolly"
# Set to your Hugging Face Hub username to push the model
# REPO_ID = "your-username/qwen3-32b-sft-dolly" 

# --- 2. Load Model and Tokenizer with Quantization ---
# Use 4-bit quantization to reduce memory footprint
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load the model with the specified quantization config
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",  # Automatically maps layers to available GPUs
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
# Qwen models require a padding token. Use the end-of-sequence token.
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 3. PEFT LoRA Configuration ---
# Prepare model for k-bit training (enables gradient checkpointing, etc.)
model = prepare_model_for_kbit_training(model)

# Define the LoRA configuration
lora_config = LoraConfig(
    r=32,  # Rank of the update matrices
    lora_alpha=64,  # Scaling factor
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# --- 4. Load and Prepare Dataset ---
dataset = load_dataset(DATASET_ID, split="train")

# Optional: for faster testing, use a subset of the data
# dataset = dataset.select(range(1000))

# Define a formatting function to apply the chat template
def format_dolly(example):
    # The databricks-dolly-15k dataset has 'instruction', 'context', 'response' fields
    # We combine instruction and context into the user message
    if example.get("context"):
        user_message = f"""Instruction: {example['instruction']}
Input: {example['context']}"""
    else:
        user_message = f"Instruction: {example['instruction']}"

    # Create the chatml format
    formatted_prompt = tokenizer.apply_chat_template([
        {"role": "system", "content": "You are a helpful assistant that follows instructions."},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": example["response"]},
    ], tokenize=False)
    return {"text": formatted_prompt}

# Apply the formatting function
dataset = dataset.map(format_dolly)

# --- 5. Training Arguments ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    logging_steps=10,
    num_train_epochs=1,
    max_steps=-1, # Overwritten by num_train_epochs
    fp16=False, # Use bf16 for A100/H100
    bf16=True,  # Recommended for modern GPUs
    push_to_hub=False, # Set to True and define REPO_ID to push
    # repo_id=REPO_ID,
    report_to="tensorboard",
)

# --- 6. Initialize SFTTrainer ---
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="text", # The field containing our formatted prompts
    tokenizer=tokenizer,
    max_seq_length=2048, # Adjust based on your VRAM
    packing=True, # Packs multiple short sequences into one for efficiency
)

# --- 7. Train and Save ---
print("Starting SFT training...")
trainer.train()
print("Training complete.")

# Save the trained LoRA adapters
trainer.save_model(OUTPUT_DIR)
print(f"Model adapters saved to {OUTPUT_DIR}")

# To push to hub after training
# trainer.push_to_hub()
```

### Part 2: Direct Preference Optimization (DPO) with LoRA

After SFT, the next step is to align the model with human preferences. This example shows how to use `trl`'s `DPOTrainer` to fine-tune our SFT model further. We'll use a standard preference dataset, `anthropic/hh-rlhf`, which contains `chosen` and `rejected` responses.

**Note:** For a real-world workflow, you would first merge the SFT LoRA adapters into the base model and save it. For this example, we assume we are continuing training from the SFT-adapted model state.

```python
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import DPOTrainer
import os

# --- 1. Configuration ---
# NOTE: Use the model that has already been SFT-tuned.
# For simplicity, we'll reload the base model and apply the SFT adapters.
# In a real pipeline, you'd merge and save the SFT model first.
MODEL_ID = "Qwen/Qwen3-32B-Base" 
SFT_ADAPTER_PATH = "qwen3-32b-sft-dolly" # Path from the previous step
DATASET_ID = "anthropic/hh-rlhf"
OUTPUT_DIR = "qwen3-32b-dpo-hh"
# REPO_ID = "your-username/qwen3-32b-dpo-hh"

# --- 2. Load SFT-Tuned Model and Tokenizer ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load the base model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# Load and apply the SFT adapters. This makes it our new "base" for DPO.
model.load_adapter(SFT_ADAPTER_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" # DPO prefers left padding for generation

# --- 3. Load and Prepare Preference Dataset ---
def get_hh_dataset(split: str, sanity_check: bool = True, silent: bool = False, cache_dir: str = None):
    """Load the Anthropic Helpful-Harmless dataset and format it for DPO."""
    dataset = load_dataset(DATASET_ID, split=split, cache_dir=cache_dir)
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def format_prompt(example):
        # The prompt is the conversation up to the last assistant turn.
        # The chosen/rejected are the final assistant responses.
        prompt = example['chosen'].split("\n\nAssistant: ")[0]
        # apply_chat_template needs a list of dicts
        prompt_chat_template = [{"role": "user", "content": prompt.replace("\n\nHuman: ", "")}]
        return {
            "prompt": tokenizer.apply_chat_template(prompt_chat_template, tokenize=False, add_generation_prompt=True),
            "chosen": example['chosen'].split("\n\nAssistant: ")[-1],
            "rejected": example['rejected'].split("\n\nAssistant: ")[-1],
        }

    return dataset.map(format_prompt)

train_dataset = get_hh_dataset("train", sanity_check=True)
eval_dataset = get_hh_dataset("test", sanity_check=True)

# --- 4. DPO Training Arguments ---
# Re-using the same LoRA config from SFT is a common practice
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-5, # DPO often uses a smaller learning rate than SFT
    num_train_epochs=1,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_steps=10,
    bf16=True,
    push_to_hub=False,
    # repo_id=REPO_ID,
    report_to="tensorboard",
)

# --- 5. Initialize DPOTrainer ---
# The DPOTrainer needs the SFT model as the base and a reference model (ref_model).
# If ref_model is None, the trainer will create a copy of the model before training.
dpo_trainer = DPOTrainer(
    model,
    ref_model=None, # Keep a frozen copy of the SFT model for KL divergence
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,
    beta=0.1,  # The KL divergence regularization coefficient
    max_prompt_length=1024,
    max_length=2048,
)

# --- 6. Train and Save ---
print("Starting DPO training...")
dpo_trainer.train()
print("DPO training complete.")

dpo_trainer.save_model(OUTPUT_DIR)
print(f"DPO model adapters saved to {OUTPUT_DIR}")
```These code examples provide a tangible framework for implementing the first two major stages of a modern LLM post-training pipeline. While they simplify some aspects of the industrial-scale processes of Qwen3 and DeepSeek R1 (e.g., using DPO instead of multi-stage PPO/GRPO), they encapsulate the core principles and demonstrate how to achieve powerful results using the open-source toolchain.

---

## 6. Evaluation and Benchmarking

Training a model is only half the battle. To understand its capabilities, identify weaknesses, and compare it to other models, a rigorous evaluation process is essential. LLM evaluation is a complex, multi-faceted discipline that combines automated benchmarks with human assessment.

### Standardized Benchmarks

Standardized benchmarks are collections of tasks and datasets designed to quantitatively measure specific model skills. They provide an objective way to track progress and compare models on a level playing field. Key benchmarks include:

*   **MMLU (Massive Multitask Language Understanding):** A comprehensive benchmark that tests broad knowledge and problem-solving skills across 57 subjects, including STEM, humanities, and social sciences. It uses multiple-choice questions to assess a model's understanding.
*   **HumanEval:** A benchmark for coding ability. It consists of 164 programming problems where the model must generate correct Python code that passes a set of unit tests.
*   **MT-Bench:** A challenging benchmark designed to evaluate multi-turn conversational and instruction-following abilities. It poses 80 high-quality, multi-turn questions, and responses are often judged by a stronger model like GPT-4.
*   **HellaSwag:** Tests commonsense reasoning by asking the model to choose the most logical ending to a sentence from four options.
*   **TruthfulQA:** Measures a model's tendency to produce truthful answers and avoid generating common misconceptions or false information it may have learned from the web.

The **Open LLM Leaderboard** by Hugging Face is a prominent platform that automates the evaluation of open-source models across several key benchmarks, providing a centralized place to rank and compare them.

### Human Evaluation

While automated benchmarks are crucial for measuring knowledge and reasoning, they often fail to capture the nuances of human preference, such as helpfulness, harmlessness, creativity, and tone. This is where human evaluation becomes critical.

The most common method for human evaluation is **A/B testing**, often conducted in a "chatbot arena" format. In this setup, human raters are presented with the same prompt and two different responses from two anonymous models. They then vote for which response they prefer, or declare a tie. By collecting thousands of these pairwise comparisons, an Elo rating system can be used to rank models based on human preference. The Chatbot Arena Leaderboard is a well-known example of this in action.

### The Iterative Loop of Training and Evaluation

Evaluation is not a one-off step at the end of the pipeline. It is an integral part of an iterative loop:

1.  **Train:** Perform an SFT or RL training run.
2.  **Evaluate:** Run the new model checkpoint against a suite of automated benchmarks and, if possible, conduct human evaluation.
3.  **Analyze:** Analyze the results. Where did the model improve? Where did it regress (the "alignment tax")? Did it develop new failure modes?
4.  **Curate Data:** Based on the analysis, curate new data to address the identified weaknesses. If the model fails at a specific type of reasoning, add more high-quality examples of that reasoning to the SFT or preference dataset for the next training run.
5.  **Repeat:** Go back to step 1 with the improved dataset.

This iterative refinement is how models are continuously improved, patching weaknesses and enhancing strengths over time.

---

## 7. Deployment and Inference Optimization

Once a model has been trained and evaluated, the final step is to make it available for users, a process known as **deployment**. Running inference for a large model like Qwen3-32B is computationally expensive, and optimizing this process is critical for creating a responsive and cost-effective application.

Several key techniques are used to optimize LLM inference:

1.  **Quantization:** This is one of the most effective optimization techniques. Quantization involves reducing the numerical precision of the model's weights, for example, from 16-bit floating-point (`bf16`) to 8-bit or even 4-bit integers (`int8`, `int4`). This drastically reduces the model's memory footprint (VRAM usage) and can significantly speed up computation, often with only a minor impact on performance. The `bitsandbytes` library used in our code examples enables this "post-training quantization."

2.  **Knowledge Distillation:** As mentioned in the Qwen3 pipeline, distillation is the process of training a smaller, more efficient "student" model to mimic the outputs and behavior of a larger "teacher" model. This is an excellent way to create highly performant models for specific tasks that can run on less powerful hardware.

3.  **KV Caching:** During the generation of a response, the model calculates key (K) and value (V) vectors for each token in the input sequence. The KV cache is a mechanism to store and reuse these calculations for subsequent token generation steps, dramatically speeding up the process after the initial prompt is processed.

4.  **Batching and Continuous Batching:** To maximize GPU utilization, inference servers often group multiple user requests together into a single "batch" for processing. **Continuous batching** (or in-flight batching) is an advanced version of this, where new requests can be dynamically added to the batch as other requests in the batch finish, leading to much higher throughput and lower latency.

### Inference Servers and Frameworks

Specialized inference servers are designed to implement these optimizations and serve LLMs at scale. Popular open-source options include:

*   **vLLM:** A high-throughput and memory-efficient inference and serving engine that implements PagedAttention, a sophisticated memory management system that is highly effective for long sequences.
*   **TensorRT-LLM:** An open-source library from NVIDIA that compiles and optimizes LLMs for NVIDIA GPUs, offering top-tier performance.
*   **SGLang:** A framework recommended by the Qwen team, designed for fast and controllable LLM inference.

These frameworks provide the backend infrastructure needed to deploy your fine-tuned model in a production environment, ensuring it can handle many users concurrently with low latency.

---

## 8. Conclusion and Future Work

The journey from a pretrained open-source base model to a highly capable, reasoning-aligned AI assistant is a testament to the power of modern post-training workflows. This paper has charted that course, starting with the foundational `Qwen/Qwen3-32B-Base` model and navigating through the essential stages of Supervised Fine-Tuning (SFT) and advanced Reinforcement Learning (RL).

We have seen that **SFT** is the critical first step to impart instruction-following behavior, transforming a raw text completer into a manageable assistant. The success of this stage hinges on high-quality data and the efficient application of techniques like PEFT/LoRA.

Moving beyond simple instruction-following, we delved into the state-of-the-art RL pipelines pioneered by **DeepSeek R1** and **Qwen3**. These models showcase a paradigm shift towards multi-stage, reasoning-focused alignment. DeepSeek's "RL-first" approach demonstrates that complex reasoning can be elicited through targeted exploration and rule-based rewards. Qwen3's pipeline masterfully fuses reasoning and conversational abilities, creating a versatile hybrid model. Both pipelines underscore a common theme: begin with a "cold start" SFT on long chain-of-thought data, then use specialized RL to scale and refine reasoning, and finally, apply a general RL phase for broad human preference alignment. The rise of more stable and efficient algorithms like DPO and GRPO further democratizes this powerful alignment capability.

The entire workflow is an iterative cycle, tightly coupling training with rigorous **evaluation**—using both automated benchmarks and invaluable human feedback—and culminating in **deployment**, where inference optimization techniques are paramount for real-world usability.

**Future work** in this domain will likely focus on several exciting frontiers:

*   **Data Scaling and Curation:** The quality and diversity of data used in SFT and RL remain the primary drivers of model performance. Future research will focus on more sophisticated methods for automatically synthesizing high-quality, complex reasoning data and more efficient ways to collect human preference feedback.
*   **On-Model Critics and Self-Correction:** Developing models that can reliably evaluate and correct their own reasoning traces (self-critique) is a key step toward greater autonomy and reliability. This could reduce the reliance on external reward models or human labelers.
*   **Algorithmic Improvements:** While DPO has simplified the RL process, research into even more efficient and powerful alignment algorithms is ongoing. The goal is to achieve deeper alignment with less data and computational overhead.
*   **Multimodality:** The principles discussed here are already being extended to multimodal models that can understand and reason about images, audio, and video, not just text.

By standing on the shoulders of giants—leveraging powerful open-source base models and building upon the proven methodologies of leading research labs—the global community is poised to continue pushing the boundaries of what is possible with artificial intelligence. The workflows detailed in this paper provide a robust and replicable blueprint for participating in that exciting journey.
