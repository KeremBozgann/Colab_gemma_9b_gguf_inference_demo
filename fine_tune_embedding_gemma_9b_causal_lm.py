
# !pip install -U transformers accelerate peft bitsandbytes datasets

import os
from dataclasses import dataclass
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType


@dataclass
class Config:
    output_dir: str = "output_dolly_qlora"
    checkpoint: str = "ytu-ce-cosmos/Turkish-Gemma-9b-v0.1"
    max_length: int = 512                 # Dolly answers are usually not huge; 512 trains faster
    seed: int = 42

    # training
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4  # global batch = 8
    per_device_eval_batch_size: int = 4
    n_epochs: int = 1
    lr: float = 2e-4
    warmup_steps: int = 50
    weight_decay: float = 0.0
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200
    optim_type: str = "adamw_8bit"

    # LoRA
    freeze_layers: int = 16               # don't adapt the first 16 layers
    num_layers: int = 42                  # you used 42; keep consistent
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"

    # data
    eval_size: float = 0.02               # 2% eval split


config = Config()
torch.manual_seed(config.seed)
np.random.seed(config.seed)


# -------------------------
# 1) Tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(config.checkpoint, use_fast=True)

# For decoder-only models, pad_token is often missing; set it to eos.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# -------------------------
# 2) 4-bit quantized base model (Causal LM)
# -------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,  # bfloat16 if supported
)

model = AutoModelForCausalLM.from_pretrained(
    config.checkpoint,
    device_map="auto",
    quantization_config=bnb_config,
)

model.config.use_cache = False
model = prepare_model_for_kbit_training(model)


# -------------------------
# 3) LoRA config for CAUSAL_LM
# -------------------------
lora_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # include o_proj often helps
    layers_to_transform=[i for i in range(config.num_layers) if i >= config.freeze_layers],
    lora_dropout=config.lora_dropout,
    bias=config.lora_bias,
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# -------------------------
# 4) Load Dolly dataset
# -------------------------
# Columns: instruction, context, response, category :contentReference[oaicite:1]{index=1}
ds = load_dataset("databricks/databricks-dolly-15k", split="train")

# quick train/eval split
ds = ds.shuffle(seed=config.seed)
splits = ds.train_test_split(test_size=config.eval_size, seed=config.seed)
train_ds, eval_ds = splits["train"], splits["test"]


# -------------------------
# 5) Build prompt template
# -------------------------
def format_example(instruction: str, context: str) -> str:
    instruction = (instruction or "").strip()
    context = (context or "").strip()

    if context:
        # Dolly often expects "instruction + context"
        # Keep it simple and consistent:
        return f"User: {instruction}\nContext: {context}\nAssistant:"
    else:
        return f"User: {instruction}\nAssistant:"


# -------------------------
# 6) Tokenize + label masking (train only on assistant answer tokens)
# -------------------------
def tokenize_and_mask(batch):
    prompts = [format_example(i, c) for i, c in zip(batch["instruction"], batch["context"])]
    answers = [(r or "").strip() for r in batch["response"]]

    # Full text = prompt + answer + eos
    full_texts = [p + " " + a + tokenizer.eos_token for p, a in zip(prompts, answers)]

    tok = tokenizer(
        full_texts,
        max_length=config.max_length,
        truncation=True,
        padding=False,
    )

    labels = []
    for p, input_ids in zip(prompts, tok["input_ids"]):
        # Tokenize prompt alone (no special tokens) to find where answer begins
        p_ids = tokenizer(p, add_special_tokens=False)["input_ids"]
        lab = input_ids.copy()
        n = min(len(p_ids), len(lab))
        lab[:n] = [-100] * n  # mask prompt tokens
        labels.append(lab)

    tok["labels"] = labels
    return tok


remove_cols = train_ds.column_names
train_ds = train_ds.map(tokenize_and_mask, batched=True, remove_columns=remove_cols)
eval_ds  = eval_ds.map(tokenize_and_mask,  batched=True, remove_columns=remove_cols)



# -------------------------
# 7) Training args + Trainer
# -------------------------
training_args = TrainingArguments(
    output_dir=config.output_dir,
    report_to="none",
    num_train_epochs=config.n_epochs,
    per_device_train_batch_size=config.per_device_train_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    per_device_eval_batch_size=config.per_device_eval_batch_size,
    learning_rate=config.lr,
    warmup_steps=config.warmup_steps,
    weight_decay=config.weight_decay,
    logging_steps=config.logging_steps,
    eval_strategy="steps",
    eval_steps=config.eval_steps,
    save_strategy="steps",
    save_steps=config.save_steps,
    optim=config.optim_type,
    fp16=True,
    remove_unused_columns=False,  # IMPORTANT when providing custom labels
)

from dataclasses import dataclass
from typing import Any, Dict, List
import torch
from transformers import PreTrainedTokenizerBase

@dataclass
class DataCollatorForCausalLMWithPadding:
    tokenizer: PreTrainedTokenizerBase
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Separate labels so tokenizer.pad doesn't try to tensorize ragged lists
        labels = [f.pop("labels") for f in features] if "labels" in features[0] else None

        # Pad only model inputs
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        # Now pad labels to the same length as input_ids and attach
        if labels is not None:
            max_len = batch["input_ids"].shape[1]
            padded_labels = []
            for lbl in labels:
                if isinstance(lbl, torch.Tensor):
                    lbl = lbl.tolist()
                lbl = lbl[:max_len]  # safety truncate
                pad_len = max_len - len(lbl)
                padded_labels.append(lbl + [self.label_pad_token_id] * pad_len)

            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        return batch


data_collator = DataCollatorForCausalLMWithPadding(tokenizer=tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
)

trainer.train()

# Save adapter (LoRA)
trainer.save_model(config.output_dir)
tokenizer.save_pretrained(config.output_dir)
print("Saved LoRA adapter + tokenizer to:", config.output_dir)

