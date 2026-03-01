
# !pip install transformers
# !pip install mamba-ssm[causal-conv1d]
# !pip install -U bitsandbytes accelerate peft transformers

import os
import copy
from dataclasses import dataclass

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    BitsAndBytesConfig,
    Gemma2ForSequenceClassification,
    GemmaTokenizerFast,
    Gemma2Config,
    PreTrainedTokenizerBase, 
    EvalPrediction,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from sklearn.metrics import log_loss, accuracy_score


@dataclass
class Config:
    output_dir: str = "output"
    checkpoint: str = "ytu-ce-cosmos/Turkish-Gemma-9b-v0.1"  # 4-bit quantized gemma-2-9b-instruct
    max_length: int = 1024
    n_splits: int = 5
    fold_idx: int = 0
    optim_type: str = "adamw_8bit"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 2  # global batch size is 8 
    per_device_eval_batch_size: int = 8
    n_epochs: int = 1
    freeze_layers: int = 16  # there're 42 layers in total, we don't add adapters to the first 16 layers
    lr: float = 2e-4
    warmup_steps: int = 20
    lora_r: int = 16
    lora_alpha: float = lora_r * 2
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    
config = Config()





lora_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    # only target self-attention
    target_modules=["q_proj", "k_proj", "v_proj"],
    layers_to_transform=[i for i in range(42) if i >= config.freeze_layers],
    lora_dropout=config.lora_dropout,
    bias=config.lora_bias,
    task_type=TaskType.SEQ_CLS,
)



tokenizer = GemmaTokenizerFast.from_pretrained(config.checkpoint)
tokenizer.add_eos_token = True  # We'll add <eos> at the end
tokenizer.padding_side = "right"






from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,  # or bfloat16 if your GPU supports it
)

model = Gemma2ForSequenceClassification.from_pretrained(
    config.checkpoint,
    num_labels=3,
    device_map="auto",
    quantization_config=bnb_config,
)

model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)





import bitsandbytes as bnb

quant_modules = (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)

found = []
for name, module in model.named_modules():
    if isinstance(module, quant_modules):
        found.append((name, type(module)))

print("Found quantized modules:", len(found))
print(found[:5])  # print a few





model.print_trainable_parameters()

import torch
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM allocated:", torch.cuda.memory_allocated()/1e9, "GB")
print("VRAM reserved :", torch.cuda.memory_reserved()/1e9, "GB")

import torch
print(torch.cuda.memory_summary())







import csv
import pandas as pd

path = "train.csv"  # use original, not your replaced one

rows = []
bad = []

with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
    reader = csv.reader(
        f,
        delimiter=",",
        quotechar='"',
        escapechar="\\",
        doublequote=True,
    )
    header = next(reader)
    n = len(header)

    for i, row in enumerate(reader, start=2):  # line numbers (header is 1)
        if len(row) != n:
            bad.append((i, len(row)))
            continue
        rows.append(row)

print("expected cols:", n)
print("bad rows (first 20):", bad[:20])

df = pd.DataFrame(rows, columns=header)






from datasets import Dataset
ds = Dataset.from_pandas(df, preserve_index=False)
ds = ds.select(list(range(10000)))





import numpy as np

def rebuild_labels(batch):
    # force numeric 0/1
    a = np.array(batch["winner_model_a"]).astype(np.int64)
    b = np.array(batch["winner_model_b"]).astype(np.int64)
    t = np.array(batch["winner_tie"]).astype(np.int64)

    mat = np.stack([a, b, t], axis=1)  # (B,3)
    labels = mat.argmax(axis=1).astype(np.int64)

    # sanity: how many rows are valid one-hot?
    sums = mat.sum(axis=1)
    # (optional) if sums are not 1, your source data is inconsistent
    # print("one-hot sums counts:", {k:int(v) for k,v in zip(*np.unique(sums, return_counts=True))})

    return {"labels": labels.tolist()}

ds = ds.map(rebuild_labels, batched=True)




# What columns exist?
print(ds.column_names)







import numpy as np
print("overall:", np.bincount(np.array(ds["labels"]), minlength=3))






import json
import math
class CustomTokenizer:
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerBase, 
        max_length: int
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, batch: dict) -> dict:
        prompt = ["<prompt>: " + self.process_text(t) for t in batch["prompt"]]
        response_a = ["\n\n<response_a>: " + self.process_text(t) for t in batch["response_a"]]
        response_b = ["\n\n<response_b>: " + self.process_text(t) for t in batch["response_b"]]
        texts = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]

        tokenized = self.tokenizer(texts, max_length=self.max_length, truncation=True)

        # IMPORTANT: keep existing labels; don't overwrite
        return {**tokenized, "labels": batch["labels"]}

        
    @staticmethod
    def process_text(cell) -> str:
        if cell is None:
            return ""
        if isinstance(cell, float) and math.isnan(cell):
            return ""
        if isinstance(cell, list):
            return " ".join(str(x) for x in cell if x is not None)

        s = str(cell).strip()
        if not s:
            return ""

        # This handles: ["..."] with lots of \n, \/, etc.
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return " ".join(str(x) for x in obj if x is not None)
            return str(obj)
        except Exception:
            return s  # fallback
        




encode = CustomTokenizer(tokenizer, max_length=config.max_length)
ds = ds.map(encode, batched=True)



def compute_metrics(eval_preds: EvalPrediction) -> dict:
    preds = eval_preds.predictions
    labels = eval_preds.label_ids
    probs = torch.from_numpy(preds).float().softmax(-1).numpy()
    loss = log_loss(y_true=labels, y_pred=probs)
    acc = accuracy_score(y_true=labels, y_pred=preds.argmax(-1))
    return {"acc": acc, "log_loss": loss}



folds = [
    (
        [i for i in range(len(ds)) if i % config.n_splits != fold_idx],
        [i for i in range(len(ds)) if i % config.n_splits == fold_idx]
    ) 
    for fold_idx in range(config.n_splits)
]




train_idx, eval_idx = folds[config.fold_idx]

trainer = Trainer(
    args=training_args, 
    model=model,
    train_dataset=ds.select(train_idx),
    eval_dataset=ds.select(eval_idx),
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)
trainer.train()





