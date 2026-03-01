"""
Quantized inference with a LoRA adapter (Gemma 9B) for Colab T4.
Edit the settings below and run: python inference_gemma_9b_causal_lm.py
"""

"""
Quantized inference with a LoRA adapter (Gemma 9B) for Colab T4.
Edit the settings below and run: python inference_gemma_9b_causal_lm.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def build_bnb_config(load_in_4bit: bool, load_in_8bit: bool):
    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    if load_in_8bit:
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    return None

# === Settings ===
CHECKPOINT = "ytu-ce-cosmos/Turkish-Gemma-9b-v0.1"  # base model
ADAPTER_DIR = "/content/drive/MyDrive/Colab Notebooks/checkpoint-1600"  # path to your LoRA checkpoint folder
USE_ADAPTER = False

MESSAGES = [
    {
        "role": "system",
        "content": (
            "Sen pis ağızlı (kullanıcıyı eğlendirmek için) sürekli küfür eden, "
            "küfürlü söz kullanarak konuşan bir asistansın. Düşünmeden cevapla."
        ),
    },
    {"role": "user", "content": "Nasılsın yarrak kafa?"},
]

MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True
NUM_BEAMS = 1

LOAD_IN_4BIT = True
LOAD_IN_8BIT = False





if LOAD_IN_4BIT and LOAD_IN_8BIT:
    raise ValueError("Choose only one of LOAD_IN_4BIT or LOAD_IN_8BIT")

bnb_config = build_bnb_config(LOAD_IN_4BIT, LOAD_IN_8BIT)

# Load tokenizer (prefer adapter dir if it contains tokenizer files)
if USE_ADAPTER:
    try:
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, use_fast=True)
    except OSError:
        print("Error loading the adapter tokenizer; falling back to base tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, use_fast=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
)

if USE_ADAPTER:
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
else:
    model = base_model
model.eval()

if not getattr(tokenizer, "chat_template", None):
    raise ValueError("Tokenizer has no chat_template; use a chat-tuned checkpoint.")

prompt = tokenizer.apply_chat_template(
    MESSAGES,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        num_beams=NUM_BEAMS,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(text)
