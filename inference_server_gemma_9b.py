"""
LLM inference server for Gemma 9B with optional LoRA adapter.
Run: uvicorn inference_server_gemma_9b:app --host 0.0.0.0 --port 8000
"""

from typing import Optional
import os
import json

import torch
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# === Settings ===
MODEL_ID = "ytu-ce-cosmos/Turkish-Gemma-9b-v0.1"  # HF repo id (set this to a quantized repo to save disk)
MODEL_IS_PREQUANTIZED = True  # True for pre-quantized repos (e.g. AWQ/GPTQ/BNB-4bit)
LOAD_IN_4BIT = False
LOAD_IN_8BIT = False
PREQUANT_BITS = 4  # used only if pre-quantized checkpoint has invalid/missing quantization_config

LOCAL_MODEL_DIR = "/content/drive/MyDrive/models/turkish-gemma-9b-4bit"
USE_LOCAL_MODEL = True
DOWNLOAD_IF_MISSING = False  # set True to download into LOCAL_MODEL_DIR

ADAPTER_DIR = "/content/drive/MyDrive/Colab Notebooks/checkpoint-1600"  # LoRA adapter
USE_ADAPTER = False

SYSTEM_MESSAGE = "Kısa ve net cevaplar ver."
API_KEY = "alfabeta11!"  # set your own key before exposing
API_KEY_HEADER = "x-api-key"

MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True
NUM_BEAMS = 1

CORS_ALLOW_ORIGINS = ["*"]  # tighten this in production
CORS_ALLOW_METHODS = ["*"]
CORS_ALLOW_HEADERS = ["*"]




class GenerateRequest(BaseModel):
    user_message: str
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    do_sample: Optional[bool] = None
    num_beams: Optional[int] = None


class GenerateResponse(BaseModel):
    text: str


def build_bnb_config(load_in_4bit: bool, load_in_8bit: bool):
    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    if load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def resolve_checkpoint() -> str:
    if USE_LOCAL_MODEL:
        if os.path.isdir(LOCAL_MODEL_DIR):
            return LOCAL_MODEL_DIR
        if DOWNLOAD_IF_MISSING:
            from huggingface_hub import snapshot_download

            snapshot_download(
                MODEL_ID,
                local_dir=LOCAL_MODEL_DIR,
                local_dir_use_symlinks=False,
            )
            return LOCAL_MODEL_DIR
        raise FileNotFoundError(
            f"Local model dir not found: {LOCAL_MODEL_DIR}. "
            "Set DOWNLOAD_IF_MISSING=True or update LOCAL_MODEL_DIR."
        )
    return MODEL_ID


def get_prequant_config_dict(bits: int) -> dict:
    if bits == 4:
        return {
            "quant_method": "bitsandbytes",
            "load_in_4bit": True,
            "load_in_8bit": False,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "float16",
        }
    return {
        "quant_method": "bitsandbytes",
        "load_in_4bit": False,
        "load_in_8bit": True,
    }


def ensure_valid_prequant_config(checkpoint: str) -> None:
    """
    Some saved checkpoints may contain `quantization_config: null` in config.json.
    Newer Transformers then crashes while loading. Fix it in-place.
    """
    if not MODEL_IS_PREQUANTIZED:
        return
    if not os.path.isdir(checkpoint):
        return

    config_path = os.path.join(checkpoint, "config.json")
    if not os.path.isfile(config_path):
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    if config_data.get("quantization_config", "__missing__") is not None:
        return

    config_data["quantization_config"] = get_prequant_config_dict(PREQUANT_BITS)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)


def load_model_and_tokenizer():
    if LOAD_IN_4BIT and LOAD_IN_8BIT:
        raise ValueError("Choose only one of LOAD_IN_4BIT or LOAD_IN_8BIT")
    if MODEL_IS_PREQUANTIZED and (LOAD_IN_4BIT or LOAD_IN_8BIT):
        raise ValueError(
            "MODEL_IS_PREQUANTIZED=True requires LOAD_IN_4BIT=False and LOAD_IN_8BIT=False."
        )

    bnb_config = None if MODEL_IS_PREQUANTIZED else build_bnb_config(LOAD_IN_4BIT, LOAD_IN_8BIT)
    checkpoint = resolve_checkpoint()
    ensure_valid_prequant_config(checkpoint)

    model_config = None
    if MODEL_IS_PREQUANTIZED:
        model_config = AutoConfig.from_pretrained(checkpoint)
        if getattr(model_config, "quantization_config", None) is None:
            # In-memory fallback for environments where config.json is stale or cached.
            model_config.quantization_config = get_prequant_config_dict(PREQUANT_BITS)

    if USE_ADAPTER:
        try:
            tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, use_fast=True)
        except OSError:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_load_kwargs = {
        "device_map": "auto",
        "dtype": torch.float16,
    }
    if bnb_config is not None:
        model_load_kwargs["quantization_config"] = bnb_config
    if model_config is not None:
        model_load_kwargs["config"] = model_config

    base_model = AutoModelForCausalLM.from_pretrained(checkpoint, **model_load_kwargs)

    if USE_ADAPTER:
        model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    else:
        model = base_model

    model.eval()

    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Tokenizer has no chat_template; use a chat-tuned checkpoint.")

    return model, tokenizer


app = FastAPI(title="Gemma 9B Inference Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=False,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
)
model, tokenizer = load_model_and_tokenizer()


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": req.user_message},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    max_new_tokens = req.max_new_tokens or MAX_NEW_TOKENS
    temperature = req.temperature if req.temperature is not None else TEMPERATURE
    top_p = req.top_p if req.top_p is not None else TOP_P
    do_sample = req.do_sample if req.do_sample is not None else DO_SAMPLE
    num_beams = req.num_beams if req.num_beams is not None else NUM_BEAMS

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][prompt_len:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not text:
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return GenerateResponse(text=text)
