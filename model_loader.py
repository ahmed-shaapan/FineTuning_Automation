# model_loader.py

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

MODEL_MAP = {
    "protgpt2": "nferruz/ProtGPT2",
    "progen2-small": "hugohrban/progen2-small",
    "progen2-medium": "hugohrban/progen2-medium",
    "progen2-large": "hugohrban/progen2-large",
    "progen2-xlarge": "hugohrban/progen2-xlarge",
    "RITA_s": "lightonai/RITA_s",
    "RITA_m": "lightonai/RITA_m",
    "RITA_l": "lightonai/RITA_l", 
    "RITA_xl": "lightonai/RITA_xl",
}

# --- MODIFIED FUNCTION ---
def load_model_and_tokenizer(model_key: str, use_quantization: bool = False):
    """
    Loads the model and tokenizer.
    If use_quantization is True, loads the model in 4-bit for QLoRA.
    Otherwise, loads in full or half precision.
    """
    if model_key not in MODEL_MAP:
        raise ValueError(f"Unknown model: {model_key}. Supported: {list(MODEL_MAP.keys())}")

    model_name = MODEL_MAP[model_key]
    print(f"[INFO] Loading model: {model_key} from {model_name} (Quantized: {use_quantization})")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if use_quantization:
        # Configuration for 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map={"": 0} # Place on the first available GPU
        )
    else:
        # Load for full fine-tuning
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto", # Automatically shard across GPUs if needed
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )

    return model, tokenizer