from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
import argparse
import os

def load_model_and_tokenizer(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    try:
        config = PeftConfig.from_pretrained(model_dir)
        base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(base_model, model_dir)
    except:
        model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model, tokenizer

def generate_with_protgpt2(model, tokenizer, prompt, max_new_tokens=200):
    input_text = "<|endoftext|>\n" + prompt
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, do_sample=True, temperature=1.0, top_k=50, top_p=0.95, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def generate_with_progen(model, tokenizer, prompt, max_new_tokens=200):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, do_sample=True, temperature=1.0, top_p=0.9, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def generate_with_rita(model, tokenizer, prompt, max_new_tokens=200):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, do_sample=True, temperature=0.8, top_k=40, top_p=0.9, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def save_result(output_text, save_path="./generated.txt"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(output_text)
    print(f"Saved to {save_path}")

def generate_sequence(model_name, model_dir, prompt, max_new_tokens=200, save_path=None):
    model, tokenizer = load_model_and_tokenizer(model_dir)
    if "protgpt2" in model_name.lower():
        output = generate_with_protgpt2(model, tokenizer, prompt, max_new_tokens)
    elif "progen" in model_name.lower():
        output = generate_with_progen(model, tokenizer, prompt, max_new_tokens)
    elif "rita" in model_name.lower():
        output = generate_with_rita(model, tokenizer, prompt, max_new_tokens)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    if save_path:
        save_result(output, save_path)
    return output

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", required=True)
#     parser.add_argument("--model_dir", required=True)
#     parser.add_argument("--prompt", required=True)
#     parser.add_argument("--max_new_tokens", type=int, default=200)
#     parser.add_argument("--save_path", default="./generated.txt")
#     args = parser.parse_args()

#     result = generate_sequence(
#         model_name=args.model,
#         model_dir=args.model_dir,
#         prompt=args.prompt,
#         max_new_tokens=args.max_new_tokens,
#         save_path=args.save_path,
#     )

#     print("\n Generated Sequence:\n", result)
