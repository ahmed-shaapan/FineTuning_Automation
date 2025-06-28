from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import argparse
import numpy as np
from peft import PeftModel, PeftConfig

def load_model(model_dir):
    try:
        config = PeftConfig.from_pretrained(model_dir)
        base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        return PeftModel.from_pretrained(base_model, model_dir)
    except:
        return AutoModelForCausalLM.from_pretrained(model_dir)

def compute_identity(pred, ref):
    matches = sum(p == r for p, r in zip(pred, ref))
    return matches / max(len(ref), 1)

def evaluate_model(model_dir, test_file):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = load_model(model_dir).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    dataset = load_dataset("text", data_files={"test": test_file})["test"]
    identities = []
    for example in dataset:
        input_ids = tokenizer(example["text"], return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_length=512)
        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        ref = example["text"].strip()
        identities.append(compute_identity(pred, ref))

    print(f"Avg Identity: {np.mean(identities):.4f}")
    avg_id = np.mean(identities) # Store the value
    print(f"Avg Identity: {avg_id:.4f}")
    return avg_id

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_dir", required=True)
#     parser.add_argument("--test_file", required=True)
#     args = parser.parse_args()
#     evaluate_model(args.model_dir, args.test_file)
