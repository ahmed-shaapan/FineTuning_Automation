# trainer.py

from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
import torch

def get_target_modules(model_name):
    if "progen" in model_name:
        return ["q_proj", "v_proj"]
    elif "rita" in model_name:
        # This might need to be "query_key_value" for some RITA versions
        return ["Wq", "Wv", "query_key_value"]
    elif "protgpt2" in model_name:
        return ["c_attn"]
    return ["q_proj", "v_proj"]

def apply_lora(model, model_name, task_type=TaskType.CAUSAL_LM):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=get_target_modules(model_name),
        lora_dropout=0.05,
        bias="none",
        task_type=task_type,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

# --- MODIFIED FUNCTION ---
def train_model(model, tokenizer, dataset, best_params, model_name, output_dir="./trained_model", use_lora=True):
    """
    Trains the model. If use_lora is True, applies LoRA adapters and trains them.
    Otherwise, performs full fine-tuning.
    """
    if use_lora:
        model = apply_lora(model, model_name)
        # Use QLoRA-optimized learning rate
        learning_rate = 2e-4
    else:
        # Use a more standard learning rate for full fine-tuning
        learning_rate = best_params.get("learning_rate", 2e-5)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        learning_rate=learning_rate,
        per_device_train_batch_size=best_params["per_device_train_batch_size"],
        num_train_epochs=best_params["num_train_epochs"],
        weight_decay=best_params["weight_decay"],
        fp16=False, # Use BF16 for modern GPUs
        bf16=torch.cuda.is_available(),
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="none",
        # Gradient checkpointing saves VRAM, essential for full fine-tuning
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
    )

    trainer.train()
    # Save model saves adapters if LoRA, full model otherwise
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved at {output_dir}")