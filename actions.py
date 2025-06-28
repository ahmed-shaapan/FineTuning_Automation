# actions.py
import os
import tempfile
import json

# Import your own project files as modules
import model_loader
import preprocess
import optuna_search
import trainer
import generate
import evaluate
from datasets import load_dataset

# This is the path to the persistent storage on RunPod.
PERSISTENT_VOLUME_PATH = "/runpod_volume"

# --- MODIFIED FUNCTION ---
def run_full_pipeline(job):
    """
    This function runs the complete fine-tuning pipeline.
    """
    job_input = job['input']
    
    # --- 1. Get instructions from the job payload ---
    model_key = job_input['model_key']
    fasta_content = job_input['fasta_content']
    n_trials = job_input.get('n_trials', 5)
    # NEW: Get the fine-tuning mode from the API
    finetune_mode = job_input.get('finetune_mode', 'qlora') # 'qlora' or 'full'
    
    user_id = job_input.get('user_id', 'default_user')
    job_id = job.get('id', 'local_job')
    
    # Determine flags based on the mode
    use_quantization = (finetune_mode == 'qlora')
    use_lora = (finetune_mode == 'qlora')
    
    print(f"--- Starting Pipeline Job: {job_id} | Mode: {finetune_mode} ---")

    # --- 2. Define persistent output directory ---
    output_dir = os.path.join(PERSISTENT_VOLUME_PATH, "output", user_id, job_id)
    os.makedirs(output_dir, exist_ok=True)
    print(f"All artifacts will be saved to: {output_dir}")

    # --- 3. Write FASTA content to a temporary file ---
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fasta", encoding='utf-8') as tmp_fasta:
        tmp_fasta.write(fasta_content)
        fasta_filepath = tmp_fasta.name
    
    try:
        # --- 4. Execute your pipeline steps ---
        
        # Preprocessing is the same
        # ... (code from your file is good here) ...
        print("[INFO] Reading and preprocessing FASTA...")
        raw_sequences = preprocess.read_fasta(fasta_filepath)
        preprocessor_key = "progen" if "progen" in model_key else "rita" if "rita" in model_key else "protgpt2"
        processed_sequences = preprocess.PREPROCESSORS[preprocessor_key](raw_sequences)
        train, val, test = preprocess.split_dataset(processed_sequences)
        preprocess.save_splits(train, val, test, output_dir)
        dataset = load_dataset("text", data_files={
            "train": os.path.join(output_dir, "train.txt"),
            "validation": os.path.join(output_dir, "validation.txt"),
        })

        print("[INFO] Running Optuna hyperparameter search...")
        # CRITICAL: Optuna needs to know whether to load a quantized model or not
        best_trial = optuna_search.optuna_search(
            model_init=lambda: model_loader.load_model_and_tokenizer(model_key, use_quantization=use_quantization)[0],
            tokenizer=model_loader.load_model_and_tokenizer(model_key, use_quantization=False)[1], # Tokenizer is not quantized
            dataset=dataset,
            output_dir=os.path.join(output_dir, "optuna"),
            n_trials=n_trials,
            direction="maximize"
        )
        print(f"[INFO] Best hyperparameters found: {best_trial.params}")

        print("[INFO] Training final model...")
        # Reload a fresh model for final training with the correct quantization setting
        model, tokenizer = model_loader.load_model_and_tokenizer(model_key, use_quantization=use_quantization)
        final_model_dir = os.path.join(output_dir, "final_model")
        trainer.train_model(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            best_params=best_trial.params,
            model_name=model_key,
            output_dir=final_model_dir,
            use_lora=use_lora
        )

        # --- 5. Clean up and return the result ---
        os.remove(fasta_filepath)
        return {
            "message": "Full pipeline completed successfully.",
            "finetune_mode": finetune_mode,
            "final_model_path": final_model_dir, # This path is on the persistent volume
            "best_hyperparameters": best_trial.params
        }

    except Exception as e:
        os.remove(fasta_filepath)
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}

# --- MODIFIED FUNCTION ---
def run_generation(job):
    """
    Generates a sequence from either a fine-tuned model or a base model.
    """
    job_input = job['input']
    
    # Get instructions
    model_dir = job_input.get('model_dir_on_volume') # For fine-tuned models
    base_model_key = job_input.get('base_model_key')    # For base models
    prompt = job_input['prompt']
    max_new_tokens = job_input.get('max_new_tokens', 200)

    if not model_dir and not base_model_key:
        return {"error": "Must provide either 'model_dir_on_volume' or 'base_model_key'."}
    
    # Determine model key and directory to load from
    model_to_load_key = base_model_key
    model_to_load_dir = model_dir

    if base_model_key:
        # If using a base model, we use its key for generation logic
        print(f"--- Starting Generation Job from BASE model: {base_model_key} ---")
        model_to_load_key = base_model_key
        # Load from the pre-warmed cache
        model, tokenizer = model_loader.load_model_and_tokenizer(base_model_key, use_quantization=False) # No need to quantize for inference
    else:
        # If using a fine-tuned model, we infer its key from the config
        print(f"--- Starting Generation Job from Fine-Tuned model: {model_dir} ---")
        try:
            config_path = os.path.join(model_dir, "config.json")
            with open(config_path) as f:
                config_data = json.load(f)
            base_name = config_data.get('_name_or_path', '')
            # Find the key from the name (e.g., "nferruz/ProtGPT2" -> "protgpt2")
            model_to_load_key = next((key for key, val in model_loader.MODEL_MAP.items() if val == base_name), None)
            if not model_to_load_key: raise ValueError("Could not determine base model from config")
        except Exception as e:
            return {"error": f"Could not load model info from fine-tuned directory: {e}"}

        model, tokenizer = generate.load_model_and_tokenizer(model_dir)

    # Run your generation logic
    if "protgpt2" in model_to_load_key:
        generated_text = generate.generate_with_protgpt2(model, tokenizer, prompt, max_new_tokens)
    elif "progen" in model_to_load_key:
        generated_text = generate.generate_with_progen(model, tokenizer, prompt, max_new_tokens)
    elif "rita" in model_to_load_key:
        generated_text = generate.generate_with_rita(model, tokenizer, prompt, max_new_tokens)
    else:
        raise ValueError(f"Unsupported model type: {model_to_load_key}")
    
    return {
        "message": "Generation successful.",
        "prompt": prompt,
        "generated_sequence": generated_text
    }
    
# ... The `run_evaluation` function is mostly okay, but ensure paths are correct
def run_evaluation(job):
    job_input = job['input']
    model_dir = job_input['model_dir_on_volume']
    # The job folder is the parent of the final_model dir
    job_folder = os.path.dirname(model_dir) 
    test_file = os.path.join(job_folder, "test.txt") 

    if not os.path.exists(test_file):
        return {"error": f"Test file not found at expected path: {test_file}"}

    print(f"--- Starting Evaluation Job on model: {model_dir} ---")
    avg_identity = evaluate.evaluate_model(
        model_dir=model_dir,
        test_file=test_file
    )
    return {
        "message": "Evaluation successful.",
        "model_dir": model_dir,
        "average_identity": avg_identity
    }