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
# Any data saved here will remain after the job is finished.
PERSISTENT_VOLUME_PATH = "/runpod_volume"

def run_full_pipeline(job):
    """
    This function replaces your main.py. It runs the complete fine-tuning pipeline.
    """
    job_input = job['input']
    
    # --- 1. Get instructions from the job payload ---
    model_key = job_input['model_key']
    fasta_content = job_input['fasta_content']
    n_trials = job_input.get('n_trials', 5) # Use .get for optional params
    use_lora = job_input.get('use_lora', True)
    
    user_id = job_input.get('user_id', 'default_user')
    job_id = job.get('id', 'local_job')

    print(f"--- Starting Full Pipeline Job: {job_id} for User: {user_id} ---")

    # --- 2. Define persistent output directory ---
    # This is the most critical change: save everything to the persistent volume.
    # The path will be unique for each job, e.g., /runpod_volume/default_user/abc-123/
    output_dir = os.path.join(PERSISTENT_VOLUME_PATH, user_id, job_id)
    os.makedirs(output_dir, exist_ok=True)
    print(f"All artifacts will be saved to: {output_dir}")

    # --- 3. Write FASTA content to a temporary file ---
    # Your preprocess script expects a file path, so we create one from the string content.
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fasta", encoding='utf-8') as tmp_fasta:
        tmp_fasta.write(fasta_content)
        fasta_filepath = tmp_fasta.name
    
    try:
        # --- 4. Execute your pipeline steps, using variables ---
        print("[INFO] Loading base model and tokenizer...")
        model, tokenizer = model_loader.load_model_and_tokenizer(model_key)

        print("[INFO] Reading and preprocessing FASTA...")
        raw_sequences = preprocess.read_fasta(fasta_filepath)
        
        # Determine the correct preprocessor key
        preprocessor_key = "progen" if "progen" in model_key else "rita" if "rita" in model_key else "protgpt2"
        processed_sequences = preprocess.PREPROCESSORS[preprocessor_key](raw_sequences)

        print("[INFO] Splitting and saving dataset splits...")
        train, val, test = preprocess.split_dataset(processed_sequences)
        preprocess.save_splits(train, val, test, output_dir) # Save splits to the persistent job folder

        print("[INFO] Loading dataset from persistent storage...")
        dataset = load_dataset("text", data_files={
            "train": os.path.join(output_dir, "train.txt"),
            "validation": os.path.join(output_dir, "validation.txt"),
            "test": os.path.join(output_dir, "test.txt"),
        })

        print("[INFO] Running Optuna hyperparameter search...")
        # Note: model_init needs to be a function that can be called without arguments
        best_trial = optuna_search.optuna_search(
            model_init=lambda: model_loader.load_model_and_tokenizer(model_key)[0],
            tokenizer=tokenizer,
            dataset=dataset,
            output_dir=os.path.join(output_dir, "optuna"),
            n_trials=n_trials,
            direction="maximize"
        )
        print(f"[INFO] Best hyperparameters found: {best_trial.params}")

        print("[INFO] Training final model...")
        # Reload a fresh model for final training
        model, tokenizer = model_loader.load_model_and_tokenizer(model_key)
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
            "model_key": model_key,
            "final_model_path": final_model_dir, # This path is on the persistent volume
            "best_hyperparameters": best_trial.params
        }

    except Exception as e:
        # Ensure cleanup happens even on failure
        os.remove(fasta_filepath)
        print(f"[ERROR] Pipeline failed for job {job_id}: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def run_generation(job):
    """
    This function replaces your generate.py.
    """
    job_input = job['input']
    
    # --- 1. Get instructions ---
    model_key = job_input['model_key']
    # This path MUST point to a completed job folder on the persistent volume
    model_dir = job_input['model_dir_on_volume'] 
    prompt = job_input['prompt']
    max_new_tokens = job_input.get('max_new_tokens', 200)

    print(f"--- Starting Generation Job ---")
    print(f"Using model at: {model_dir}")
    
    # --- 2. Run your generation logic ---
    generated_text = generate.generate_sequence(
        model_name=model_key,
        model_dir=model_dir, # Pass the persistent path
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        save_path=None # We don't save to a file, we return the text
    )
    
    return {
        "message": "Generation successful.",
        "prompt": prompt,
        "generated_sequence": generated_text
    }

def run_evaluation(job):
    """
    This function replaces your evaluate.py.
    """
    job_input = job['input']
    
    # --- 1. Get instructions ---
    model_dir = job_input['model_dir_on_volume']
    # The test file was created during the pipeline and is in the same job folder
    test_file = os.path.join(model_dir, "../test.txt") 

    print(f"--- Starting Evaluation Job ---")
    print(f"Evaluating model at: {model_dir}")
    print(f"Using test file: {test_file}")

    # --- 2. Run your evaluation logic ---
    avg_identity = evaluate.evaluate_model(
        model_dir=model_dir,
        test_file=test_file
    )
    
    return {
        "message": "Evaluation successful.",
        "model_dir": model_dir,
        "average_identity": avg_identity
    }