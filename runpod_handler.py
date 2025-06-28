# runpod_handler.py
import runpod
# Import the functions we just created
from actions import run_full_pipeline, run_generation, run_evaluation

def handler(job):
    """
    The main entry point for all RunPod serverless jobs.
    It checks the job_type and delegates to the appropriate function.
    """
    job_input = job['input']
    job_type = job_input.get('job_type')

    print(f"Received job of type: {job_type}")

    if job_type == 'full_pipeline':
        return run_full_pipeline(job)
    elif job_type == 'generate':
        return run_generation(job)
    elif job_type == 'evaluate':
        return run_evaluation(job)
    else:
        return {"error": f"Invalid job_type: '{job_type}'. Must be 'full_pipeline', 'generate', or 'evaluate'."}

# Standard RunPod boilerplate to start the worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})