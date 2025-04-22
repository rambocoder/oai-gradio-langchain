import random
import mlflow
import json
import os
from openai import OpenAI, RateLimitError, APIError
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()

# mlflow ui --host 0.0.0.0 --port 5000

# --- Configuration ---
QA_DATA_FILE = os.path.join(os.path.dirname(__file__), "qa_data.json")
MODELS_TO_EVALUATE = ["gpt-4o-mini", "gpt-3.5-turbo"]
MAX_RETRIES_JUDGE = 3
INITIAL_DELAY = 1
JUDGE_MODEL = "gpt-4o"  # Model used for judging answers
MLFLOW_EXPERIMENT_NAME = "OpenAI QA Accuracy Comparison"
# Optional: Define a system prompt for the chat models
SYSTEM_PROMPT = "You are a helpful assistant. Answer the following question concisely."
# Rate limiting / concurrency control
MAX_WORKERS = 5 # Adjust based on your OpenAI rate limits and system resources

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- OpenAI Client Setup ---
try:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client. Is OPENAI_API_KEY set? Error: {e}")
    exit(1)

# --- Helper Functions ---

def load_qa_data(filepath):
    """Loads question-answer pairs from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logging.info(f"Successfully loaded {len(data)} Q&A pairs from {filepath}")
        return data
    except FileNotFoundError:
        logging.error(f"Error: Data file not found at {filepath}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {filepath}")
        return None

def get_openai_answer(model_name, question, max_retries=3, initial_delay=1):
    """Gets an answer from the specified OpenAI model with retry logic."""
    attempt = 0
    delay = initial_delay
    while attempt < max_retries:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question}
                ],
                temperature=0.2, # Lower temperature for more factual answers
                max_tokens=100   # Adjust as needed
            )
            answer = response.choices[0].message.content.strip()
            return answer
        except RateLimitError as e:
            attempt += 1
            logging.warning(f"Rate limit hit for model {model_name} on question '{question[:30]}...'. Retrying in {delay}s... (Attempt {attempt}/{max_retries})")
            time.sleep(delay)
            delay *= 2 # Exponential backoff
        except APIError as e:
            attempt += 1
            logging.error(f"OpenAI API error for model {model_name} on question '{question[:30]}...': {e}. Retrying in {delay}s... (Attempt {attempt}/{max_retries})")
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            logging.error(f"An unexpected error occurred getting answer for model {model_name} on question '{question[:30]}...': {e}")
            return None # Return None on unexpected errors after retries fail
    logging.error(f"Failed to get answer for model {model_name} after {max_retries} retries for question '{question[:30]}...'.")
    return None

def compare_answers(question, generated_answer, ground_truth):
    """
    Compares the generated answer with the ground truth using GPT-4 as a judge.
    Returns True if the generated answer is deemed correct, False otherwise.
    """
    if generated_answer is None:
        return False
        
    judge_prompt = f"""You are an impartial judge evaluating answer quality.
Question: {question}

Compare these two answers and determine if they convey the same meaning:

Ground Truth: {ground_truth}
Generated Answer: {generated_answer}

Respond with only 'True' if they convey the same core meaning, or 'False' if they differ significantly."""

    try:
        judge_response = client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4 for better judgment
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.1  # Low temperature for consistent judgments
        )
        result = judge_response.choices[0].message.content.strip().lower()
        print(f"Judge question: {judge_prompt}")
        print(f"Judge response: {result}")
        return result == 'true'
    except Exception as e:
        logging.error(f"Error in LLM judgment: {e}")
        return False
    
def compare_answers_llm_as_judge(question, generated_answer, ground_truth, max_retries=MAX_RETRIES_JUDGE, initial_delay=INITIAL_DELAY):
    """
    Compares the generated answer with the ground truth using an LLM as a judge.
    Returns True if the judge deems the answer correct/acceptable, False otherwise.
    """
    if generated_answer is None or generated_answer.startswith("ERROR") or generated_answer.startswith("EXECUTION ERROR"):
        logging.warning(f"Judge: Generated answer is None or an error for Q: '{question[:30]}...'")
        return False # Cannot judge an error or missing answer

    # Simple check: if answers are identical (case-insensitive), they are correct. Saves an LLM call.
    if generated_answer.strip().lower() == ground_truth.strip().lower():
        logging.info(f"Judge: Exact match for Q: '{question[:30]}...'. Skipping LLM judge.")
        return True

    judge_system_prompt = """You are an impartial judge evaluating the correctness of an AI-generated answer based on a provided question and its ground truth answer.
Focus on factual accuracy and semantic equivalence. The generated answer does not need to be word-for-word identical to the ground truth, but it must convey the same core meaning and be factually correct according to the ground truth.
Ignore minor differences in phrasing, capitalization, or punctuation if the essential information is correct.
Respond with only one word: 'CORRECT' if the generated answer is acceptable, or 'INCORRECT' if it is not."""

    judge_user_prompt = f"""Please evaluate the following:
[Question]: {question}
[Ground Truth Answer]: {ground_truth}
[Generated Answer]: {generated_answer}

Is the [Generated Answer] factually correct and semantically equivalent to the [Ground Truth Answer] given the [Question]? Respond with only 'CORRECT' or 'INCORRECT'."""

    attempt = 0
    delay = initial_delay
    while attempt < max_retries:
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": judge_system_prompt},
                    {"role": "user", "content": judge_user_prompt}
                ],
                temperature=0.0, # Use 0 temperature for deterministic judgment
                max_tokens=10 # Expecting only 'CORRECT' or 'INCORRECT'
            )
            judge_response = response.choices[0].message.content.strip().upper()

            logging.info(f"Judge ({JUDGE_MODEL}) assessment for Q: '{question[:30]}...' -> {judge_response}")

            if judge_response == "CORRECT":
                return True
            elif judge_response == "INCORRECT":
                return False
            else:
                # Handle unexpected responses from the judge LLM
                attempt += 1
                logging.warning(f"Judge ({JUDGE_MODEL}) returned an unexpected response: '{judge_response}'. Retrying in {delay}s... (Attempt {attempt}/{max_retries})")
                time.sleep(delay + random.uniform(0, 1))
                delay *= 2

        except RateLimitError as e:
            attempt += 1
            logging.warning(f"Rate limit hit for Judge ({JUDGE_MODEL}) on question '{question[:30]}...'. Retrying in {delay}s... (Attempt {attempt}/{max_retries})")
            time.sleep(delay + random.uniform(0, 1))
            delay *= 2
        except APIError as e:
            attempt += 1
            logging.error(f"OpenAI API error for Judge ({JUDGE_MODEL}) on question '{question[:30]}...': {e}. Retrying in {delay}s... (Attempt {attempt}/{max_retries})")
            time.sleep(delay + random.uniform(0, 1))
            delay *= 2
        except Exception as e:
            logging.error(f"An unexpected error occurred during LLM-as-judge evaluation for question '{question[:30]}...': {e}")
            # Default to incorrect if judge fails unexpectedly after retries
            return False # Or handle this case differently if needed

    logging.error(f"Failed to get judgment from model {JUDGE_MODEL} after {max_retries} retries for question '{question[:30]}...'. Defaulting to INCORRECT.")
    return False
# --- Main Evaluation Logic ---

def evaluate_model(model_name, qa_data):
    """Evaluates a single model on the Q&A data."""
    correct_count = 0
    total_count = len(qa_data)
    results = [] # Store individual results

    # Use ThreadPoolExecutor for concurrent API calls
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a future for each question
        future_to_qa = {executor.submit(get_openai_answer, model_name, item['question']): item for item in qa_data}

        for future in as_completed(future_to_qa):
            item = future_to_qa[future]
            question = item['question']
            ground_truth = item['answer']
            try:
                generated_answer = future.result()
                is_correct = compare_answers_llm_as_judge(question, generated_answer, ground_truth)
                if is_correct:
                    correct_count += 1
                results.append({
                    "question": question,
                    "ground_truth": ground_truth,
                    "generated_answer": generated_answer if generated_answer else "ERROR/TIMEOUT",
                    "is_correct": is_correct
                })
                logging.info(f"Model: {model_name} | Q: '{question[:30]}...' | A: '{generated_answer[:30]} | Correct: {is_correct}")
            except Exception as exc:
                logging.error(f"Model: {model_name} | Q: '{question[:30]}...' generated an exception: {exc}")
                results.append({
                    "question": question,
                    "ground_truth": ground_truth,
                    "generated_answer": f"EXECUTION ERROR: {exc}",
                    "is_correct": False
                })

    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    logging.info(f"Evaluation finished for model: {model_name}. Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")
    return accuracy, results

# --- MLflow Experiment ---

if __name__ == "__main__":
    qa_data = load_qa_data(QA_DATA_FILE)
    if not qa_data:
        exit(1)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    for model_name in MODELS_TO_EVALUATE:
        logging.info(f"--- Starting evaluation for model: {model_name} ---")
        with mlflow.start_run(run_name=f"eval_{model_name}") as run:
            # Log parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("num_questions", len(qa_data))
            mlflow.log_param("system_prompt", SYSTEM_PROMPT)

            # Perform evaluation
            start_time = time.time()
            accuracy, results = evaluate_model(model_name, qa_data)
            end_time = time.time()
            duration = end_time - start_time

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("evaluation_duration_seconds", duration)
            mlflow.log_metric("correct_answers", int(accuracy/100 * len(qa_data)))
            mlflow.log_metric("total_questions", len(qa_data))


            # Log results table as artifact (optional but very useful)
            results_filename = f"{model_name}_evaluation_results.json"
            with open(results_filename, 'w') as f:
                json.dump(results, f, indent=2)
            mlflow.log_artifact(results_filename)
            os.remove(results_filename) # Clean up local file after logging

            # Log the input data as an artifact
            mlflow.log_artifact(QA_DATA_FILE)

            logging.info(f"--- Finished evaluation for model: {model_name}. Run ID: {run.info.run_id} ---")

    logging.info("All evaluations complete. Run 'mlflow ui' in your terminal to view results.")