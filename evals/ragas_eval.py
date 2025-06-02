import os
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness
from openai import OpenAI

# Load QA data
with open('evals/qa_data.json', 'r') as f:
    qa_data = json.load(f)

# Prepare data for RAGAS
# For RAGAS evaluation on faithfulness and answer_correctness,
# we need a dataset with at least 'question', 'answer', and 'contexts'.
# Since we don't have contexts in the provided qa_data.json,
# we'll create dummy contexts for demonstration purposes.
# In a real RAG scenario, you would extract contexts from your retrieval system.

# Assuming your model is accessible via an API or loaded locally
# Replace with your actual model loading/calling logic
def get_model_response(question):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content

def get_dummy_context(question):
    # In a real application, this would involve retrieving relevant documents
    # For this example, we'll just return a placeholder
    return ["This is a dummy context for the question: " + question]

data = []
for item in qa_data:
    question = item["question"]
    ground_truth = item["answer"]
    answer = get_model_response(question)
    contexts = get_dummy_context(question) # Replace with actual context retrieval
    data.append({
        "question": question,
        "answer": answer,
        "ground_truth": ground_truth,
        "contexts": contexts
    })

# Create a RAGAS Dataset
dataset = Dataset.from_list(data)

# Define the metrics to evaluate
metrics = [
    faithfulness,
    answer_correctness,
]

# Evaluate the dataset
results = evaluate(
    dataset,
    metrics=metrics,
)

# Print the evaluation results
print(results)

# You can also convert the results to a pandas DataFrame
results_df = results.to_pandas()
print(results_df)

# Optionally, save the results
results_df.to_csv("ragas_evaluation_results.csv", index=False)