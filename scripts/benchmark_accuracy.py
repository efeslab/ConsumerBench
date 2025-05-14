import os
import json
import random
import argparse
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from datasets import load_dataset

# Constants for the MMLU benchmark
MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality", "international_law",
    "jurisprudence", "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting", "professional_law", "professional_medicine",
    "professional_psychology", "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions"
]

class MMLUBenchmark:
    def __init__(self, api_base, api_key=None, model="gpt-3.5-turbo", subjects=None, 
                 few_shot=5, max_samples=None, output_dir="results", batch_size=16,
                 system_prompt=None):
        """
        Initialize the MMLU benchmark.
        
        Args:
            api_base: Base URL for the OpenAI-compatible API (e.g., "http://localhost:8000/v1")
            api_key: API key (if needed)
            model: Model name to use
            subjects: List of MMLU subjects to test (None for all)
            few_shot: Number of few-shot examples (0 for zero-shot)
            max_samples: Maximum number of samples to test per subject (None for all)
            output_dir: Directory to save results
            batch_size: Number of queries to send in each batch
            system_prompt: System prompt to control model behavior (None for default)
        """
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.subjects = subjects if subjects else MMLU_SUBJECTS
        self.few_shot = few_shot
        self.max_samples = max_samples
        self.output_dir = output_dir
        self.batch_size = batch_size
        
        # Default system prompt to use if none provided
        self.system_prompt = system_prompt or (
            "You are a helpful assistant taking a multiple-choice test. "
            "For each question, select the best answer from the provided options. "
            "You MUST directly output your answer using the format 'The correct choice is X', "
            "where X is A, B, C, or D. Do not provide any explanations or analysis."
            "No other text should be included in the response."
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Options for the multiple choice questions
        self.options = ["A", "B", "C", "D"]
        
    def download_dataset(self):
        """Download the MMLU dataset."""
        print("Loading MMLU dataset...")
        self.dataset = {}
        
        for split in ["dev", "test"]:
            self.dataset[split] = {}
            for subject in tqdm(self.subjects, desc=f"Loading {split} datasets"):
                try:
                    data = load_dataset("cais/mmlu", subject, split=split)
                    self.dataset[split][subject] = data
                except Exception as e:
                    print(f"Error loading {subject} ({split}): {e}")
                    continue
        
        print("Dataset loaded successfully.")
    
    def prepare_few_shot_examples(self, subject):
        """Prepare few-shot examples for a given subject."""
        if self.few_shot == 0:
            return ""
            
        if subject not in self.dataset["dev"]:
            print(f"Warning: No dev set found for {subject}. Using zero-shot.")
            return ""
            
        examples = []
        dev_data = self.dataset["dev"][subject]
        
        # Select random examples for few-shot learning
        indices = random.sample(range(len(dev_data)), min(self.few_shot, len(dev_data)))
        
        for idx in indices:
            question = dev_data[idx]["question"]
            choices = dev_data[idx]["choices"]
            answer_idx = dev_data[idx]["answer"]
            answer = self.options[answer_idx]
            
            example = f"Question: {question}\n"
            for i, choice in enumerate(choices):
                example += f"{self.options[i]}: {choice}\n"
            example += f"Answer: The correct choice is {answer}\n\n"
            examples.append(example)
            
        return "".join(examples)
    
    def query_model(self, prompt):
        """Query the model via the OpenAI-compatible API."""
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,  # Use deterministic responses
            "max_tokens": 5000  # We only need a short response
        }
        
        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Error querying model: {e}")
            return None
    
    def extract_answer(self, response):
        """Extract the answer (A, B, C, or D) from the model's response."""
        if not response:
            return None
            
        # Check for "The correct choice is X" pattern
        import re

        # strip off <think> and </think> tags if they exist, and anything in between
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        match = re.search(r"The correct choice is ([A-D])", response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
            
        # # Look for the first occurrence of A, B, C, or D in the response
        # for char in response:
        #     if char.upper() in self.options:
        #         return char.upper()
                
        # # If no direct match, try to find the option in the response
        # response = response.upper()
        # for opt in self.options:
        #     if opt in response:
        #         return opt
                
        # # If still no match, check for option-like patterns (e.g., "option a", "choice a")
        # for opt in self.options:
        #     if f"OPTION {opt}" in response or f"CHOICE {opt}" in response:
        #         return opt
                
        # Fall back to the first character if it's a letter
        # if response and response[0].isalpha():
        #     return response[0].upper()
            
        return None
    
    def batch_query_model(self, prompts, batch_size=16):
        """Query the model with a batch of prompts."""
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Process batches of prompts
        all_responses = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_requests = [
                {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 5000
                }
                for prompt in batch_prompts
            ]
            
            try:
                # Use async requests if available, otherwise do sequential batch processing
                try:
                    import asyncio
                    import aiohttp
                    
                    async def fetch_responses():
                        async with aiohttp.ClientSession() as session:
                            tasks = []
                            for req in batch_requests:
                                tasks.append(
                                    session.post(
                                        f"{self.api_base}/chat/completions",
                                        headers=headers,
                                        json=req,
                                        timeout=600
                                    )
                                )
                            responses = await asyncio.gather(*tasks, return_exceptions=True)
                            
                            # Process responses properly
                            batch_responses = []
                            for resp in responses:
                                if isinstance(resp, Exception):
                                    batch_responses.append(None)
                                else:
                                    try:
                                        # Need to await the json parsing
                                        resp_json = await resp.json()
                                        batch_responses.append(resp_json["choices"][0]["message"]["content"].strip())
                                    except Exception as e:
                                        print(f"Error parsing response: {e}")
                                        batch_responses.append(None)
                            return batch_responses
                    
                    # Run the async requests and get parsed responses directly
                    batch_responses = asyncio.run(fetch_responses())
                
                except (ImportError, ModuleNotFoundError):
                    # Fall back to sequential processing if async libraries aren't available
                    batch_responses = []
                    for req in batch_requests:
                        try:
                            response = requests.post(
                                f"{self.api_base}/chat/completions",
                                headers=headers,
                                json=req,
                                timeout=30
                            )
                            response.raise_for_status()
                            batch_responses.append(response.json()["choices"][0]["message"]["content"].strip())
                        except Exception as e:
                            print(f"Error in sequential request: {e}")
                            batch_responses.append(None)
                
                all_responses.extend(batch_responses)
                
            except Exception as e:
                print(f"Error in batch query: {e}")
                # On batch failure, append None for each prompt in this batch
                all_responses.extend([None] * len(batch_prompts))
        
        return all_responses
    
    def run_benchmark(self):
        """Run the MMLU benchmark."""
        print(f"Running MMLU benchmark with {self.model} using {self.few_shot}-shot examples")
        
        # Download the dataset if not already done
        if not hasattr(self, 'dataset'):
            self.download_dataset()
            
        results = {}
        all_correct = 0
        all_total = 0
        
        for subject in self.subjects:
            if subject not in self.dataset["test"]:
                print(f"Skipping {subject}: No test data available")
                continue
                
            print(f"Testing {subject}...")
            test_data = self.dataset["test"][subject]
            few_shot_examples = self.prepare_few_shot_examples(subject)
            
            # Determine the number of samples to test
            num_samples = min(len(test_data), self.max_samples) if self.max_samples else len(test_data)
            
            # Prepare all prompts for this subject
            all_prompts = []
            all_correct_answers = []
            all_questions_data = []
            
            for i in range(num_samples):
                question = test_data[i]["question"]
                choices = test_data[i]["choices"]
                correct_idx = test_data[i]["answer"]
                correct_answer = self.options[correct_idx]
                
                # Prepare the prompt
                test_prompt = f"{few_shot_examples}Question: {question}\n"
                for j, choice in enumerate(choices):
                    test_prompt += f"{self.options[j]}: {choice}\n"
                test_prompt += "Answer:"
                
                all_prompts.append(test_prompt)
                all_correct_answers.append(correct_answer)
                all_questions_data.append({
                    "question": question,
                    "choices": choices,
                    "correct_answer": correct_answer
                })
            
            # Query the model with all prompts in batches
            batch_size = self.batch_size
            print(f"Processing {len(all_prompts)} questions in batches of {batch_size}...")

            all_prompts = all_prompts[:16]

            all_responses = self.batch_query_model(all_prompts, batch_size)
            
            # Process the responses
            correct = 0
            total = 0
            all_answers = []
            
            for i, response in enumerate(tqdm(all_responses, desc=subject)):
                predicted_answer = self.extract_answer(response)
                correct_answer = all_correct_answers[i]
                
                # Record the result
                is_correct = predicted_answer == correct_answer
                question_data = all_questions_data[i]
                
                all_answers.append({
                    "subject": subject,
                    "question": question_data["question"],
                    "choices": question_data["choices"],
                    "correct_answer": correct_answer,
                    "predicted_answer": predicted_answer,
                    "is_correct": is_correct,
                    "full_response": response
                })
                
                if is_correct:
                    correct += 1
                total += 1
                
            # Calculate accuracy for this subject
            accuracy = correct / total if total > 0 else 0
            results[subject] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": total
            }
            
            print(f"{subject} accuracy: {accuracy:.4f} ({correct}/{total})")
            
            all_correct += correct
            all_total += total
            
            # Save detailed results for this subject
            with open(f"{self.output_dir}/{subject}_details.json", "w") as f:
                json.dump(all_answers, f, indent=2)
                
        # Calculate overall accuracy
        overall_accuracy = all_correct / all_total if all_total > 0 else 0
        results["overall"] = {
            "accuracy": overall_accuracy,
            "correct": all_correct,
            "total": all_total
        }
        
        print(f"\nOverall accuracy: {overall_accuracy:.4f} ({all_correct}/{all_total})")
        
        # Save summary results
        with open(f"{self.output_dir}/summary.json", "w") as f:
            json.dump(results, f, indent=2)
            
        # Create a DataFrame for easier analysis
        df_results = pd.DataFrame([
            {"subject": subject, "accuracy": data["accuracy"], 
             "correct": data["correct"], "total": data["total"]}
            for subject, data in results.items() if subject != "overall"
        ])
        
        # Sort by accuracy
        df_results = df_results.sort_values("accuracy", ascending=False)
        df_results.to_csv(f"{self.output_dir}/results.csv", index=False)
        
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MMLU benchmark on a local LLM")
    parser.add_argument("--api-base", type=str, required=True, help="Base URL for the API")
    parser.add_argument("--api-key", type=str, default=None, help="API key (if needed)")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Model name")
    parser.add_argument("--subjects", nargs="+", default=None, 
                        help="Specific subjects to test (default: all)")
    parser.add_argument("--few-shot", type=int, default=5, 
                        help="Number of few-shot examples (0 for zero-shot)")
    parser.add_argument("--max-samples", type=int, default=None, 
                        help="Maximum samples per subject")
    parser.add_argument("--output-dir", type=str, default="results", 
                        help="Directory to save results")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Number of queries to send in each batch")
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="Custom system prompt to use (default: predefined prompt)")
    
    args = parser.parse_args()
    
    benchmark = MMLUBenchmark(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        subjects=args.subjects,
        few_shot=args.few_shot,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        system_prompt=args.system_prompt
    )
    
    benchmark.run_benchmark()