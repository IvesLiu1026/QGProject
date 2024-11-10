import os
import openai
import json
import re
import time
from dotenv import load_dotenv
import multiprocessing
import math

# Set base directory paths
DATASET_DIR = "../../dataset/split_dataset"
OUTPUT_DIR = "../../results/1108/CoT"
INTERMEDIATE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "intermediate_outputs")
ERROR_LOG_FILE = os.path.join(OUTPUT_DIR, "error_log.txt")

# List of API keys (add your 17 keys here)
API_KEYS = [
    "ALEX1_SAMBANOVA_API_KEY", 
    "ALEX2_SAMBANOVA_API_KEY", 
    "ALEX3_SAMBANOVA_API_KEY", 
    "ALEX4_SAMBANOVA_API_KEY", 
    "ALEX5_SAMBANOVA_API_KEY",
    "ALEX6_SAMBANOVA_API_KEY", 
    "ALEX7_SAMBANOVA_API_KEY", 
    "ALEX8_SAMBANOVA_API_KEY", 
    "ALEX9_SAMBANOVA_API_KEY", 
    "ALEX10_SAMBANOVA_API_KEY",
    "ALEX11_SAMBANOVA_API_KEY", 
    "NYCU_IVES_SAMBANOVA_API_KEY", 
    "TW_IVES_SAMBANOVA_API_KEY", 
    "BRIAN_SAMBANOVA_API_KEY", 
    "YU_XIANG_SAMBANOVA_API_KEY",
    "SOLO_TO_SAMBANOVA_API_KEY", 
    "BA_VO_SAMBANOVA_API_KEY"
]

# Total pages to process
TOTAL_PAGES = 122
PAGES_PER_API = math.ceil(TOTAL_PAGES / len(API_KEYS))

# CoT Prompts
content_analysis_prompt = """
Analyze the following transcript and identify key information, categorizing it under each Bloom's Taxonomy level. This analysis will serve as the foundation for creating assessment questions.

1. **Knowledge**: List specific facts or details that can be recalled directly.
2. **Comprehension**: Summarize important concepts that require understanding or interpretation.
3. **Application**: Describe scenarios or real-world applications where the content can be applied.
4. **Analysis**: Identify relationships, contrasts, or cause-and-effect connections in the content.
5. **Synthesis**: Provide insights where ideas could be combined to form new understanding.
6. **Evaluation**: List any parts of the transcript where judgment, critique, or justification would be appropriate.

**Transcript**:
{transcript}
"""

initial_question_generation_prompt = """
Using the content analysis provided, create two multiple-choice questions for each Bloom's Taxonomy level. Each question should include one correct answer and three plausible distractors. Ensure the questions span across these levels:

1. **Knowledge**: Test recall of specific facts.
2. **Comprehension**: Assess understanding of key concepts.
3. **Application**: Present scenarios that require applying the information.
4. **Analysis**: Form questions that require breaking down or examining information.
5. **Synthesis**: Develop questions that require combining ideas or creating new insights.
6. **Evaluation**: Create questions that require judgment or justification.

**Content Analysis**:
{content_analysis}

**Format**:
1) [Your question here]
    - A: [Option A]
    - B: [Option B]
    - C: [Option C]
    - D: [Option D]
[Correct answer]: [Correct option]
"""

refinement_prompt = """
You are an expert educational content creator refining multiple-choice questions. Based on the transcript, initial set of questions, and prior feedback, review and refine the questions. Follow these steps:

1. Ensure each question aligns with the key concepts in the transcript.
2. Enhance clarity and conciseness.
3. Confirm distractors are plausible but incorrect.
4. Ensure correct answers are accurate.

If no further refinement is necessary, indicate by saying "No more refinement needed."

Provide the refined questions in the following format:

1) [Refined question here]
    - A: [Option A]
    - B: [Option B]
    - C: [Option C]
    - D: [Option D]
[Correct answer]: [Correct option]

Transcript:
{transcript}

Initial Questions:
{initial_questions}

Feedback:
{feedback}
"""

# Function to sanitize filenames
def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

# Function to log errors
def log_error(message):
    with open(ERROR_LOG_FILE, "a") as error_file:
        error_file.write(message + "\n")

# Function to call the OpenAI API
def call_openai_api(api_key, prompt, retries=10):
    backoff = 5
    for attempt in range(retries):
        try:
            client = openai.OpenAI(api_key=api_key, base_url="https://api.sambanova.ai/v1")
            response = client.chat.completions.create(
                model="Meta-Llama-3.1-405B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                top_p=0.9,
            )
            return response.choices[0].message.content.strip()
        except openai.RateLimitError as e:
            log_error(f"Rate limit error with {api_key}. Retry {attempt + 1}/10: {e}")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
    log_error("Max retries reached for API key: " + api_key)
    return None

# Main function to process a set of pages with a specific API key
def process_pages(api_key, pages):
    for page_number in pages:
        dataset_file = os.path.join(DATASET_DIR, f"dataset_page_{page_number}.jsonl")
        output_file = os.path.join(OUTPUT_DIR, f"page_{page_number}.jsonl")
        
        # Load dataset
        with open(dataset_file, "r", encoding="utf-8-sig") as file:
            data = [json.loads(line) for line in file]

        results = []
        for idx, item in enumerate(data):
            title = item["title"]
            category = item["category"]
            transcript = item["transcript"]["en"]

            # Step 1: Content Analysis
            prompt = content_analysis_prompt.format(transcript=transcript)
            content_analysis_output = call_openai_api(api_key, prompt)
            if not content_analysis_output:
                log_error(f"Failed to perform content analysis for page {page_number}, item {idx + 1}")
                continue

            # Step 2: Generate Initial Questions (similar to above; skipped for brevity)
            # Save intermediate output, perform refinements, etc.

            # Store results
            results.append({
                "title": title,
                "category": category,
                "multiple-choice": []  # Add your generated/refined questions here
            })

        # Save results to file
        with open(output_file, "w", encoding="utf-8") as file:
            for entry in results:
                file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Page {page_number} completed using API key {api_key}")

# Distribute pages across processes
def main():
    # Split pages into chunks for each API key
    page_chunks = [list(range(i * PAGES_PER_API + 1, min((i + 1) * PAGES_PER_API + 1, TOTAL_PAGES + 1))) for i in range(len(API_KEYS))]

    # Create a multiprocessing pool with 17 processes
    with multiprocessing.Pool(processes=len(API_KEYS)) as pool:
        # Launch each process with its respective API key and pages
        pool.starmap(process_pages, zip(API_KEYS, page_chunks))

if __name__ == "__main__":
    main()