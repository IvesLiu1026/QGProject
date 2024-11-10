import os
import openai
import json
import re
import time
from dotenv import load_dotenv
import sys

page_number = sys.argv[1]

load_dotenv()

client = openai.OpenAI(
    api_key=os.getenv("ALEX11_SAMBANOVA_API_KEY"),
    base_url="https://api.sambanova.ai/v1",
)

DATASET_DIR = "../../dataset/split_dataset"
OUTPUT_FILE = f"../../results/1108/CoT/page_{page_number}.jsonl"
ERROR_LOG_FILE = "error_log.txt"
INTERMEDIATE_OUTPUT_DIR = f"../../results/1108/CoT/intermediate_outputs/{page_number}"

if not os.path.exists(INTERMEDIATE_OUTPUT_DIR):
    os.makedirs(INTERMEDIATE_OUTPUT_DIR)

dataset_file = f"{DATASET_DIR}/dataset_page_{page_number}.jsonl"

# Load dataset
with open(dataset_file, "r", encoding="utf-8-sig") as file:
    data = [json.loads(line) for line in file]

# Extract titles, categories, transcripts
titles = [item["title"] for item in data]
categories = [item["category"] for item in data]
transcripts = [item["transcript"]["en"] for item in data]

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

# Constants for retry mechanism and refinement cycles
MAX_RETRIES = 10
INITIAL_BACKOFF = 5
MAX_REFINEMENT_CYCLES = 20
BACKOFF_MULTIPLIER = 2

# Function to sanitize filenames by replacing invalid characters
def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def log_error(message):
    with open(ERROR_LOG_FILE, "a") as error_file:
        error_file.write(message + "\n")

def call_openai_api(prompt, retries=MAX_RETRIES):
    backoff = INITIAL_BACKOFF
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="Meta-Llama-3.1-405B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                top_p=0.9,
            )
            return response.choices[0].message.content.strip()
        except openai.RateLimitError as e:
            log_error(f"Rate limit error. Retry {attempt + 1}/{retries}: {e}")
            time.sleep(backoff)
            backoff = min(backoff * BACKOFF_MULTIPLIER, 60)
    log_error("Max retries reached.")
    return None

def content_analysis(transcript, title):
    sanitized_title = sanitize_filename(title)
    prompt = content_analysis_prompt.format(transcript=transcript)
    analysis_output = call_openai_api(prompt)
    if analysis_output:
        with open(f"{INTERMEDIATE_OUTPUT_DIR}/{sanitized_title}_content_analysis.json", "w", encoding="utf-8") as f:
            json.dump({"content_analysis": analysis_output}, f, ensure_ascii=False, indent=4)
    return analysis_output

# Updated function to handle UTF-8 encoding when writing JSON files
def generate_initial_questions(content_analysis, title):
    prompt = initial_question_generation_prompt.format(content_analysis=content_analysis)
    questions_output = call_openai_api(prompt)
    if questions_output:
        # Use UTF-8 encoding to write JSON file
        with open(f"{INTERMEDIATE_OUTPUT_DIR}/{sanitize_filename(title)}_initial_questions.json", "w", encoding="utf-8") as f:
            json.dump({"initial_questions": questions_output}, f, ensure_ascii=False, indent=4)
    return questions_output

def refine_questions(transcript, initial_questions_str, feedback, title, cycle):
    sanitized_title = sanitize_filename(title)
    prompt = refinement_prompt.format(
        transcript=transcript,
        initial_questions=initial_questions_str,
        feedback=feedback
    )
    refined_output = call_openai_api(prompt)
    if refined_output:
        with open(f"{INTERMEDIATE_OUTPUT_DIR}/{sanitized_title}_refinement_cycle_{cycle}.json", "w", encoding="utf-8") as f:
            json.dump({"refinement_output": refined_output}, f, ensure_ascii=False, indent=4)
    return refined_output

def parse_questions(generated_output):
    if not generated_output:
        return []

    questions = []
    current_question = None
    current_options = []
    correct_option = None
    lines = generated_output.strip().split('\n')

    for line in lines:
        question_match = re.match(r'\d+\)\s*(.*)', line.strip())
        option_match = re.match(r'-\s*([A-D]):\s*(.*)', line.strip())
        answer_match = re.match(r'\[Correct answer\]:\s*([A-D])', line.strip())

        if question_match:
            if current_question:
                questions.append({
                    "question": current_question,
                    "options": current_options,
                    "correct_option": correct_option
                })

            current_question = question_match.group(1)
            current_options = []
            correct_option = None

        elif option_match:
            option_label = option_match.group(1)
            option_text = option_match.group(2)
            current_options.append({
                "label": option_label,
                "text": option_text
            })

        elif answer_match:
            correct_option = answer_match.group(1)

    if current_question:
        questions.append({
            "question": current_question,
            "options": current_options,
            "correct_option": correct_option
        })

    return questions

results = []

for idx, (title, category, transcript) in enumerate(zip(titles, categories, transcripts)):
    # Step 1: Content Analysis
    content_analysis_output = content_analysis(transcript, title)
    if not content_analysis_output:
        log_error(f"Failed to perform content analysis for video {idx + 1} of page {page_number}.")
        continue

    # Step 2: Initial Question Generation
    initial_questions_output = generate_initial_questions(content_analysis_output, title)
    if not initial_questions_output:
        log_error(f"Failed to generate initial questions for video {idx + 1} of page {page_number}.")
        continue

    initial_questions = parse_questions(initial_questions_output)
    initial_questions_str = "\n".join([
        f"{i + 1}) {q['question']}\n" + "\n".join(f"    - {opt['label']}: {opt['text']}" for opt in q['options']) + f"\n[Correct answer]: {q['correct_option']}\n"
        for i, q in enumerate(initial_questions)
    ])

    feedback = "Initial round of feedback."

    # Step 3: Refinement Loop
    for cycle in range(MAX_REFINEMENT_CYCLES):
        refined_output = refine_questions(transcript, initial_questions_str, feedback, title, cycle + 1)
        
        if not refined_output:
            log_error(f"Refinement failed for video {idx + 1} of page {page_number}, cycle {cycle + 1}.")
            break
        
        if "No more refinement needed" in refined_output:
            print(f"No further refinement needed for video {idx + 1} on cycle {cycle + 1}.")
            break
        
        feedback_section = refined_output.split("Questions:", 1)
        feedback = feedback_section[0].strip() if len(feedback_section) > 1 else "Further feedback provided."
        refined_questions = parse_questions(feedback_section[-1])

        initial_questions_str = "\n".join([
            f"{i + 1}) {q['question']}\n" + "\n".join(f"    - {opt['label']}: {opt['text']}" for opt in q['options']) + f"\n[Correct answer]: {q['correct_option']}\n"
            for i, q in enumerate(refined_questions)
        ])
        initial_questions = refined_questions

    results.append({
        "title": title,
        "category": category,
        "multiple-choice": initial_questions
    })

    print(f"Completed refinement for video {idx + 1} of page {page_number}")

# Save results to file
with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    for entry in results:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"All questions combined and saved to {OUTPUT_FILE}")
print(f"Errors, if any, have been logged to {ERROR_LOG_FILE}")
