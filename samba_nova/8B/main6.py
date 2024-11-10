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
    api_key=os.getenv("SOLO_TO_SAMBANOVA_API_KEY"),
    base_url="https://api.sambanova.ai/v1",
)

DATASET_DIR = "../dataset/split_dataset"
OUTPUT_FILE = "../results/1106/refinement_bt/page_" + page_number + ".jsonl"

dataset_file = f"{DATASET_DIR}/dataset_page_{page_number}.jsonl"

# Load dataset
with open(dataset_file, "r", encoding="utf-8-sig") as file:
    data = [json.loads(line) for line in file]

# Extract titles, categories, transcripts
titles = [item["title"] for item in data]
categories = [item["category"] for item in data]
transcripts = [item["transcript"]["en"] for item in data]

purpose_driven_prompt = """
You are an expert educational content creator. Based on the following transcript and the purpose of the questions, generate ten multiple-choice questions and including one correct answer and three distractors.

Types of Questions:
1. Factual Questions: Test recall of specific facts.
2. Conceptual Questions: Assess understanding of concepts.
3. Application Questions: Test ability to apply knowledge to new situations.
4. Analytical Questions: Require analysis and interpretation of information.

Provide the questions and answers in the following format:
1) [Your question here]
    - A: [Option A]
    - B: [Option B]
    - C: [Option C]
    - D: [Option D]
[Your answer here]: [Correct option]

Category: {category}

Transcript:
{transcript}
"""


# Bloom's Taxonomy Prompt
bloom_taxonomy_prompt = """
You are an expert educational content creator. Based on the following transcript, generate ten multiple-choice questions that span across Bloom's Taxonomy levels. Each question should assess one of the following cognitive levels:
You don't have to include the levels in the questions, but ensure that the questions cover a range of cognitive skills.

1. **Knowledge**: Questions that test recall of specific facts from the transcript.
2. **Comprehension**: Questions that check understanding by asking learners to summarize or explain.
3. **Application**: Questions that test the ability to apply knowledge in real-world scenarios.
4. **Analysis**: Questions that require breaking down information into parts and understanding relationships.
5. **Synthesis**: Questions that encourage combining ideas to form new insights or solutions.
6. **Evaluation**: Questions that ask for judgment based on criteria or justification of opinions.

Please provide the questions and answers in the following format:
1) [Your question here]
    - A: [Option A]
    - B: [Option B]
    - C: [Option C]
    - D: [Option D]
[Correct answer]: [Correct option]

Category: {category}

Transcript:
{transcript}
"""

# Refinement Prompt
refinement_prompt = """
You are an expert educational content creator. Based on the following transcript and the initial set of multiple-choice questions, please review and refine the questions to ensure they are:

- Clear and unambiguous.
- Factually accurate and aligned with the transcript content.
- Appropriately challenging for the intended audience.
- Covering key concepts and important details from the transcript.
- Each question has one correct answer and three plausible distractors.

Instructions:

- Correct any factual errors in the questions or answer choices.
- Improve the wording for clarity and conciseness.
- Ensure the distractors are plausible but clearly incorrect.
- Remove any ambiguity in the questions or answer choices.
- Maintain the original format.

Provide the refined questions and answers in the following format:

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
"""

# Constants for retry mechanism and maximum refinement cycles
MAX_RETRIES = 10
INITIAL_BACKOFF = 5
MAX_REFINEMENT_CYCLES = 10  # Set a limit on the number of refinement cycles
BACKOFF_MULTIPLIER = 2

def generate_questions(prompt, category, transcript):
    retries = 0
    backoff = INITIAL_BACKOFF
    prompt_formatted = prompt.format(category=category, transcript=transcript)

    while retries < MAX_RETRIES:
        try:
            response = client.chat.completions.create(
                model="Meta-Llama-3.1-8B-Instruct",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert educational content creator. "
                            "Your task is to generate comprehensive, multiple-choice questions based on a provided transcript."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt_formatted
                    }
                ],
                temperature=0.7,
                top_p=0.9,
            )
            generated_output = response.choices[0].message.content.strip()
            # print(f"Generated output: {generated_output}")  # Debug: Log generated output
            return generated_output

        except openai.RateLimitError as e:
            print(f"Rate limit error encountered: {e}. Retrying in {backoff} seconds...")
            time.sleep(backoff)  # Wait before retrying
            retries += 1
            backoff = min(backoff * BACKOFF_MULTIPLIER, 60)  # Cap backoff at 60 seconds

    print("Max retries reached. Skipping this request.")
    return None

def refine_questions(transcript, initial_questions_str):
    retries = 0
    backoff = INITIAL_BACKOFF
    prompt_formatted = refinement_prompt.format(transcript=transcript, initial_questions=initial_questions_str)

    while retries < MAX_RETRIES:
        try:
            response = client.chat.completions.create(
                model="Meta-Llama-3.1-405B-Instruct",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert educational content creator. "
                            "Your task is to refine multiple-choice questions based on a provided transcript and initial questions."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt_formatted
                    }
                ],
                temperature=0.7,
                top_p=0.9,
            )
            refined_output = response.choices[0].message.content.strip()
            # print(f"Refined output: {refined_output}")  # Debug: Log refined output
            return refined_output

        except openai.RateLimitError as e:
            print(f"Rate limit error encountered: {e}. Retrying in {backoff} seconds...")
            time.sleep(backoff)  # Wait before retrying
            retries += 1
            backoff = min(backoff * BACKOFF_MULTIPLIER, 60)  # Cap backoff at 60 seconds

    print("Max retries reached. Skipping this request.")
    return None

def parse_questions(generated_output):
    if not generated_output:
        print("No generated output to parse.")
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

    # print(f"Parsed questions: {questions}")
    return questions

results = []

for idx, (title, category, transcript) in enumerate(zip(titles, categories, transcripts)):
    # Initial generation using Bloom's Taxonomy Prompt
    # generated_output = generate_questions(bloom_taxonomy_prompt, category, transcript)
    generated_output = generate_questions(purpose_driven_prompt, category, transcript)

    parsed_questions = parse_questions(generated_output)

    if not parsed_questions:
        print(f"Retrying for page {page_number}, video {idx + 1}...")
        continue

    # Refinement loop
    refined_questions = parsed_questions
    for cycle in range(MAX_REFINEMENT_CYCLES):
        initial_questions_str = ''
        for i, q in enumerate(refined_questions, 1):
            initial_questions_str += f"{i}) {q['question']}\n"
            for option in q['options']:
                initial_questions_str += f"    - {option['label']}: {option['text']}\n"
            initial_questions_str += f"[Correct answer]: {q['correct_option']}\n\n"

        refined_output = refine_questions(transcript, initial_questions_str)
        new_refined_questions = parse_questions(refined_output) if refined_output else refined_questions

        # Break if no further refinement is achieved
        if new_refined_questions == refined_questions:
            print(f"No further refinement needed for video {idx + 1} on cycle {cycle + 1}.")
            break

        refined_questions = new_refined_questions

    # Append final refined questions
    results.append({
        "title": title,
        "category": category,
        "multiple-choice": refined_questions
    })

    print(f"Completed refinement for video {idx + 1} of page {page_number}")

# Save results to file
with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    for entry in results:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"All questions combined and saved to {OUTPUT_FILE}")
