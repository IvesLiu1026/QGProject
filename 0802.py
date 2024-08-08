import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

DATASET_FILE = "dataset/dataset.jsonl"
OUTPUT_FILE = "results/generate_temp.json"

llm = ChatOllama(
    model="gemma:latest",
    keep_alive=-1,
    temperature=0.2,
    max_new_tokens=8192
)
prompt_template = """
user
You are an expert educational content creator. Based on the following video transcript and its category, generate five multiple-choice questions that can help understand the content better. Each question should have four choices labeled A, B, C, and D. The questions should cover key facts, definitions, and important details mentioned in the transcript. Ensure the options are plausible to avoid easy guessing. Provide the questions and answers in the following format:

1) [Your question here]
    - A: [Option A]
    - B: [Option B]
    - C: [Option C]
    - D: [Option D]
Distractor answer: [Distractor options]
Correct answer: [Correct option]

Category: {category}

Transcript:
{transcript}
assistant
"""
def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8-sig") as file:
        for line in file:
            yield json.loads(line)

data = list(read_jsonl(DATASET_FILE))[0:2]

titles = [item["title"] for item in data]
categories = [item["category"] for item in data]
transcripts = [item["transcript"]["en"] for item in data]
def generate_questions(category, transcript):
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()
    generated_chunks = []

    for chunk in chain.stream({"category": category, "transcript": transcript}):
        generated_chunks.append(chunk)

    generated_output = "".join(generated_chunks)
    return generated_output
def parse_questions(generated_output):
    questions = []
    current_question = None
    current_options = []
    distractor_option = None
    correct_option = None

    lines = generated_output.strip().split('\n')
    for line in lines:
        question_match = re.match(r'\d+\)\s*(.*)', line.strip())
        option_match = re.match(r'-\s*([A-D]):\s*(.*)', line.strip())
        distractor_match = re.match(r'Distractor answer:\s*(.*)', line.strip(), re.IGNORECASE)
        answer_match = re.match(r'Correct answer:\s*([A-D])', line.strip(), re.IGNORECASE)

        if question_match:
            if current_question:
                questions.append({
                    "question": current_question,
                    "options": current_options,
                    "distractor_option": distractor_option,
                    "correct_option": correct_option
                })
            current_question = question_match.group(1)
            current_options = []
        elif option_match:
            current_options.append({
                "label": option_match.group(1),
                "text": option_match.group(2)
            })
        elif distractor_match:
            distractor_option = distractor_match.group(1)
        elif answer_match:
            correct_option = answer_match.group(1)

    if current_question:
        questions.append({
            "question": current_question,
            "options": current_options,
            "distractor_option": distractor_option,
            "correct_option": correct_option
        })

    return questions
def generate_and_parse(title, category, transcript):
    generated_output = generate_questions(category, transcript)
    questions = parse_questions(generated_output)
    return {
        "title": title,
        "category": category,
        "multiple-choice": questions
    }

def process_transcripts(titles, categories, transcripts):
    all_questions = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(generate_and_parse, title, category, transcript)
            for title, category, transcript in zip(titles, categories, transcripts)
        ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                all_questions.append(result)

    return all_questions

all_questions = process_transcripts(titles, categories, transcripts)
def write_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

write_json(all_questions, OUTPUT_FILE)
print(f"Questions have been saved to {OUTPUT_FILE}")
