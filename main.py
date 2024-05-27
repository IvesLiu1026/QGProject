import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import ijson
import re

DATASET_FILE = "../dataset/dataset.json"
OUTPUT_FILE = "../results/TWCC_generate.json"

llm = ChatOllama(
    model="dolphin-llama3",
    keep_alive=-1,
    temperature=0.7,
    max_new_tokens=8192
)

with open(DATASET_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)
    
# data = data[:2]

titles = [item["title"] for item in data]
categories = [item["category"] for item in data]
transcripts = [item["transcript"] for item in data]

# Dolphin-llama3 model template
"""
{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

prompt_template = """
user
You are an expert educational content creator. Based on the following video transcript and its category, generate five multiple-choice questions that can help understand the content better. Each question should have four choices labeled A, B, C, and D. The questions should cover key facts, definitions, and important details mentioned in the transcript. Ensure the options are plausible to avoid easy guessing. Provide the questions and answers in the following format:

1) [Your question here]
    - A: [Option A]
    - B: [Option B]
    - C: [Option C]
    - D: [Option D]

Category: {category}

Transcript:
{transcript}
assistant
"""

def generate_questions(category, transcript):
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = prompt | llm | StrOutputParser()

    generated_chunks = []

    for chunk in chain.stream({"category": category, "transcript": transcript}):
        print(chunk, end="", flush=True)
        generated_chunks.append(chunk)

    generated_output = "".join(generated_chunks)
    print(generated_output)
    return generated_output


def parse_questions(generated_output):
    questions = []
    current_question = None
    current_options = []

    lines = generated_output.strip().split('\n')
    for line in lines:
        question_match = re.match(r'\d+\)\s*(.*)', line.strip())
        option_match = re.match(r'-\s*([A-D]):\s*(.*)', line.strip())

        if question_match:
            if current_question:
                questions.append({
                    "question": current_question,
                    "options": current_options
                })
            current_question = question_match.group(1)
            current_options = []
        elif option_match:
            current_options.append({
                "label": option_match.group(1),
                "text": option_match.group(2)
            })
    
    if current_question:
        questions.append({
            "question": current_question,
            "options": current_options
        })

    return questions

all_questions = []
for title, category, transcript in zip(titles, categories, transcripts):
    generated_output = generate_questions(category, transcript)
    questions = parse_questions(generated_output)
    all_questions.append({
        "title": title,
        "category": category,
        "multiple-choice": questions
    })

with open(OUTPUT_FILE, "w") as file:
    json.dump(all_questions, file, indent=4)

print(f"Questions have been saved to {OUTPUT_FILE}")