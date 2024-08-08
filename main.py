import json
import re
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# File paths
DATASET_FILE = "dataset/dataset.json"
OUTPUT_FILE_TEMPLATE = "results/generate_all_{}.json"

# Initialize LLM
llm = ChatOllama(
    model="dolphin-llama3:latest",
    keep_alive=-1,
    temperature=0.2,
    max_new_tokens=8192
)

# Load dataset
with open(DATASET_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)

data = data[:2]
titles = [item["title"] for item in data]
categories = [item["category"] for item in data]
transcripts = [item["transcript"] for item in data]

# Summarization prompt template
summary_prompt_template = """
You are an expert summarizer. Based on the following video transcript, provide a concise summary that captures the main points, key facts, and important details.

Transcript:
{transcript}
"""

# Prompt templates for different question generation approaches
# zero-shot
chain_of_thought_prompt = """
You are an expert educational content creator. Based on the following summary and its category, follow these steps to generate five multiple-choice questions:

1. Identify the main themes in the summary.
2. Create a question for each theme, ensuring it targets specific facts or details.
3. Provide four plausible answer choices for each question, with one correct answer.

Provide the questions and answers in the following format:
1) [Your question here]
    - A: [Option A]
    - B: [Option B]
    - C: [Option C]
    - D: [Option D]
[Your answer here]: [Correct option]

Category: {category}

Summary:
{summary}
"""

# 
few_shot_prompt = """
You are an expert educational content creator. Based on the following examples and the provided summary, generate five multiple-choice questions.

Example 1:
1) What is the main theme of the text?
    - A: Education
    - B: Technology
    - C: Healthcare
    - D: Environment
Correct Answer: B

Example 2:
1) Which factor is highlighted as a key challenge?
    - A: Economic instability
    - B: Political unrest
    - C: Technological advancements
    - D: Social inequality
Correct Answer: A

Provide the questions and answers in the following format:
1) [Your question here]
    - A: [Option A]
    - B: [Option B]
    - C: [Option C]
    - D: [Option D]
[Your answer here]: [Correct option]

Category: {category}

Summary:
{summary}
"""

generated_knowledge_prompt = """
You are an expert educational content creator. Based on the following summary and additional context, generate five multiple-choice questions.

Additional Context:
The main causes of World War II include territorial disputes, economic instability, the rise of totalitarian regimes, and the failure of international diplomacy.

Provide the questions and answers in the following format:
1) [Your question here]
    - A: [Option A]
    - B: [Option B]
    - C: [Option C]
    - D: [Option D]
[Your answer here]: [Correct option]

Category: {category}

Summary:
{summary}
"""

self_consistency_prompt = """
You are an expert educational content creator. Based on the following summary and previous questions, generate five multiple-choice questions that are consistent with the given information.

Previous Questions:
1) What is the main theme of the text?
    - A: Technology
    - B: Environment
    - C: Healthcare
    - D: Education
Correct Answer: A

Provide the questions and answers in the following format:
1) [Your question here]
    - A: [Option A]
    - B: [Option B]
    - C: [Option C]
    - D: [Option D]
[Your answer here]: [Correct option]

Category: {category}

Summary:
{summary}
"""

zero_shot_prompt = """
You are an expert educational content creator. Based on the following summary, generate five multiple-choice questions that cover key facts and details. Ensure each question has four answer choices and one correct answer.

Provide the questions and answers in the following format:
1) [Your question here]
    - A: [Option A]
    - B: [Option B]
    - C: [Option C]
    - D: [Option D]
[Your answer here]: [Correct option]

Category: {category}

Summary:
{summary}
"""

# Function to generate summary
def generate_summary(transcript):
    prompt = ChatPromptTemplate.from_template(summary_prompt_template)
    chain = prompt | llm | StrOutputParser()
    generated_chunks = []
    for chunk in chain.stream({"transcript": transcript}):
        generated_chunks.append(chunk)
    generated_output = "".join(generated_chunks)
    return generated_output.strip()

# Function to generate questions from summary
def generate_questions(prompt_template, category, summary):
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()
    generated_chunks = []
    for chunk in chain.stream({"category": category, "summary": summary}):
        generated_chunks.append(chunk)
    generated_output = "".join(generated_chunks)
    return generated_output

# Function to parse generated questions
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

# Different prompt templates
prompts = [
    ("chain_of_thought", chain_of_thought_prompt),
    ("few_shot", few_shot_prompt),
    ("generated_knowledge", generated_knowledge_prompt),
    ("self_consistency", self_consistency_prompt),
    ("zero_shot", zero_shot_prompt)
]

# Process each transcript to generate summaries and questions
for prompt_name, prompt in prompts:
    all_questions = []
    total_videos = len(titles)
    for idx, (title, category, transcript) in enumerate(zip(titles, categories, transcripts)):
        summary = generate_summary(transcript)
        generated_output = generate_questions(prompt, category, summary)
        questions = parse_questions(generated_output)
        all_questions.append({
            "title": title,
            "category": category,
            "multiple-choice": questions
        })
        print(f"Progress: {idx + 1}/{total_videos} videos processed using {prompt_name} prompt")
    
    output_file = OUTPUT_FILE_TEMPLATE.format(prompt_name)
    with open(output_file, "w") as file:
        json.dump(all_questions, file, indent=4)
    
    print(f"Questions have been saved to {output_file} using {prompt_name} prompt")
