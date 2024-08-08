import json
import re
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# File paths
DATASET_FILE = "dataset/dataset.json"
OUTPUT_FILE_TEMPLATE = "results/0808_test/generate_all_{}.json"

# Initialize LLM
llm = ChatOllama(
    model="llama3.1:latest",
    keep_alive=-1,
    temperature=0.2,
    max_new_tokens=8192
)

# Load dataset
with open(DATASET_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)

data = data[:100]

titles = [item["title"] for item in data]
categories = [item["category"] for item in data]
transcripts = [item["transcript"] for item in data]

# Detailed Summarization Prompt Template
detailed_summary_prompt = """
You are an expert summarizer. Based on the following video transcript, follow these steps to provide a detailed summary that captures the main points, key facts, and important details. Ensure the summary is comprehensive and covers all significant aspects of the content.

Step 1: Identify Main Ideas and Key Points
- Extract the main ideas from each section of the transcript.
- Identify key facts, important details, and significant findings.

Step 2: Structure the Summary
- Organize the summary into an introduction, body, and conclusion.
- Ensure each part covers different aspects of the content.

Step 3: Maintain an Objective Tone
- Summarize the content objectively without personal bias.
- Use neutral language to convey the information.

Step 4: Ensure Comprehensive Coverage
- Include essential aspects such as context, methodology, results, and conclusions.
- Avoid omitting critical information.

Step 5: Ensure Logical Flow
- Use transition words and phrases to connect ideas.
- Ensure the summary flows logically from one point to the next.

Transcript:
{transcript}
"""

# Simple Summarization Prompt Template
simple_summary_prompt = """
You are an expert summarizer. Based on the following video transcript, provide a concise summary that captures the main points, key facts, and important details.

Transcript:
{transcript}
"""

# MCQ Generation from Summary Prompt Template
mcq_from_summary_prompt = """
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

# Function to generate detailed summary
def generate_detailed_summary(transcript):
    prompt = ChatPromptTemplate.from_template(detailed_summary_prompt)
    chain = prompt | llm | StrOutputParser()
    generated_chunks = []
    for chunk in chain.stream({"transcript": transcript}):
        generated_chunks.append(chunk)
    generated_output = "".join(generated_chunks)
    return generated_output.strip()

# Function to generate simple summary
def generate_simple_summary(transcript):
    prompt = ChatPromptTemplate.from_template(simple_summary_prompt)
    chain = prompt | llm | StrOutputParser()
    generated_chunks = []
    for chunk in chain.stream({"transcript": transcript}):
        generated_chunks.append(chunk)
    generated_output = "".join(generated_chunks)
    return generated_output.strip()

# Function to generate questions from summary
def generate_questions_from_summary(category, summary):
    prompt = ChatPromptTemplate.from_template(mcq_from_summary_prompt)
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

# Process each transcript to generate questions using both summarizers
# all_questions_detailed = []
all_questions_simple = []
total_videos = len(titles)

for idx, (title, category, transcript) in enumerate(zip(titles, categories, transcripts)):
    # # Detailed Summary Method
    # detailed_summary = generate_detailed_summary(transcript)
    # generated_output_detailed = generate_questions_from_summary(category, detailed_summary)
    # questions_detailed = parse_questions(generated_output_detailed)
    # all_questions_detailed.append({
    #     "title": title,
    #     "category": category,
    #     "multiple-choice": questions_detailed
    # })
    
    # Simple Summary Method
    simple_summary = generate_simple_summary(transcript)
    generated_output_simple = generate_questions_from_summary(category, simple_summary)
    questions_simple = parse_questions(generated_output_simple)
    all_questions_simple.append({
        "title": title,
        "category": category,
        "multiple-choice": questions_simple
    })
    
    print(f"Progress: {idx + 1}/{total_videos} videos processed")

# # Save detailed summary MCQs
# output_file_detailed = OUTPUT_FILE_TEMPLATE.format("detailed_summary_to_mcq")
# with open(output_file_detailed, "w") as file:
#     json.dump(all_questions_detailed, file, indent=4)
# print(f"Detailed Summary Questions have been saved to {output_file_detailed}")

# Save simple summary MCQs
output_file_simple = OUTPUT_FILE_TEMPLATE.format("simple_summary_to_mcq")
with open(output_file_simple, "w") as file:
    json.dump(all_questions_simple, file, indent=4)
print(f"Simple Summary Questions have been saved to {output_file_simple}")
