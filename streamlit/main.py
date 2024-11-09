import re
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Initialize LLM
llm = ChatOllama(
    model="llama3.1:8b",
    keep_alive=-1,
    temperature=0.7,
    max_new_tokens=8192
)


# Update the function signature to accept category
def generate_questions(prompt_template, category, text):
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()
    generated_chunks = [chunk for chunk in chain.stream({"category": category, "transcript": text})]
    return "".join(generated_chunks)


# Function to parse generated questions
def parse_questions(generated_output):
    questions = []
    current_question, current_options = None, []
    lines = generated_output.strip().split('\n')
    for line in lines:
        question_match = re.match(r'\d+\)\s*(.*)', line.strip())
        option_match = re.match(r'-\s*([A-D]):\s*(.*)', line.strip())
        if question_match:
            if current_question:
                questions.append({"question": current_question, "options": current_options})
            current_question = question_match.group(1)
            current_options = []
        elif option_match:
            current_options.append({"label": option_match.group(1), "text": option_match.group(2)})
    if current_question:
        questions.append({"question": current_question, "options": current_options})
    return questions
