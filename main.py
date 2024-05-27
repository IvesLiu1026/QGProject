import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import ijson
import re

DATASET_FILE = "D:/NYCU1122courses/Project/ollama/dataset/dataset.json"
OUTPUT_FILE = "generated_questions.json"

# Initialize the ChatOllama model
llm = ChatOllama(
    model="dolphin-llama3:latest",
    keep_alive=-1,  # keep the model loaded indefinitely
    temperature=0.7,
    max_new_tokens=8192  # Adjust tokens to handle multiple questions and choices
)

with open(DATASET_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)
    
data = data[:2]

titles = [item["title"] for item in data]
categories = [item["category"] for item in data]
transcripts = [item["transcript"] for item in data]


# transcript = "How does the difference between point 0000000398 and point 00000000398 cause one to have red eyes after swimming? To answer this, we first need a way of dealing with rather small numbers, or in some cases extremely large numbers. This leads us\nto the concept of logarithms. Well, what are logarithms? Let's take the base number, b, and raise it to a power, p, like 2 to the 3rd power, and have it equal a number n. We get an exponential equation:\nb raised to the p power, equals n. In our example, that'd be 2 raised to the 3rd power, equals 8. The exponent p is said to be the logarithm of the number n. Most of the time this would be written: \"log, base b, of a number\nequals p, the power.\" This is starting to sound a bit confusing\nwith all the variables, so let's show this with an example. What is the value of log base 10 of 10,000? The same question could be asked\nusing exponents: \"10 raised to what power is 10,000?\" Well, 10 to the 4th is 10,000. So, log base 10 of 10,000 must equal 4. This example can also be completed\nvery simply on a scientific calculator. Log base 10 is used so frequently in the sciences that it has the honor of having\nits own button on most calculators. If the calculator will figure out\nlogs for me, why study them? Just a quick reminder: the log button only computes\nlogarithms of base 10. What if you want to go into\ncomputer science and need to understand base 2? So what is log base 2 of 64? In other words,\n2 raised to what power is 64? Well, use your fingers.\n2, 4, 8, 16, 32, 64. So log base 2 of 64 must equal 6. So what does this have to do\nwith my eyes turning red in some swimming pools and not others? Well, it leads us into an interesting use of logarithms in chemistry: finding the pH of water samples. pH tells us how acidic\nor basic a sample is, and can be calculated with the formula:  pH equals negative log base 10 of\nthe hydrogen ion concentration, or H plus. We can find the pH of water samples with hydrogen ion concentration of\npoint 0000000398 and point 00000000398 quickly on a calculator. Punch: negative log\nof each of those numbers, and you'll see the pH's are 7.4 and 8.4. Since the tears in our eyes\nhave a pH of about 7.4, the H plus concentration of .0000000398 will feel nice on your eyes, but the pH of 8.4\nwill make you feel itchy and red. It's easy to remember logarithms\n\"log base b of some number n equals p\" by repeating: \"The base raised\nto what power equals the number?\" \"The BASE raised to what POWER\nequals the NUMBER?\" So now we know\nlogarithms are very powerful when dealing with\nextremely small or large numbers. Logarithms can even be used instead of eyedrops after swimming."
# category = "Numbers & Operations"

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
    # Define the prompt
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Create the chain
    chain = prompt | llm | StrOutputParser()

    # Collect chunks in a list
    generated_chunks = []

    # Generate questions and collect the output
    for chunk in chain.stream({"category": category, "transcript": transcript}):
        print(chunk, end="", flush=True)
        generated_chunks.append(chunk)

    # Combine the chunks into a single string
    generated_output = "".join(generated_chunks)
    print(generated_output)
    return generated_output



# Function to parse the generated questions and convert them to the desired JSON format
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

# Parse the generated output and write to a file
all_questions = []
for title, category, transcript in zip(titles, categories, transcripts):
    generated_output = generate_questions(category, transcript)
    questions = parse_questions(generated_output)
    all_questions.append({
        "title": title,
        "category": category,
        "multiple-choice": questions
    })

# Write the output to a file
with open(OUTPUT_FILE, "w") as file:
    json.dump(all_questions, file, indent=4)

print(f"Questions have been saved to {OUTPUT_FILE}")