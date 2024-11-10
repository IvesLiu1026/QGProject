import os
import re
import json
import csv
import openai
import time
from dotenv import load_dotenv

load_dotenv()

# Initialize the OpenAI client
client = openai.OpenAI(
    api_key=os.environ.get("MICAL_SAMBANOVA_API_KEY"),
    base_url="https://api.sambanova.ai/v1",
)

# Set the range of pages to evaluate
start_page = 96  # Starting page number
end_page = 103   # Ending page number

# Load JSONL data
def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data

# Define the matching prompt template
def create_matching_prompt(transcript, ground_truth_mcq, generated_mcqs):
    return """
You are an expert evaluator for educational content. Given a transcript and a set of ground truth multiple-choice questions (MCQs), your task is to match each ground truth MCQ with the most similar generated MCQ from the provided set. To determine similarity, consider the question content, options, and overall intent.

**Transcript**:
{transcript}

**Ground Truth MCQ**:
Question: {gt_question}
Options: {gt_options}
Correct Option: {gt_correct_option}

**Generated MCQs**:
{generated_mcqs}

Please identify the generated MCQ that most closely matches the ground truth MCQ, based on question intent, options, and content.
Provide the index of the most similar generated MCQ.
""".format(
        transcript=transcript,
        gt_question=ground_truth_mcq['question'],
        gt_options=', '.join([option['label'] + '. ' + option['text'] for option in ground_truth_mcq['options']]),
        gt_correct_option=ground_truth_mcq['correct_option'],
        generated_mcqs=''.join([
            "Question {idx}: {question} Options: {options} Correct Option: {correct}\n".format(
                idx=idx + 1,
                question=gq['question'],
                options=', '.join([opt['label'] + '. ' + opt['text'] for opt in gq['options']]),
                correct=gq['correct_option']
            ) for idx, gq in enumerate(generated_mcqs)
        ])
    )

# Define the evaluation prompt template
def create_evaluation_prompt(transcript, ground_truth_mcq, generated_mcq):
    return """
You are an expert evaluator for educational content. Given a transcript, a set of ground truth multiple-choice questions (MCQs), and a set of generated MCQs, your task is to:

1. Match each ground truth MCQ with the most similar generated MCQ from the provided set. To determine similarity, consider the question content, options, and overall intent of the MCQ.

**Transcript**:
{transcript}

**Ground Truth MCQ**:
Question: {gt_question}
Options: {gt_options}
Correct Option: {gt_correct_option}

**Generated MCQ**:
Question: {gen_question}
Options: {gen_options}
Correct Option: {gen_correct_option}

2. Evaluate each matched MCQ pair based on the following criteria:
   - **Relevance (0.0000 - 1.0000)**: Assess how closely the generated question aligns with the content and intent of the ground truth question.
   - **Correct Answer Matching (0.0000 - 1.0000)**: Check if the correct answer option in the generated question matches the ground truth answer.
   - **Distractor Plausibility (0.0000-1.0000)**: Evaluate the quality of the distractors (incorrect options) in the generated question.
   - **Clarity and Readability (0.0000-1.0000)**: Rate how clear and understandable the question and options are.

**Response Format**:
1. **MCQ Pair**: Display both the Ground Truth and Generated MCQ.
2. **Scores**:
   - Relevance Score: X.XXXX
   - Correct Answer Matching Score: X.XXXX
   - Distractor Plausibility Score: X.XXXX
   - Clarity and Readability Score: X.XXXX
   - Total Score: X.XXXX
""".format(
        transcript=transcript,
        gt_question=ground_truth_mcq['question'],
        gt_options=', '.join([option['label'] + '. ' + option['text'] for option in ground_truth_mcq['options']]),
        gt_correct_option=ground_truth_mcq['correct_option'],
        gen_question=generated_mcq['question'],
        gen_options=', '.join([option['label'] + '. ' + option['text'] for option in generated_mcq['options']]),
        gen_correct_option=generated_mcq['correct_option']
    )

# Function to parse scores from the response text
def parse_scores(response_text):
    scores = {
        "Relevance Score": 0.0,
        "Correct Answer Matching Score": 0.0,
        "Distractor Plausibility Score": 0.0,
        "Clarity and Readability Score": 0.0,
        "Total Score": 0.0
    }

    score_patterns = {
        "Relevance Score": r"Relevance Score:\s*([\d.]+)",
        "Correct Answer Matching Score": r"Correct Answer Matching Score:\s*([\d.]+)",
        "Distractor Plausibility Score": r"Distractor Plausibility Score:\s*([\d.]+)",
        "Clarity and Readability Score": r"Clarity and Readability Score:\s*([\d.]+)",
        "Total Score": r"Total Score:\s*([\d.]+)"
    }

    for score_type, pattern in score_patterns.items():
        match = re.search(pattern, response_text)
        if match:
            scores[score_type] = round(float(match.group(1)), 4)

    return scores

# Function to send a request with retry logic for handling rate limits
def send_request_with_retries(prompt, max_retries=10):
    retries = 0
    backoff = 5  # Initial backoff time in seconds

    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model='Meta-Llama-3.1-405B-Instruct',
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                top_p=1.0
            )
            return response.choices[0].message.content.strip()

        except openai.RateLimitError:
            print(f"Rate limit exceeded. Retrying in {backoff} seconds...")
            time.sleep(backoff)
            retries += 1
            backoff *= 2  # Exponential backoff

        except Exception as e:
            print(f"An error occurred: {e}")
            break

    print("Max retries exceeded. Skipping this request.")
    return None

# Main execution loop for each page within the specified range
for page_number in range(start_page, end_page + 1):
    # File paths with corrected directories and unique outputs per page
    GENERATED_FILE = f"../results/1108/CoT_405B/page_{page_number}.jsonl"
    GROUND_TRUTH_FILE = f"../dataset/split_dataset/dataset_page_{page_number}.jsonl"
    OUTPUT_JSON_FILE = f"evaluation/evaluation_scores_page_{page_number}.json"  # Unique JSON output per page
    OUTPUT_CSV_FILE = f"evaluation/evaluation_scores_page_{page_number}.csv"    # Unique CSV output per page
    LOG_FILE = f"evaluation/evaluation_log_page_{page_number}.txt"              # Unique log file per page

    print(f"Processing page {page_number}...")

    generated_questions = load_jsonl(GENERATED_FILE)
    ground_truth_questions = load_jsonl(GROUND_TRUTH_FILE)

    matches_data = []  # Track matches for each page
    scores_data = []   # Track scores for each page

    with open(LOG_FILE, "w", encoding="utf-8") as log_file:
        for idx, ground_truth in enumerate(ground_truth_questions):
            transcript = ground_truth.get("transcript", {}).get("en", "")
            gt_mcqs = ground_truth.get("multiple-choice", [])

            generated = generated_questions[idx] if idx < len(generated_questions) else None
            if not generated:
                continue

            gen_mcqs = generated.get("multiple-choice", [])

            for q_idx, gt_q in enumerate(gt_mcqs):
                print(f"Matching Ground Truth Set {idx + 1}, Question {q_idx + 1} on page {page_number}")
                matching_prompt = create_matching_prompt(transcript, gt_q, gen_mcqs)
                match_response_text = send_request_with_retries(matching_prompt)

                if match_response_text:
                    log_file.write(f"Match Phase - Ground Truth Set {idx + 1}, Question {q_idx + 1}\n")
                    log_file.write(f"Matching Prompt:\n{matching_prompt}\n")
                    log_file.write(f"Matching Response:\n{match_response_text}\n\n")

                    # Check if there's a valid match before proceeding
                    match_search = re.search(r"(\d+)", match_response_text)
                    if match_search:
                        match_idx = int(match_search.group(1)) - 1
                        if 0 <= match_idx < len(gen_mcqs):
                            matches_data.append({
                                "ground_truth_index": q_idx,
                                "generated_index": match_idx,
                                "page": page_number,
                                "video_number": idx + 1
                            })
                        else:
                            print(f"Warning: Match index out of range for Ground Truth Set {idx + 1}, Question {q_idx + 1} on page {page_number}")
                            log_file.write(f"Warning: Match index out of range for Ground Truth Set {idx + 1}, Question {q_idx + 1}\n")
                    else:
                        print(f"Warning: No valid match index found in response for Ground Truth Set {idx + 1}, Question {q_idx + 1} on page {page_number}")
                        log_file.write(f"Warning: No valid match index found in response for Ground Truth Set {idx + 1}, Question {q_idx + 1}\n")
                else:
                    print(f"Warning: No response text received for matching Ground Truth Set {idx + 1}, Question {q_idx + 1} on page {page_number}")
                    log_file.write(f"Warning: No response text received for matching Ground Truth Set {idx + 1}, Question {q_idx + 1}\n")

        for match in matches_data:
            gt_idx = match["ground_truth_index"]
            gen_idx = match["generated_index"]
            page = match["page"]
            video = match["video_number"]

            ground_truth = ground_truth_questions[video - 1]
            generated = generated_questions[video - 1]
            
            transcript = ground_truth.get("transcript", {}).get("en", "")
            gt_q = ground_truth.get("multiple-choice", [])[gt_idx]
            gen_q = generated.get("multiple-choice", [])[gen_idx]

            print(f"Scoring Ground Truth Set {video}, Ground Truth Question {gt_idx + 1} with Generated Question {gen_idx + 1}")
            evaluation_prompt = create_evaluation_prompt(transcript, gt_q, gen_q)
            eval_response_text = send_request_with_retries(evaluation_prompt)

            if eval_response_text:
                log_file.write(f"Scoring Phase - Ground Truth Set {video}, Ground Truth Question {gt_idx + 1}, Matched Generated Question {gen_idx + 1}\n")
                log_file.write(f"Evaluation Prompt:\n{evaluation_prompt}\n")
                log_file.write(f"Evaluation Response:\n{eval_response_text}\n\n")

                parsed_scores = parse_scores(eval_response_text)

                scores_data.append({
                    "page": page,
                    "video_number": video,
                    "ground_truth_index": gt_idx + 1,
                    "generated_index": gen_idx + 1,
                    "relevance_score": parsed_scores.get("Relevance Score", 0.0),
                    "correct_answer_matching_score": parsed_scores.get("Correct Answer Matching Score", 0.0),
                    "distractor_plausibility_score": parsed_scores.get("Distractor Plausibility Score", 0.0),
                    "clarity_and_readability_score": parsed_scores.get("Clarity and Readability Score", 0.0),
                    "total_score": parsed_scores.get("Total Score", 0.0)
                })

    # Save results to JSONL format per page
    with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as jsonl_file:
        for score in scores_data:
            jsonl_file.write(json.dumps(score) + "\n")

    # Save results to CSV format per page
    with open(OUTPUT_CSV_FILE, "w", newline='', encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "page", "video_number", "ground_truth_index", "generated_index",
            "relevance_score", "correct_answer_matching_score",
            "distractor_plausibility_score", "clarity_and_readability_score", "total_score"
        ])
        for score in scores_data:
            csv_writer.writerow([
                score["page"], score["video_number"], score["ground_truth_index"], score["generated_index"],
                score["relevance_score"], score["correct_answer_matching_score"],
                score["distractor_plausibility_score"], score["clarity_and_readability_score"],
                score["total_score"]
            ])

    print(f"Page {page_number} - Matching and scoring responses have been saved to {LOG_FILE}")
    print(f"Page {page_number} - Scores have been saved to {OUTPUT_JSON_FILE} and {OUTPUT_CSV_FILE}")
