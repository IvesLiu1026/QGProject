import json
from rouge_score import rouge_scorer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def extract_questions(data):
    questions_dict = {}
    for item in data:
        title = item['title']
        category = item.get('category', '')
        questions = []
        for q in item['multiple-choice']:
            question = q['question']
            options = [option['text'] for option in q['options']]
            questions.append({'question': question, 'options': options})
        questions_dict[title] = {'category': category, 'questions': questions}
    return questions_dict

def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {key: float(value.fmeasure) for key, value in scores.items()}

def evaluate_option_pair(original_options, generated_option):
    max_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    
    for orig_opt in original_options:
        rouge_scores = calculate_rouge(orig_opt, generated_option)
        max_scores['rouge1'] = max(max_scores['rouge1'], rouge_scores['rouge1'])
        max_scores['rouge2'] = max(max_scores['rouge2'], rouge_scores['rouge2'])
        max_scores['rougeL'] = max(max_scores['rougeL'], rouge_scores['rougeL'])

    return max_scores

def find_best_matching_question(original_questions, generated_question):
    best_question_score = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    best_question = None

    for original_question in original_questions:
        question_score = calculate_rouge(original_question['question'], generated_question['question'])
        if question_score['rouge1'] > best_question_score['rouge1']:
            best_question_score = question_score
            best_question = original_question

    return best_question, best_question_score

def evaluate_question_pair(original_questions, generated_question):
    best_question, best_question_score = find_best_matching_question(original_questions, generated_question)

    if best_question is None:
        return None

    options_rouge = [evaluate_option_pair(best_question['options'], gen_opt) for gen_opt in generated_question['options']]

    return {
        'question_rouge': best_question_score,
        'options_rouge': options_rouge
    }

def evaluate_questions(original_questions, generated_questions):
    results = []

    for title, original_data in tqdm(original_questions.items(), desc="Processing titles"):
        if title in generated_questions:
            generated_data = generated_questions[title]
            category = original_data['category']
            original_qs = original_data['questions']
            generated_qs = generated_data['questions']
            multiple_choice = []

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(evaluate_question_pair, original_qs, generated_q) for generated_q in generated_qs]
                for future in tqdm(as_completed(futures), desc="Processing questions", leave=False, total=len(futures)):
                    question_scores = future.result()
                    if question_scores:
                        multiple_choice.append(question_scores)

            question_rouge1_scores = [q['question_rouge']['rouge1'] for q in multiple_choice]
            question_rouge2_scores = [q['question_rouge']['rouge2'] for q in multiple_choice]
            question_rougeL_scores = [q['question_rouge']['rougeL'] for q in multiple_choice]
            option_rouge1_scores = [opt['rouge1'] for q in multiple_choice for opt in q['options_rouge']]
            option_rouge2_scores = [opt['rouge2'] for q in multiple_choice for opt in q['options_rouge']]
            option_rougeL_scores = [opt['rougeL'] for q in multiple_choice for opt in q['options_rouge']]

            avg_question_rouge1 = sum(question_rouge1_scores) / len(question_rouge1_scores) if question_rouge1_scores else 0
            avg_question_rouge2 = sum(question_rouge2_scores) / len(question_rouge2_scores) if question_rouge2_scores else 0
            avg_question_rougeL = sum(question_rougeL_scores) / len(question_rougeL_scores) if question_rougeL_scores else 0
            avg_option_rouge1 = sum(option_rouge1_scores) / len(option_rouge1_scores) if option_rouge1_scores else 0
            avg_option_rouge2 = sum(option_rouge2_scores) / len(option_rouge2_scores) if option_rouge2_scores else 0
            avg_option_rougeL = sum(option_rougeL_scores) / len(option_rougeL_scores) if option_rougeL_scores else 0

            results.append({
                'title': title,
                'category': category,
                'questions-average-rouge1': avg_question_rouge1,
                'questions-average-rouge2': avg_question_rouge2,
                'questions-average-rougeL': avg_question_rougeL,
                'options-average-rouge1': avg_option_rouge1,
                'options-average-rouge2': avg_option_rouge2,
                'options-average-rougeL': avg_option_rougeL,
                'multiple-choice': multiple_choice
            })
    
    return results

def save_to_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def main():
    original_file_path = '../dataset/dataset.json'
    generated_file_path = '../results/dolphin-llama3-generated.json'
    output_file_path = 'rouge_score.json'
    
    original_data = load_json(original_file_path)
    generated_data = load_json(generated_file_path)
    
    original_questions = extract_questions(original_data)
    generated_questions = extract_questions(generated_data)
    
    evaluated_data = evaluate_questions(original_questions, generated_questions)
    
    save_to_json(evaluated_data, output_file_path)
    print(f"ROUGE evaluation results saved to {output_file_path}")

if __name__ == "__main__":
    main()
