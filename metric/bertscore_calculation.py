import json
from bert_score import score as bert_score
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

def calculate_bertscore(reference, candidate):
    P, R, F1 = bert_score([candidate], [reference], model_type='bert-base-uncased', verbose=False)
    return float(F1.item())

def evaluate_option_pair(original_options, generated_option):
    return max(calculate_bertscore(orig_opt, generated_option) for orig_opt in original_options)

def find_best_matching_question(original_questions, generated_question):
    best_question_score = 0
    best_question = None

    for original_question in original_questions:
        question_score = calculate_bertscore(original_question['question'], generated_question['question'])
        if question_score > best_question_score:
            best_question_score = question_score
            best_question = original_question

    return best_question, best_question_score

def evaluate_question_pair(original_questions, generated_question):
    best_question, best_question_score = find_best_matching_question(original_questions, generated_question)

    if best_question is None:
        return None

    options_bertscore = [evaluate_option_pair(best_question['options'], gen_opt) for gen_opt in generated_question['options']]

    return {
        'question_bertscore': best_question_score,
        'options_bertscore': options_bertscore
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

            question_bertscores = [q['question_bertscore'] for q in multiple_choice]
            option_bertscores = [opt for q in multiple_choice for opt in q['options_bertscore']]

            avg_question_bertscore = sum(question_bertscores) / len(question_bertscores) if question_bertscores else 0
            avg_option_bertscore = sum(option_bertscores) / len(option_bertscores) if option_bertscores else 0
            avg_bertscore = (avg_question_bertscore + avg_option_bertscore) / 2

            results.append({
                'title': title,
                'category': category,
                'average-bertscore': avg_bertscore,
                'questions-average-bertscore': avg_question_bertscore,
                'options-average-bertscore': avg_option_bertscore,
                'multiple-choice': multiple_choice
            })
    
    return results

def save_to_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def main():
    original_file_path = '../dataset/dataset.json'
    generated_file_path = '../results/dolphin-llama3-generated.json'
    output_file_path = 'bertscore_score.json'
    
    original_data = load_json(original_file_path)
    generated_data = load_json(generated_file_path)
    
    original_questions = extract_questions(original_data)
    generated_questions = extract_questions(generated_data)
    
    evaluated_data = evaluate_questions(original_questions, generated_questions)
    
    save_to_json(evaluated_data, output_file_path)
    print(f"BERTScore evaluation results saved to {output_file_path}")

if __name__ == "__main__":
    main()
