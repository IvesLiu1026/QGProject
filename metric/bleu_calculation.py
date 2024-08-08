import json
import csv
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter
from fractions import Fraction
import math
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

def modified_precision(reference, hypothesis, n):
    counts = Counter(tuple(hypothesis[i:i+n]) for i in range(len(hypothesis)-n+1))
    max_counts = Counter()
    for ref in reference:
        ref_counts = Counter(tuple(ref[i:i+n]) for i in range(len(ref)-n+1))
        for ngram in counts:
            max_counts[ngram] = max(max_counts[ngram], ref_counts[ngram])
    numerator = sum((min(count, max_counts[ngram]) for ngram, count in counts.items()))
    denominator = max(1, sum(counts.values()))
    return Fraction(numerator, denominator)

def brevity_penalty(reference, hypothesis):
    ref_length = min(len(ref) for ref in reference)
    hyp_length = len(hypothesis)
    if hyp_length > ref_length:
        return 1
    else:
        return math.exp(1 - ref_length / hyp_length)

def calculate_bleu(reference, hypothesis):
    weights = [0.25, 0.25, 0.25, 0.25]
    p_n = [modified_precision(reference, hypothesis, i+1) for i in range(4)]
    s = [w * math.log(p_n[i].numerator / p_n[i].denominator) if p_n[i].denominator != 0 and p_n[i].numerator != 0 else 0 for i, w in enumerate(weights)]
    bp = brevity_penalty(reference, hypothesis)
    return bp * math.exp(sum(s))

def evaluate_option_pair(original_options, generated_option):
    return max(calculate_bleu([orig_opt], generated_option) for orig_opt in original_options)

def find_best_matching_question(original_questions, generated_question):
    best_question_score = 0
    best_question = None

    for original_question in original_questions:
        question_score = calculate_bleu(original_question['question'], generated_question['question'])
        if question_score > best_question_score:
            best_question_score = question_score
            best_question = original_question

    return best_question, best_question_score

def evaluate_question_pair(original_questions, generated_question):
    best_question, best_question_score = find_best_matching_question(original_questions, generated_question)

    if best_question is None:
        return None

    options_bleu = [evaluate_option_pair(best_question['options'], gen_opt) for gen_opt in generated_question['options']]

    return {
        'question_bleu': best_question_score,
        'options_bleu': options_bleu
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

            question_bleu_scores = [q['question_bleu'] for q in multiple_choice]
            option_bleu_scores = [opt for q in multiple_choice for opt in q['options_bleu']]

            avg_question_bleu = sum(question_bleu_scores) / len(question_bleu_scores) if question_bleu_scores else 0
            avg_option_bleu = sum(option_bleu_scores) / len(option_bleu_scores) if option_bleu_scores else 0

            results.append({
                'title': title,
                'category': category,
                'questions-average-bleu': avg_question_bleu,
                'options-average-bleu': avg_option_bleu,
                'multiple-choice': multiple_choice
            })
    
    return results

def save_to_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def save_to_csv(data, file_path):
    with open(file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Title', 'Category', 'Questions Average BLEU', 'Options Average BLEU'])
        for item in data:
            writer.writerow([item['title'], item['category'], item['questions-average-bleu'], item['options-average-bleu']])

def main():
    original_file_path = '../dataset/dataset.json'
    generated_file_path = '../results/0802/gemma2_9b/generate_self_consistency.json'
    output_json_file_path = '0802/gemma2_9b/bleu/sc_bleu_score.json'
    output_csv_file_path = '0802/gemma2_9b/bleu/sc_bleu_score.csv'
    
    original_data = load_json(original_file_path)
    generated_data = load_json(generated_file_path)
    
    original_questions = extract_questions(original_data)
    generated_questions = extract_questions(generated_data)
    
    evaluated_data = evaluate_questions(original_questions, generated_questions)
    
    save_to_json(evaluated_data, output_json_file_path)
    save_to_csv(evaluated_data, output_csv_file_path)
    print(f"BLEU evaluation results saved to {output_json_file_path} and {output_csv_file_path}")

if __name__ == "__main__":
    main()
