import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter
from fractions import Fraction
import math
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
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

def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {key: float(value.fmeasure) for key, value in scores.items()}

def calculate_bertscore(reference, candidate):
    P, R, F1 = bert_score([candidate], [reference], model_type='bert-base-uncased', verbose=False)
    return float(F1.item())

def calculate_cosine_similarity(reference, candidate, model):
    reference_embedding = model.encode(reference)
    candidate_embedding = model.encode(candidate)
    return float(cosine_similarity([reference_embedding], [candidate_embedding])[0][0])

def evaluate_option_pair(original_options, generated_option, model):
    max_scores = {
        'bleu': 0,
        'rouge1': 0,
        'rouge2': 0,
        'rougeL': 0,
        'bert-score': 0,
        'cosine-similarity': 0
    }
    
    for orig_opt in original_options:
        max_scores['bleu'] = max(max_scores['bleu'], calculate_bleu([orig_opt], generated_option))
        rouge_scores = calculate_rouge(orig_opt, generated_option)
        max_scores['rouge1'] = max(max_scores['rouge1'], rouge_scores['rouge1'])
        max_scores['rouge2'] = max(max_scores['rouge2'], rouge_scores['rouge2'])
        max_scores['rougeL'] = max(max_scores['rougeL'], rouge_scores['rougeL'])
        max_scores['bert-score'] = max(max_scores['bert-score'], calculate_bertscore(orig_opt, generated_option))
        max_scores['cosine-similarity'] = max(max_scores['cosine-similarity'], calculate_cosine_similarity(orig_opt, generated_option, model))

    return max_scores

def evaluate_question_pair(original_q, generated_q, model):
    question_bleu = calculate_bleu(original_q['question'], generated_q['question'])
    question_rouge = calculate_rouge(original_q['question'], generated_q['question'])
    question_bertscore = calculate_bertscore(original_q['question'], generated_q['question'])
    question_cosine = calculate_cosine_similarity(original_q['question'], generated_q['question'], model)
    
    question_scores = {
        'bleu': question_bleu,
        'rouge1': question_rouge['rouge1'],
        'rouge2': question_rouge['rouge2'],
        'rougeL': question_rouge['rougeL'],
        'bert-score': question_bertscore,
        'cosine-similarity': question_cosine
    }

    options_results = []
    for gen_opt in generated_q['options']:
        option_scores = evaluate_option_pair(original_q['options'], gen_opt, model)
        options_results.append({
            'text': gen_opt,
            **option_scores
        })

    question_scores['options'] = options_results
    return question_scores

def evaluate_questions(original_questions, generated_questions):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    results = []

    for title, original_data in tqdm(original_questions.items(), desc="Processing titles"):
        if title in generated_questions:
            generated_data = generated_questions[title]
            category = original_data['category']
            original_qs = original_data['questions']
            generated_qs = generated_data['questions']
            multiple_choice = []

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(evaluate_question_pair, original_q, generated_q, model) for original_q, generated_q in zip(original_qs, generated_qs)]
                for future in tqdm(as_completed(futures), desc="Processing questions", leave=False, total=len(futures)):
                    question_scores = future.result()
                    multiple_choice.append(question_scores)

            question_bleu_scores = [q['bleu'] for q in multiple_choice]
            question_rouge1_scores = [q['rouge1'] for q in multiple_choice]
            question_rouge2_scores = [q['rouge2'] for q in multiple_choice]
            question_rougeL_scores = [q['rougeL'] for q in multiple_choice]
            question_bertscores = [q['bert-score'] for q in multiple_choice]
            question_cosine_scores = [q['cosine-similarity'] for q in multiple_choice]

            avg_question_bleu = sum(question_bleu_scores) / len(question_bleu_scores) if question_bleu_scores else 0
            avg_question_rouge1 = sum(question_rouge1_scores) / len(question_rouge1_scores) if question_rouge1_scores else 0
            avg_question_rouge2 = sum(question_rouge2_scores) / len(question_rouge2_scores) if question_rouge2_scores else 0
            avg_question_rougeL = sum(question_rougeL_scores) / len(question_rougeL_scores) if question_rougeL_scores else 0
            avg_question_bertscore = sum(question_bertscores) / len(question_bertscores) if question_bertscores else 0
            avg_question_cosine = sum(question_cosine_scores) / len(question_cosine_scores) if question_cosine_scores else 0

            option_bleu_scores = [opt['bleu'] for q in multiple_choice for opt in q['options']]
            option_rouge1_scores = [opt['rouge1'] for q in multiple_choice for opt in q['options']]
            option_rouge2_scores = [opt['rouge2'] for q in multiple_choice for opt in q['options']]
            option_rougeL_scores = [opt['rougeL'] for q in multiple_choice for opt in q['options']]
            option_bertscores = [opt['bert-score'] for q in multiple_choice for opt in q['options']]
            option_cosine_scores = [opt['cosine-similarity'] for q in multiple_choice for opt in q['options']]

            avg_option_bleu = sum(option_bleu_scores) / len(option_bleu_scores) if option_bleu_scores else 0
            avg_option_rouge1 = sum(option_rouge1_scores) / len(option_rouge1_scores) if option_rouge1_scores else 0
            avg_option_rouge2 = sum(option_rouge2_scores) / len(option_rouge2_scores) if option_rouge2_scores else 0
            avg_option_rougeL = sum(option_rougeL_scores) / len(option_rougeL_scores) if option_rougeL_scores else 0
            avg_option_bertscore = sum(option_bertscores) / len(option_bertscores) if option_bertscores else 0
            avg_option_cosine = sum(option_cosine_scores) / len(option_cosine_scores) if option_cosine_scores else 0

            avg_bleu = (avg_question_bleu + avg_option_bleu) / 2
            avg_rouge1 = (avg_question_rouge1 + avg_option_rouge1) / 2
            avg_rouge2 = (avg_question_rouge2 + avg_option_rouge2) / 2
            avg_rougeL = (avg_question_rougeL + avg_option_rougeL) / 2
            avg_bertscore = (avg_question_bertscore + avg_option_bertscore) / 2
            avg_cosine = (avg_question_cosine + avg_option_cosine) / 2

            results.append({
                'title': title,
                'category': category,
                'average-bleu': avg_bleu,
                'average-rouge1': avg_rouge1,
                'average-rouge2': avg_rouge2,
                'average-rougeL': avg_rougeL,
                'average-bert-score': avg_bertscore,
                'average-cosine-similarity': avg_cosine,
                'multiple-choice': multiple_choice
            })
    
    return results

def save_to_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def main():
    original_file_path = '../dataset/dataset.json'
    generated_file_path = '../results/dolphin-llama3-generated.json'
    output_file_path = 'llama3-v2.json'
    
    original_data = load_json(original_file_path)[0:3]
    generated_data = load_json(generated_file_path)[0:3]
    
    original_questions = extract_questions(original_data)
    generated_questions = extract_questions(generated_data)
    
    evaluated_data = evaluate_questions(original_questions, generated_questions)
    
    save_to_json(evaluated_data, output_file_path)
    print(f"Evaluation results saved to {output_file_path}")

if __name__ == "__main__":
    main()
