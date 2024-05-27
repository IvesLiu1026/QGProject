import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter
from fractions import Fraction
import math
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_json(file_path):
    with open(file_path, 'r') as file:
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
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {key: float(value.fmeasure) for key, value in scores.items()}

def calculate_bertscore(reference, candidate):
    P, R, F1 = bert_score([candidate], [reference], model_type='bert-base-uncased', verbose=False)
    return float(F1.item())

def calculate_cosine_similarity(reference, candidate, model):
    reference_embedding = model.encode(reference)
    candidate_embedding = model.encode(candidate)
    return float(cosine_similarity([reference_embedding], [candidate_embedding])[0][0])

def evaluate_questions(original_questions, generated_questions):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    results = []

    for title, original_data in original_questions.items():
        if title in generated_questions:
            generated_data = generated_questions[title]
            category = original_data['category']
            original_qs = original_data['questions']
            generated_qs = generated_data['questions']
            multiple_choice = []

            question_bleu_scores = []
            question_rouge_scores = []
            question_bertscores = []
            question_cosine_scores = []
            option_bleu_scores = []
            option_rouge_scores = []
            option_bertscores = []
            option_cosine_scores = []

            for original_q, generated_q in zip(original_qs, generated_qs):
                question_bleu = calculate_bleu(original_q['question'], generated_q['question'])
                question_rouge = calculate_rouge(original_q['question'], generated_q['question'])
                question_bertscore = calculate_bertscore(original_q['question'], generated_q['question'])
                question_cosine = calculate_cosine_similarity(original_q['question'], generated_q['question'], model)
                
                question_bleu_scores.append(question_bleu)
                question_rouge_scores.append(question_rouge['rougeL'])
                question_bertscores.append(question_bertscore)
                question_cosine_scores.append(question_cosine)

                options_results = []
                for orig_opt, gen_opt in zip(original_q['options'], generated_q['options']):
                    option_bleu = calculate_bleu(orig_opt, gen_opt)
                    option_rouge = calculate_rouge(orig_opt, gen_opt)
                    option_bertscore = calculate_bertscore(orig_opt, gen_opt)
                    option_cosine = calculate_cosine_similarity(orig_opt, gen_opt, model)
                    
                    option_bleu_scores.append(option_bleu)
                    option_rouge_scores.append(option_rouge['rougeL'])
                    option_bertscores.append(option_bertscore)
                    option_cosine_scores.append(option_cosine)

                    options_results.append({
                        'text': gen_opt,
                        'bleu': option_bleu,
                        'rouge': option_rouge,
                        'bert-score': option_bertscore,
                        'cosine-similarity': option_cosine
                    })

                multiple_choice.append({
                    'question': generated_q['question'],
                    'bleu': question_bleu,
                    'rouge': question_rouge,
                    'bert-score': question_bertscore,
                    'cosine-similarity': question_cosine,
                    'options': options_results
                })

            avg_question_bleu = sum(question_bleu_scores) / len(question_bleu_scores) if question_bleu_scores else 0
            avg_question_rouge = sum(question_rouge_scores) / len(question_rouge_scores) if question_rouge_scores else 0
            avg_question_bertscore = sum(question_bertscores) / len(question_bertscores) if question_bertscores else 0
            avg_question_cosine = sum(question_cosine_scores) / len(question_cosine_scores) if question_cosine_scores else 0

            avg_option_bleu = sum(option_bleu_scores) / len(option_bleu_scores) if option_bleu_scores else 0
            avg_option_rouge = sum(option_rouge_scores) / len(option_rouge_scores) if option_rouge_scores else 0
            avg_option_bertscore = sum(option_bertscores) / len(option_bertscores) if option_bertscores else 0
            avg_option_cosine = sum(option_cosine_scores) / len(option_cosine_scores) if option_cosine_scores else 0

            avg_bleu = (avg_question_bleu + avg_option_bleu) / 2
            avg_rouge = (avg_question_rouge + avg_option_rouge) / 2
            avg_bertscore = (avg_question_bertscore + avg_option_bertscore) / 2
            avg_cosine = (avg_question_cosine + avg_option_cosine) / 2

            results.append({
                'title': title,
                'category': category,
                'average-bleu': avg_bleu,
                'average-rouge': avg_rouge,
                'average-bert-score': avg_bertscore,
                'average-cosine-similarity': avg_cosine,
                'multiple-choice': multiple_choice
            })
    
    return results

def save_to_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def main():
    original_file_path = '../original_questions.json'
    generated_file_path = '../generated_questions.json'
    output_file_path = 'all_v2.json'
    
    original_data = load_json(original_file_path)
    generated_data = load_json(generated_file_path)
    
    original_questions = extract_questions(original_data)
    generated_questions = extract_questions(generated_data)
    
    evaluated_data = evaluate_questions(original_questions, generated_questions)
    
    save_to_json(evaluated_data, output_file_path)
    print(f"Evaluation results saved to {output_file_path}")

if __name__ == "__main__":
    main()