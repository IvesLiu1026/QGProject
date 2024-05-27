import json
from collections import Counter
from fractions import Fraction
import math
from nltk.translate.bleu_score import SmoothingFunction

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def extract_questions(data):
    questions_dict = {}
    for item in data:
        title = item['title']
        questions = []
        for q in item['multiple-choice']:
            question = q['question']
            options = [option['text'] for option in q['options']]
            questions.append({'question': question, 'options': options})
        questions_dict[title] = questions
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

def evaluate_questions(original_questions, generated_questions):
    scores = {}
    for title in original_questions.keys():
        if title in generated_questions:
            original_qs = original_questions[title]
            generated_qs = generated_questions[title]
            title_scores = []
            for i in range(min(len(original_qs), len(generated_qs))):
                original_question = original_qs[i]['question'].split()
                generated_question = generated_qs[i]['question'].split()
                question_score = calculate_bleu([original_question], generated_question)
                
                original_options = [opt.split() for opt in original_qs[i]['options']]
                generated_options = [opt.split() for opt in generated_qs[i]['options']]
                options_score = []
                for j in range(min(len(original_options), len(generated_options))):
                    options_score.append(calculate_bleu([original_options[j]], generated_options[j]))
                
                title_scores.append({
                    'question_score': float(question_score),
                    'options_scores': [float(score) for score in options_score]
                })
            scores[title] = title_scores
    return scores

def main():
    original_file_path = '../original_questions.json'
    generated_file_path = '../generated_questions.json'
    
    original_data = load_json(original_file_path)
    generated_data = load_json(generated_file_path)
    
    original_questions = extract_questions(original_data)
    generated_questions = extract_questions(generated_data)
    
    scores = evaluate_questions(original_questions, generated_questions)
    
    for title, title_scores in scores.items():
        print(f"Title: {title}")
        for i, score in enumerate(title_scores):
            print(f"  Question {i+1} - BLEU Score: {score['question_score']:.4f}")
            for j, option_score in enumerate(score['options_scores']):
                print(f"    Option {j+1} - BLEU Score: {option_score:.4f}")
        print()

if __name__ == "__main__":
    main()
