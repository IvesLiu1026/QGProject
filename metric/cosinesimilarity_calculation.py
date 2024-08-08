import json
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

def calculate_cosine_similarity(reference, candidate, model):
    reference_embedding = model.encode(reference)
    candidate_embedding = model.encode(candidate)
    return float(cosine_similarity([reference_embedding], [candidate_embedding])[0][0])

def evaluate_option_pair(original_options, generated_option, model):
    return max(calculate_cosine_similarity(orig_opt, generated_option, model) for orig_opt in original_options)

def find_best_matching_question(original_questions, generated_question, model):
    best_question_score = 0
    best_question = None

    for original_question in original_questions:
        question_score = calculate_cosine_similarity(original_question['question'], generated_question['question'], model)
        if question_score > best_question_score:
            best_question_score = question_score
            best_question = original_question

    return best_question, best_question_score

def evaluate_question_pair(original_questions, generated_question, model):
    best_question, best_question_score = find_best_matching_question(original_questions, generated_question, model)

    if best_question is None:
        return None

    options_cosine = [evaluate_option_pair(best_question['options'], gen_opt, model) for gen_opt in generated_question['options']]

    return {
        'question_cosine': best_question_score,
        'options_cosine': options_cosine
    }

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
                futures = [executor.submit(evaluate_question_pair, original_qs, generated_q, model) for generated_q in generated_qs]
                for future in tqdm(as_completed(futures), desc="Processing questions", leave=False, total=len(futures)):
                    question_scores = future.result()
                    if question_scores:
                        multiple_choice.append(question_scores)

            question_cosine_scores = [q['question_cosine'] for q in multiple_choice]
            option_cosine_scores = [opt for q in multiple_choice for opt in q['options_cosine']]

            avg_question_cosine = sum(question_cosine_scores) / len(question_cosine_scores) if question_cosine_scores else 0
            avg_option_cosine = sum(option_cosine_scores) / len(option_cosine_scores) if option_cosine_scores else 0
            avg_cosine = (avg_question_cosine + avg_option_cosine) / 2

            results.append({
                'title': title,
                'category': category,
                'average-cosine-similarity': avg_cosine,
                'questions-average-cosine-similarity': avg_question_cosine,
                'options-average-cosine-similarity': avg_option_cosine,
                'multiple-choice': multiple_choice
            })
    
    return results

def save_to_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def main():
    original_file_path = '../dataset/dataset.json'
    generated_file_path = '../generate10.json'
    output_file_path = 'cosine_similarity_score.json'
    
    original_data = load_json(original_file_path)
    generated_data = load_json(generated_file_path)
    
    original_questions = extract_questions(original_data)
    generated_questions = extract_questions(generated_data)
    
    evaluated_data = evaluate_questions(original_questions, generated_questions)
    
    save_to_json(evaluated_data, output_file_path)
    print(f"Cosine Similarity evaluation results saved to {output_file_path}")

if __name__ == "__main__":
    main()
