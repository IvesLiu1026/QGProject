import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

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

def calculate_cosine_similarity(reference, candidate, model):
    reference_embedding = model.encode(reference)
    candidate_embedding = model.encode(candidate)
    return cosine_similarity([reference_embedding], [candidate_embedding])[0][0]

def evaluate_questions_cosine(original_questions, generated_questions):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    scores = {}
    for title in original_questions.keys():
        if title in generated_questions:
            original_qs = original_questions[title]
            generated_qs = generated_questions[title]
            title_scores = []
            for i in range(min(len(original_qs), len(generated_qs))):
                original_question = original_qs[i]['question']
                generated_question = generated_qs[i]['question']
                question_score = calculate_cosine_similarity(original_question, generated_question, model)
                
                original_options = [opt for opt in original_qs[i]['options']]
                generated_options = [opt for opt in generated_qs[i]['options']]
                options_score = []
                for j in range(min(len(original_options), len(generated_options))):
                    options_score.append(calculate_cosine_similarity(original_options[j], generated_options[j], model))
                
                title_scores.append({
                    'question_score': question_score,
                    'options_scores': options_score
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
    
    scores = evaluate_questions_cosine(original_questions, generated_questions)
    
    for title, title_scores in scores.items():
        print(f"Title: {title}")
        for i, score in enumerate(title_scores):
            print(f"  Question {i+1} - Cosine Similarity: {score['question_score']:.4f}")
            for j, option_score in enumerate(score['options_scores']):
                print(f"    Option {j+1} - Cosine Similarity: {option_score:.4f}")
        print()

if __name__ == "__main__":
    main()
