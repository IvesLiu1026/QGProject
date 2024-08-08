**Chain-of-Thought Prompting**

```plaintext
  You are an expert educational content creator. Based on the following summary and its category, follow these steps to generate five multiple-choice questions:

  1. Identify the main themes in the summary.
  2. Create a question for each theme, ensuring it targets specific facts or details.
  3. Provide four plausible answer choices for each question, with one correct answer.

  Provide the questions and answers in the following format:
  1) [Your question here]
      - A: [Option A]
      - B: [Option B]
      - C: [Option C]
      - D: [Option D]
  [Your answer here]: [Correct option]

  Category: {category}

  Summary:
  {summary}
```

**few-shot**
  
```plaintext
You are an expert educational content creator. Based on the following examples and the provided summary, generate five multiple-choice questions.

Example 1:
1) What is the main theme of the text?
    - A: Education
    - B: Technology
    - C: Healthcare
    - D: Environment
Correct Answer: B

Example 2:
1) Which factor is highlighted as a key challenge?
    - A: Economic instability
    - B: Political unrest
    - C: Technological advancements
    - D: Social inequality
Correct Answer: A

Provide the questions and answers in the following format:
1) [Your question here]
    - A: [Option A]
    - B: [Option B]
    - C: [Option C]
    - D: [Option D]
[Your answer here]: [Correct option]

Category: {category}

Summary:
{summary}
```

**generated-knowledge**
```plaintext
You are an expert educational content creator. Based on the following summary and additional context, generate five multiple-choice questions.

Additional Context:
The main causes of World War II include territorial disputes, economic instability, the rise of totalitarian regimes, and the failure of international diplomacy.

Provide the questions and answers in the following format:
1) [Your question here]
    - A: [Option A]
    - B: [Option B]
    - C: [Option C]
    - D: [Option D]
[Your answer here]: [Correct option]

Category: {category}

Summary:
{summary}
```

**self-consistency**
```plaintext
You are an expert educational content creator. Based on the following summary and previous questions, generate five multiple-choice questions that are consistent with the given information.

Previous Questions:
1) What is the main theme of the text?
    - A: Technology
    - B: Environment
    - C: Healthcare
    - D: Education
Correct Answer: A

Provide the questions and answers in the following format:
1) [Your question here]
    - A: [Option A]
    - B: [Option B]
    - C: [Option C]
    - D: [Option D]
[Your answer here]: [Correct option]

Category: {category}

Summary:
{summary}
```


**zero-shot**
```plaintext
You are an expert educational content creator. Based on the following summary, generate five multiple-choice questions that cover key facts and details. Ensure each question has four answer choices and one correct answer.

Provide the questions and answers in the following format:
1) [Your question here]
    - A: [Option A]
    - B: [Option B]
    - C: [Option C]
    - D: [Option D]
[Your answer here]: [Correct option]

Category: {category}

Summary:
{summary}
```


