### Highlight Key Points and Ask Questions

```json
{
  "task": "highlight_key_points",
  "input": {
    "title": "Why is rice so popular? - Carolyn Beans",
    "category": "Science & Technology",
    "transcript_url": "https://ed.ted.com/lessons/why-is-rice-so-popular-carolyn-beans"
  }
}
```


### Iterative Refinement


Step 1: Generate Initial Questions
```json
{
  "task": "generate_initial_questions",
  "input": {
    "context": "<insert_context_here>"
  }
}
```


Step 2: Refine Questions
```json
{
  "task": "refine_questions",
  "input": {
    "initial_questions": "<insert_initial_questions_here>",
    "feedback": "Ensure questions are clear, specific, and cover key points."
  }
}
```


### Summarization Before Question Generation

```json
{
  "task": "summarize",
  "input": {
    "title": "Why is rice so popular? - Carolyn Beans",
    "category": "Science & Technology",
    "transcript_url": "https://ed.ted.com/lessons/why-is-rice-so-popular-carolyn-beans"
  }
}
```
