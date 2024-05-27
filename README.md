
# Question Generation from Video Transcripts

This project uses the `ChatOllama` model to generate multiple-choice questions from video transcripts. The questions are intended to help understand the content better by focusing on key facts, definitions, and important details mentioned in the transcripts.

## Prerequisites

Ensure you have the following installed:

- Python 3.6 or later
- Pip (Python package installer)

## Installation

First, clone the repository and navigate to the project directory:

```sh
git clone https://github.com/IvesLiu1026/QGProject.git
cd <repository-directory>
```

Then, install the required Python packages using `requirements.txt`:

```sh
pip install -r requirements.txt
```

## Requirements

The `requirements.txt` file should include:

```plaintext
tqdm
langchain-community
langchain-core
ijson
```

## Usage

1. **Dataset Preparation**: Ensure your dataset is in JSON format with the required fields (`title`, `category`, `transcript`). The file should be located at the path specified in the script.

2. **Running the Script**: Execute the `main.py` script to generate questions.

```sh
python main.py
```

This will process the dataset and generate a JSON file with the multiple-choice questions.

## File Structure

- `main.py`: The main script to run the question generation.
- `dataset/`: Directory containing the input JSON dataset.
- `generated_questions.json`: The output file containing the generated multiple-choice questions.
