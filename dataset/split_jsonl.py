import json
from collections import defaultdict

# Input file path (replace 'dataset.jsonl' with your actual file path)
input_file = 'dataset.jsonl'

# Dictionary to store entries by page number
page_groups = defaultdict(list)

# Read the original JSONL file and group by page number
with open(input_file, 'r', encoding='utf-8') as file:
    for line in file:
        entry = json.loads(line)
        page = entry.get('page')  # Assumes 'page' is a key in each JSON entry
        if page is not None:
            page_groups[page].append(line)

# Write each page group to a separate file
for page, lines in page_groups.items():
    output_file = f'split_dataset/dataset_page_{page}.jsonl'
    with open(output_file, 'w', encoding='utf-8') as output:
        output.writelines(lines)
    print(f"Created {output_file} with {len(lines)} entries")
