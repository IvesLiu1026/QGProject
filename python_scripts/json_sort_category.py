import json

input_file_path = '../dataset/dataset.json'
output_file_path = '../dataset/category_sorted_dataset.json'

with open(input_file_path, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

sorted_data = sorted(data, key=lambda x: x.get('category', '').strip())

with open(output_file_path, 'w', encoding='utf-8') as json_output_file:
    json.dump(sorted_data, json_output_file, ensure_ascii=False, indent=4)

print(f'Sorted JSON file has been saved to {output_file_path}')
