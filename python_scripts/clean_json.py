import ijson
import json

# Define the input and output file paths
input_file_path = 'dataset.json'  # Update this path with your actual JSON file path
output_file_path = 'cleaned_dataset_v2.json'  # Update this path with your desired output JSON file path

# Initialize a list to hold the cleaned data
cleaned_data = []

# Open the JSON file and initialize the ijson parser
with open(input_file_path, 'r', encoding='utf-8') as json_file:
    objects = ijson.items(json_file, 'item')

    # Iterate over objects in the JSON file
    for obj in objects:
        transcript = obj.get('transcript', '').strip()
        multiple_choice = obj.get('multiple-choice', None)
        
        # Normalize multiple-choice field
        if isinstance(multiple_choice, str):
            multiple_choice = multiple_choice.strip()
            if multiple_choice in ["[]", ""]:
                multiple_choice = []

        # Filter out objects based on the conditions
        if transcript != 'Transcript not available' and multiple_choice:
            cleaned_data.append(obj)

# Write the cleaned data to a new JSON file
with open(output_file_path, 'w', encoding='utf-8') as json_output_file:
    json.dump(cleaned_data, json_output_file, ensure_ascii=False, indent=4)

print(f'Cleaned JSON file has been saved to {output_file_path}')
