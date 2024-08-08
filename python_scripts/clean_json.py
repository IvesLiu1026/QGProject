import ijson
import json
from decimal import Decimal

# Define the input and output file paths
input_file_path = 'D:\\NYCU1122courses\\Project\\ollama\\metric\\gemma7b-v1.json'  # Update this path with your actual JSON file path
output_file_path = 'D:\\NYCU1122courses\\Project\\ollama\\metric\\gemma7b-v1-cleaned.json'  # Update this path with your desired output JSON file path

# Function to convert Decimal objects to float
def convert_decimal(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError

# Initialize a list to hold the cleaned data
cleaned_data = []

# Open the JSON file and initialize the ijson parser
with open(input_file_path, 'r', encoding='utf-8') as json_file:
    objects = ijson.items(json_file, 'item')

    # Iterate over objects in the JSON file
    for obj in objects:
        multiple_choice = obj.get('multiple-choice', None)
        
        # Normalize multiple-choice field
        if isinstance(multiple_choice, str):
            multiple_choice = multiple_choice.strip()
            if multiple_choice in ["[]", ""]:
                multiple_choice = []
        
        # Update the object with the cleaned multiple_choice field
        obj['multiple-choice'] = multiple_choice
        
        # Filter out items where multiple_choice is an empty list
        if multiple_choice:
            cleaned_data.append(obj)

# Write the cleaned data to a new JSON file
with open(output_file_path, 'w', encoding='utf-8') as json_output_file:
    json.dump(cleaned_data, json_output_file, ensure_ascii=False, indent=4, default=convert_decimal)

print(f'Cleaned JSON file has been saved to {output_file_path}')
