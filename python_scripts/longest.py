import ijson

# Define the input file path
input_file_path = 'dataset.json'  # Update this path with your actual JSON file path

# Initialize variables to track the longest transcript
longest_transcript = ""
longest_length = 0

# Open the JSON file and initialize the ijson parser
with open(input_file_path, 'r', encoding='utf-8') as json_file:
    objects = ijson.items(json_file, 'item')

    # Iterate over objects in the JSON file
    for obj in objects:
        transcript = obj.get('transcript', '')
        if len(transcript) > longest_length:
            longest_transcript = transcript
            longest_length = len(transcript)

# Output the longest transcript
print(f'The longest transcript is:\n{longest_transcript}')
print(f'\nLength of the longest transcript: {longest_length}')
