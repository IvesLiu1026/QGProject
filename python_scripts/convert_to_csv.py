import ijson
import csv

# Define the input and output file paths
input_file_path = '../dataset/data_info/missing_report.json'  # Update this path
output_file_path = '../dataset/data_info/missing_report.csv'  # Update this path

# Open the JSON file and initialize the ijson parser
with open(input_file_path, 'r', encoding='utf-8') as json_file:
    # 'item' is used to iterate over the objects in the JSON array
    objects = ijson.items(json_file, 'item')

    # Open the CSV file for writing
    with open(output_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = None
        
        # Iterate over objects in the JSON file
        for obj in objects:
            if writer is None:
                # Initialize CSV writer and write header
                writer = csv.DictWriter(csv_file, fieldnames=obj.keys())
                writer.writeheader()
            
            # Write row to CSV
            writer.writerow(obj)
