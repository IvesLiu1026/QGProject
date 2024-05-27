import pandas as pd

# Define the input and output file paths
input_file_path = '../dataset/dataset.csv'  # Update with your actual CSV file path
output_file_path = 'category_sorted.csv'  # Update with your desired output CSV file path
count = 'category_counts.txt'  # Update with your desired output file path

# Read the CSV file into a DataFrame
df = pd.read_csv(input_file_path)

# Sort the DataFrame by the 'category' field
df_sorted = df.sort_values(by='category')

# Save the sorted DataFrame to a new CSV file
df_sorted.to_csv(output_file_path, index=False)

# Calculate the number of each category
category_counts = df['category'].value_counts()

# Calculate the total number of items
total_count = category_counts.sum()

# Write the number of each category and the total count to the text file
with open(count, 'w') as file:
    file.write(category_counts.to_string())
    file.write(f'\n\nTotal number of items: {total_count}')



print(f'Sorted CSV file has been saved to {output_file_path}')
