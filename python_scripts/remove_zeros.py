import pandas as pd

# Function to read CSV data from a file
def read_csv(file_path):
    return pd.read_csv(file_path)

# Function to write CSV data to a file
def write_csv(file_path, data):
    data.to_csv(file_path, index=False)

# Function to filter out rows where both specified columns are 0
def filter_data(df):
    return df[(df['Questions Average ROUGE-1'] != 0) | (df['Questions Average ROUGE-2'] != 0) | (df['Questions Average ROUGE-L'] != 0) | (df['Options Average ROUGE-1'] != 0) | (df['Options Average ROUGE-2'] != 0) | (df['Options Average ROUGE-L'] != 0)]
    # return df[(df['Questions Average BLEU'] != 0) | (df['Options Average BLEU'] != 0)]
# Main script
def main(input_file, output_file):
    # Read the CSV data from the input file
    df = read_csv(input_file)

    # Filter the data
    filtered_df = filter_data(df)

    # Write the filtered data to the output file
    write_csv(output_file, filtered_df)

    print(f"Filtered data has been written to {output_file}")

# Example usage
input_file = '../results/0808/score/rouge/simple_summary.csv'  # Path to your input CSV file
output_file = '../results/0808/score/rouge/simple_summary.csv'  # Path to your output CSV file

main(input_file, output_file)
