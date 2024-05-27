import ijson

# Define the input and output file paths
input_file_path = '../dataset/raw_dataset.json'  # Update this path with your actual JSON file path
output_file_path = '../dataset/data_info/pages_videos_report.txt'  # Update this path with your desired output file path
missing_pages_output_path = '../dataset/data_info/missing_pages_report.txt'  # Update this path with your desired output file path

# Initialize a dictionary to count videos per page
page_video_count = {}
page_lessons = {}

# Open the JSON file and initialize the ijson parser
with open(input_file_path, 'r', encoding='utf-8') as json_file:
    objects = ijson.items(json_file, 'item')

    # Iterate over objects in the JSON file
    for obj in objects:
        page = obj.get('page')
        lesson = obj.get('lesson')
        if page:
            page_str = int(page)
            if page_str in page_video_count:
                page_video_count[page_str] += 1
                page_lessons[page_str].append(lesson)
            else:
                page_video_count[page_str] = 1
                page_lessons[page_str] = [lesson]

# Define the expected number of videos per page
def expected_videos(page):
    return 5 if page == 122 else 24

# Write the video counts per page and missing lessons to a file
with open(output_file_path, 'w', encoding='utf-8') as file:
    for page in range(1, 123):
        page_str = int(page)
        if page_str not in page_video_count:
            file.write(f'page {page}: missing\n')
        else:
            actual_videos = page_video_count[page_str]
            expected_video_count = expected_videos(page)
            if actual_videos == expected_video_count:
                file.write(f'page {page}: {actual_videos} no missing video\n')
            else:
                missing_videos = expected_video_count - actual_videos
                file.write(f'page {page}: {actual_videos} missing videos: {missing_videos}\n')
                missing_lessons = set(range(1, expected_video_count + 1)) - set(page_lessons[page_str])
                file.write(f'missing lessons: {missing_lessons}\n')

print(f'Pages and video counts have been saved to {output_file_path}')