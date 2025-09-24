import json
import os
import concurrent.futures
import time
import random
from functools import partial

from src.Llama_index_sandbox import YOUTUBE_VIDEO_DIRECTORY
from src.Llama_index_sandbox.utils.utils import root_directory, timeit


def format_time(ms):
    seconds, milliseconds = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


def process_utterance(utterance, sentence_count):
    output = []
    current_speaker = utterance['speaker']
    current_start = utterance['start']
    current_content = []
    current_sentence_count = 0

    for word in utterance['words']:
        current_content.append(word['text'])

        if "." in word['text']:
            current_sentence_count += 1

        if current_sentence_count == sentence_count:
            formatted_start = format_time(current_start)
            formatted_end = format_time(word['end'])
            segment = f"{formatted_start} - {formatted_end}, Speaker {current_speaker}: {' '.join(current_content)}"
            output.append(segment)

            # Reset for next segment
            current_content = []
            current_sentence_count = 0
            current_start = word['end']

    if current_content:
        formatted_start = format_time(current_start)
        formatted_end = format_time(utterance['end'])
        segment = f"{formatted_start} - {formatted_end}, Speaker {current_speaker}: {' '.join(current_content)}"
        output.append(segment)

    return output


def process_transcript(file_path, log, sentence_count=7):  # TODO 2023-10-05: the sentence count is a parameter to evauluate/optimise for
    SKIP_EXISTING = False  # Set to False if you want to re-process already processed files.
    try:
        # Save the results locally
        output_filename = os.path.splitext(os.path.basename(file_path))[0] + "_processed_diarized.txt"
        output_path = os.path.join(os.path.dirname(file_path), output_filename)
        # Check if the file already exists and SKIP_EXISTING is set to True
        if SKIP_EXISTING and os.path.exists(output_path):
            # print(f"Skipping {file_path.split('/')[-1]} as processed file already exists.")
            return

        if log:
            print(f"Processing: {file_path.split('/')[-1]}")

        with open(file_path, 'r') as f:
            file_content = f.read()
            if not file_content.strip():
                if log:
                    print(f"Empty JSON at {output_filename}. Returning...")
                # shutil.rmtree(os.path.dirname(file_path))
                return
            try:
                data = json.loads(file_content)
            except json.JSONDecodeError:
                if log:
                    print(f"Invalid JSON content in {output_filename}. Returning...")
                # shutil.rmtree(os.path.dirname(file_path))
                return

            if not data:
                if log:
                    print("no data!")
                return

        all_segments = []
        for utterance in data:
            all_segments.extend(process_utterance(utterance, sentence_count))

        # random time sleep
        time.sleep(random.randint(0, 2))
        with open(output_path, 'w') as output_file:
            for segment in all_segments:
                output_file.write(segment + '\n')
        if log:
            print(f"Saved {output_filename}")
    except Exception as e:
        if log:
            print(f"Error processing {output_filename}: {e}")


@timeit
def run(log=True):
    data_directory = YOUTUBE_VIDEO_DIRECTORY

    # First, map all directories and their files
    dir_files_map = {}
    for root, _, files in os.walk(data_directory):
        if files:  # Only process directories with files
            json_files = [f for f in files if f.endswith("_diarized_content.json")]
            txt_files = [f for f in files if f.endswith("_diarized_content_processed_diarized.txt")]

            if json_files:  # Only store if there are JSON files
                dir_files_map[root] = {
                    'json': json_files,
                    'txt': txt_files
                }

    # Now filter for files that need processing
    files_to_process = []
    for directory, file_info in dir_files_map.items():
        # Create a set of base names from txt files for faster lookup
        processed_bases = {
            txt_file.replace("_diarized_content_processed_diarized.txt", "")
            for txt_file in file_info['txt']
        }

        # Check each JSON file to see if it needs processing
        for json_file in file_info['json']:
            base_name = json_file.replace("_diarized_content.json", "")

            # Only process if the corresponding txt file doesn't exist
            if base_name not in processed_bases:
                files_to_process.append(os.path.join(directory, json_file))
                if log:
                    print(f"Will process: {os.path.join(directory, json_file)}")
            elif log:
                print(f"Skipping (already processed): {os.path.join(directory, json_file)}")

    if log:
        print(f"\nTotal files to process: {len(files_to_process)}")

    if files_to_process:
        # Create a partial function that includes the log flag
        process_with_log = partial(process_transcript, log=log)

        # Process the files in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(process_with_log, files_to_process)
    else:
        if log:
            print("No files need processing.")


if __name__ == "__main__":
    run()
