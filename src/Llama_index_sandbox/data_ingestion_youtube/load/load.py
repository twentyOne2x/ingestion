import concurrent.futures
import logging
import os
from functools import partial
from pathlib import Path
from typing import Union
import pandas as pd
import re
from llama_index.core.schema import Document

from src.Llama_index_sandbox import root_dir, YOUTUBE_VIDEO_DIRECTORY
from src.Llama_index_sandbox.constants import *
from src.Llama_index_sandbox.data_ingestion_youtube.load import create_transcripts_from_raw_json_utterances
from src.Llama_index_sandbox.data_ingestion_youtube.load.clean_transcripts_utterances import correct_typos_in_files
from src.Llama_index_sandbox.utils.utils import timeit, root_directory, start_logging, save_successful_load_to_csv, compute_new_entries, save_metadata_to_pipeline_dir
from src.utils.global_thread_guard import get_global_thread_limiter


def load_single_video_transcript(youtube_videos_df, file_path):
    try:
        # Extract video ID from the filename (now in position after date)
        filename = os.path.basename(file_path)
        # Format: YYYY-MM-DD_VIDEO_ID_Title_diarized_content_processed_diarized.txt
        parts = filename.split('_')

        # Video ID is at position 1 (after date)
        if len(parts) < 2:
            logging.error(f"Invalid filename format: {filename}")
            return [], {}

        video_id = parts[1]  # Extract the 11-character video ID

        # Match by video_id if column exists, otherwise fallback to URL matching
        if 'video_id' in youtube_videos_df.columns:
            video_row = youtube_videos_df[youtube_videos_df['video_id'] == video_id]
        else:
            # Fallback: try to match by URL containing the video ID
            video_row = youtube_videos_df[youtube_videos_df['url'].str.contains(video_id, na=False)]

        if video_row.empty:
            logging.info(f"Could not find video with ID {video_id} from file {filename}. Passing.")
            return [], {}

        # Safely access the first row if it exists
        video_data = video_row.iloc[0]

        from src.Llama_index_sandbox.custom_pymupdfreader.base import PyMuPDFReader
        loader = PyMuPDFReader()
        documents = loader.load(file_path=file_path)

        # Validation checks
        assert video_data['channel_name'] != video_data[
            'title'], f"Channel name and title are the same for {video_data['title']}"
        assert pd.notna(video_data['title']), f"Title is NaN for {video_data['url']}"
        assert pd.notna(video_data['channel_name']), f"channel_name is NaN for {video_data['url']}"
        assert pd.notna(video_data['url']), f"url is NaN for {video_data['url']}"
        assert pd.notna(video_data['published_date']), f"published_date is NaN for {video_data['url']}"

        # Update metadata for all documents
        for document in documents:
            if 'file_path' in document.metadata.keys():
                del document.metadata['file_path']

            # FIX: Check and clean the text content properly
            current_text = document.get_content()  # Get the current text content
            if '<|endoftext|>' in current_text:
                logging.error(f"Found <|endoftext|> in video ID {video_id} with {file_path}")
                # Create a new document with cleaned text
                cleaned_text = current_text.replace('<|endoftext|>', '')
                # Since we can't modify text directly, we need to recreate the document
                document = Document(
                    text=cleaned_text,
                    metadata=document.metadata,
                    id_=document.id_,
                    embedding=document.embedding
                )

            # Update metadata
            document.metadata.update({
                'document_type': DOCUMENT_TYPES.YOUTUBE_VIDEO.value,
                'video_id': video_id,  # Add video ID to metadata
                'title': video_data['title'],
                'channel_name': video_data['channel_name'],
                'published_date': video_data['published_date'],
                'url': video_data['url']
            })

        save_successful_load_to_csv(
            documents[0],
            csv_filename='youtube_videos.csv',
            fieldnames=['video_id', 'title', 'channel_name', 'published_date', 'url']
        )

        documents_details = {
            'document_type': DOCUMENT_TYPES.YOUTUBE_VIDEO.value,
            'video_id': video_id,
            'title': video_data['title'],
            'channel_name': video_data['channel_name'],
            'video_link': video_data['url'],
            'release_date': video_data['published_date']
        }
        return documents, documents_details

    except Exception as e:
        logging.error(f"Error: [{e}]: Failed to process {file_path}")
        return [], {}

@timeit
def load_video_transcripts(directory_path: Union[str, Path], add_new_transcripts=True, overwrite=False, num_files: int = None, files_window=None):
    root_dir = root_directory()
    # Convert directory_path to a Path object if it is not already
    if not isinstance(directory_path, Path):
        directory_path = Path(directory_path)

    if add_new_transcripts:
        logging.info("Creating transcripts from raw json utterances")
        create_transcripts_from_raw_json_utterances.run(log=True)
        correct_typos_in_files(log=False)
    else:
        logging.info("Skipping transcript creation from raw json utterances")

    all_documents = []
    all_metadata = []
    videos_path = f"{root_dir}/datasets/evaluation_data/youtube_videos.csv"

    latest_df = pd.read_csv(videos_path)
    headers = ['video_id', 'title', 'channel_name', 'published_date', 'url']

    if not os.path.exists(f"{root_dir}/pipeline_storage/youtube_videos.csv"):
        logging.info("No existing youtube_videos.csv found. Creating a new one.")
        pd.DataFrame(columns=headers).to_csv(f"{root_dir}/pipeline_storage/youtube_videos.csv", index=False)
    current_df = pd.read_csv(f"{root_dir}/pipeline_storage/youtube_videos.csv")
    youtube_videos_df = compute_new_entries(latest_df=latest_df, current_df=current_df, left_key='url', right_key='url', overwrite=overwrite)

    assert youtube_videos_df.shape[0] > 0, "Could not load YouTube videos CSV."
    partial_load_single_transcript = partial(load_single_video_transcript, youtube_videos_df=youtube_videos_df)
    video_transcripts_loaded_count = 0

    # Convert directory_path to a Path object if it is not already
    if not isinstance(directory_path, Path):
        directory_path = Path(directory_path)

    # Recursively find all .txt files in all subdirectories
    all_files = list(directory_path.rglob("*.txt"))  # This can be large depending on your filesystem

    # If num_files is provided, slice the list to process only the first 'num_files' files
    if num_files is not None:
        files = all_files[:num_files]  # Slice the list to get only the number of files you want
    else:
        files = all_files  # Otherwise, process all files

    # Using ThreadPoolExecutor to load PDFs in parallel
    worker_count = min(32, (os.cpu_count() or 1) + 4)
    limiter = get_global_thread_limiter()
    with limiter.claim(worker_count, label="transcripts-load"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            # Map over all video transcript files in the directory
            futures = {executor.submit(partial_load_single_transcript, file_path=video_transcript): video_transcript for video_transcript in files}

            for future in concurrent.futures.as_completed(futures):
                video_transcript = futures[future]
                try:
                    documents, all_documents_details = future.result()
                    # if documents is an empty list then continue
                    if not documents:
                        continue
                    all_documents.extend(documents)
                    all_metadata.append(all_documents_details)
                    video_transcripts_loaded_count += 1
                except Exception as e:
                    logging.info(f"Failed to process {str(video_transcript).replace(root_dir, '')}, passing: {e}")
                    # Check if the file name does not start with a date in the format yyyy-mm-dd_
                    if not re.match(r'^\d{4}-\d{2}-\d{2}_[a-zA-Z0-9_-]{11}_', os.path.basename(video_transcript)):
                        try:
                            os.remove(video_transcript)
                            logging.info(f"Deleted invalid file: {video_transcript}")
                        except OSError as delete_error:
                            logging.error(f"Error deleting file {video_transcript}: {delete_error}")
                    pass
    logging.info(f"Successfully loaded [{video_transcripts_loaded_count}] documents from video transcripts.")

    save_metadata_to_pipeline_dir(all_metadata, root_dir, dir='pipeline_storage/youtube_videos.csv', drop_key='video_link', headers=headers)
    # assert len(all_documents) > 1, f"Loaded only {len(all_documents)} documents from video transcripts. Something went wrong."
    return all_documents


if __name__ == '__main__':
    start_logging('log_prefix')
    load_video_transcripts(directory_path=Path(YOUTUBE_VIDEO_DIRECTORY), add_new_transcripts=False, num_files=None, overwrite=True)
