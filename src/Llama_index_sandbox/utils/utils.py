import csv
import shutil
import time
import logging
import inspect
from datetime import datetime
from functools import wraps
from pathlib import Path
from datetime import datetime
from functools import wraps, lru_cache
from typing import Iterable, Optional

from llama_index.legacy import OpenAIEmbedding
from llama_index.legacy.embeddings import HuggingFaceEmbedding

from llama_index.legacy.core.llms.types import ChatMessage, MessageRole
import pandas as pd


import subprocess

from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone


ROOT_MARKERS: tuple[str, ...] = (
    ".git",
    "pyproject.toml",
    "setup.cfg",
    "src",
)

def _in_docker() -> bool:
    if os.environ.get("IN_DOCKER", "").lower() in {"1", "true", "yes"}:
        return True
    return Path("/.dockerenv").exists()

def _scan_upward_for_root(start: Path, markers: Iterable[str], max_levels: int = 12) -> Optional[Path]:
    cur = start.resolve()
    root = cur.anchor
    for _ in range(max_levels):
        if any((cur / m).exists() for m in markers):
            return cur
        if str(cur) == root:
            break
        cur = cur.parent
    return None

def _git_root(start: Path) -> Optional[Path]:
    if not shutil.which("git"):
        return None
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(start),
            stderr=subprocess.DEVNULL,
            timeout=1.5,
        )
        return Path(out.decode("utf-8", "replace").strip())
    except Exception:
        return None


@lru_cache(maxsize=1)
def root_directory(start_from: Optional[Path | str] = None) -> str:
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        p = Path(env_root).expanduser().resolve()
        if p.exists():
            return str(p)

    if _in_docker():
        p = Path("/app")
        if p.exists():
            return str(p)

    start = Path(start_from) if start_from else Path.cwd()

    git_root = _git_root(start)
    if git_root and git_root.exists():
        return str(git_root)

    scanned = _scan_upward_for_root(start, ROOT_MARKERS, max_levels=16)
    if scanned:
        return str(scanned)

    here = Path(__file__).resolve().parent
    scanned_from_here = _scan_upward_for_root(here, ROOT_MARKERS, max_levels=16)
    if scanned_from_here:
        return str(scanned_from_here)

    raise RuntimeError(
        "Could not determine project root. Set PROJECT_ROOT environment variable to override."
    )


def start_logging(log_prefix):
    # Ensure that root_directory() is defined and returns the path to the root directory

    logs_dir = f'{root_directory()}/logs/txt'

    # Create a 'logs' directory if it does not exist, with exist_ok=True to avoid FileExistsError
    os.makedirs(logs_dir, exist_ok=True)

    # Get the current date and time
    now = datetime.now()
    timestamp_str = now.strftime('%Y-%m-%d_%H-%M')

    # Set up the logging level
    root_logger = logging.getLogger()

    # If handlers are already present, we can disable them.
    if root_logger.hasHandlers():
        # Clear existing handlers from the root logger
        root_logger.handlers.clear()

    root_logger.setLevel(logging.INFO)

    # Add handler to log messages to a file
    log_filename = f'{logs_dir}/{timestamp_str}_{log_prefix}.log'
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(file_handler)

    # Add handler to log messages to the standard output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(console_handler)

    # Now, any logging.info() call will append the log message to the specified file and the standard output.
    logging.info(f'********* {log_prefix} LOGGING STARTED *********')


def timeit(func):
    """
    A decorator that logs the time a function takes to execute along with the directory and filename.

    Args:
        func (callable): The function being decorated.

    Returns:
        callable: The wrapped function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        The wrapper function to execute the decorated function and log its execution time and location.

        Args:
            *args: Variable length argument list to pass to the decorated function.
            **kwargs: Arbitrary keyword arguments to pass to the decorated function.

        Returns:
            The value returned by the decorated function.
        """
        if os.getenv('ENVIRONMENT') == 'LOCAL':
            # Get the current file's path and extract directory and filename
            file_path = inspect.getfile(func)
            directory, filename = os.path.split(file_path)
            dir_name = os.path.basename(directory)

            # Log start of function execution
            logging.info(f"{dir_name}.{filename}.{func.__name__} STARTED.")
            start_time = time.time()

            # Call the decorated function and store its result
            result = func(*args, **kwargs)

            end_time = time.time()
            elapsed_time = end_time - start_time
            minutes, seconds = divmod(elapsed_time, 60)

            # Log end of function execution
            logging.info(f"{dir_name}.{filename}.{func.__name__} COMPLETED, took {int(minutes)} minutes and {seconds:.2f} seconds to run.\n")

            return result
        else:
            # If not in 'LOCAL' environment, just call the function without timing
            return func(*args, **kwargs)

    return wrapper

def get_last_index_embedding_params():
    index_dir = f"{root_directory()}/.storage/research_pdf/"
    index = sorted(os.listdir(index_dir))[-1].split('_')
    index_date = index[0]
    embedding_model_name = index[1]
    embedding_model_chunk_size = int(index[2])
    chunk_overlap = int(index[3])
    vector_space_distance_metric = 'cosine'  # TODO 2023-11-02: save vector_space_distance_metric in index name
    return embedding_model_name, embedding_model_chunk_size, chunk_overlap, vector_space_distance_metric



import os


def save_successful_load_to_csv(documents_details, csv_filename='docs.csv', fieldnames=['title', 'authors', 'pdf_link', 'release_date', 'document_name']):
    # Define the directory where you want to save the successful loads CSV
    from src.Llama_index_sandbox import output_dir

    # Create the directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(output_dir, csv_filename)
    file_exists = os.path.isfile(csv_path)

    if isinstance(documents_details, dict):
        # Filter documents_details for only the fields in fieldnames
        filtered_documents_details = {field: documents_details[field] for field in fieldnames}
    else:
        filtered_documents_details = {field: documents_details.extra_info[field] for field in fieldnames}

    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header only once if the file does not exist
        if not file_exists:
            writer.writeheader()

        # Write the filtered document details to the CSV
        writer.writerow(filtered_documents_details)


def get_embedding_model(embedding_model_name):
    if embedding_model_name == "text-embedding-ada-002":
        # embedding_model = OpenAIEmbedding(disallowed_special=())
        embedding_model = OpenAIEmbedding()  # https://github.com/langchain-ai/langchain/issues/923 encountered the same issue (2023-11-22)
    else:
        embedding_model = HuggingFaceEmbedding(
            model_name=embedding_model_name,
            # device='cuda'
        )
    # else:
    #     assert False, f"The embedding model is not supported: [{embedding_model_name}]"
    return embedding_model


def load_csv_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        logging.warning(f"CSV file not found at path: {file_path}")
        return pd.DataFrame()  # Return an empty DataFrame if file doesn't exist


@timeit
def compute_new_entries(latest_df: pd.DataFrame, current_df: pd.DataFrame, left_key='pdf_link', right_key='pdf_link', overwrite=False) -> pd.DataFrame:
    """
    Compute the difference between latest_df and research_papers,
    returning a DataFrame with entries not yet in research_papers.

    Parameters:
    - latest_df (pd.DataFrame): DataFrame loaded from latest_df.csv
    - current_df (pd.DataFrame): DataFrame loaded from current_df.csv

    Returns:
    - pd.DataFrame: DataFrame with entries not yet in research_papers.csv
    """
    # Assuming there's a unique identifier column named 'id' in both DataFrames
    # Adjust 'id' to the column name you use as a unique identifier
    if overwrite:
        logging.info(f"New to be added to the database found: [{len(latest_df)}]")
        return latest_df
    else:
        new_entries_df = latest_df[~latest_df[left_key].isin(current_df[right_key])]
        logging.info(f"New to be added to the database found: [{len(new_entries_df)}]")
    return new_entries_df


def load_vector_store_from_pinecone_database(delete_old_index=False, new_index=False, index_name=os.environ.get("PINECONE_INDEX_NAME", "icmfyi")):
    pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )
    if new_index:
        # pass
        if delete_old_index:
            logging.warning(f"Are you sure you want to delete the old index with name [{index_name}]?")
            pc.delete_index(index_name)
        # Dimensions are for text-embedding-ada-002
        from pinecone import ServerlessSpec
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    return vector_store


def load_vector_store_from_pinecone_database_legacy(index_name=os.environ.get("PINECONE_INDEX_NAME", "icmfyi")):
    pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )

    pinecone_index = pc.Index(index_name)
    # from llama_index.legacy.vector_stores import PineconeVectorStore
    import llama_index.legacy.vector_stores as legacy_vector_stores

    vector_store = legacy_vector_stores.PineconeVectorStore(pinecone_index=pinecone_index)
    return vector_store


def save_metadata_to_pipeline_dir(all_metadata, root_dir, dir='pipeline_storage/docs.csv', drop_key='pdf_link', headers=None):
    try:
        # Convert metadata to DataFrame
        df = pd.DataFrame(all_metadata)

        # Filter columns based on headers if provided
        if headers is not None:
            df = df[headers]

        csv_path = os.path.join(root_dir, dir)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        # Check if the CSV file already exists
        if os.path.exists(csv_path):
            # Read existing data and combine with new data
            existing_df = pd.read_csv(csv_path)
            combined_df = pd.concat([existing_df, df]).drop_duplicates(subset=[drop_key])
        else:
            # Use the new data as is if file doesn't exist
            combined_df = df

        # Save the combined data to CSV
        combined_df.to_csv(csv_path, index=False)
        logging.info(f"Metadata with # of unique entries [{combined_df.shape[0]}] saved to [{csv_path}]")
    except Exception as e:
        logging.error(f"Failed to save metadata to [{dir}]. Error: {e}")


if __name__ == '__main__':
    pass
