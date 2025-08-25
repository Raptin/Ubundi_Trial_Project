import os
import time
import random
from typing import List, Optional
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from openai import RateLimitError, APIError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimitedOpenAIEmbeddings:
    """
    A wrapper around OpenAIEmbeddings that implements rate limiting
    to avoid hitting OpenAI's rate limits.
    """

    def __init__(
        self,
        openai_api_key: str,
        requests_per_minute: int = 60,  # Default OpenAI limit for most users
        requests_per_day: int = 100000,  # Default OpenAI limit
        max_retries: int = 5,
        base_delay: float = 1.0,
    ):

        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.requests_per_minute = requests_per_minute
        self.requests_per_day = requests_per_day
        self.max_retries = max_retries
        self.base_delay = base_delay

        # Track request timing
        self.request_times: List[float] = []
        self.daily_requests = 0
        self.last_reset = time.time()

        # Calculate minimum delay between requests
        self.min_delay = 60.0 / self.requests_per_minute

        logger.info(
            f"Rate limiter initialized: {self.requests_per_minute} requests/minute, "
            f"{self.requests_per_day} requests/day"
        )

    def _cleanup_old_requests(self):
        """Remove requests older than 1 minute from tracking"""
        current_time = time.time()
        self.request_times = [t for t in self.request_times if current_time - t < 60.0]

    def _wait_if_needed(self):
        """Wait if we're approaching rate limits"""
        current_time = time.time()

        # Reset daily counter if it's a new day
        if current_time - self.last_reset >= 86400:  # 24 hours
            self.daily_requests = 0
            self.last_reset = current_time

        # Check daily limit
        if self.daily_requests >= self.requests_per_day:
            wait_time = 86400 - (current_time - self.last_reset)
            logger.warning(f"Daily limit reached. Waiting {wait_time:.2f} seconds")
            time.sleep(wait_time)
            self.daily_requests = 0
            self.last_reset = time.time()

        # Check per-minute limit
        self._cleanup_old_requests()
        if len(self.request_times) >= self.requests_per_minute:
            oldest_request = min(self.request_times)
            wait_time = 60.0 - (current_time - oldest_request) + 0.1  # Add small buffer
            if wait_time > 0:
                logger.info(f"Rate limit approaching. Waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
                self._cleanup_old_requests()

        # Ensure minimum delay between requests
        if self.request_times:
            time_since_last = current_time - self.request_times[-1]
            if time_since_last < self.min_delay:
                sleep_time = self.min_delay - time_since_last
                logger.debug(f"Enforcing minimum delay: {sleep_time:.2f} seconds")
                time.sleep(sleep_time)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed documents with rate limiting and retry logic
        """
        embeddings = []

        for i, text in enumerate(texts):
            logger.info(f"Processing text chunk {i+1}/{len(texts)}")

            # Wait if needed to respect rate limits
            self._wait_if_needed()

            # Attempt embedding with retries
            for attempt in range(self.max_retries):
                try:
                    # Record request time
                    current_time = time.time()
                    self.request_times.append(current_time)
                    self.daily_requests += 1

                    # Get embedding for single text
                    text_embedding = self.embeddings.embed_query(text)
                    embeddings.append(text_embedding)

                    logger.info(f"Successfully embedded chunk {i+1}")
                    break

                except RateLimitError as e:
                    logger.warning(f"Rate limit hit on attempt {attempt + 1}: {e}")
                    if attempt < self.max_retries - 1:
                        # Exponential backoff with jitter
                        delay = self.base_delay * (2**attempt) + random.uniform(0, 1)
                        logger.info(f"Waiting {delay:.2f} seconds before retry")
                        time.sleep(delay)
                        # Remove the failed request from tracking
                        if self.request_times:
                            self.request_times.pop()
                        self.daily_requests -= 1
                    else:
                        logger.error(
                            f"Failed to embed chunk {i+1} after {self.max_retries} attempts"
                        )
                        raise

                except APIError as e:
                    logger.error(f"API error on attempt {attempt + 1}: {e}")
                    if attempt < self.max_retries - 1:
                        delay = self.base_delay * (2**attempt) + random.uniform(0, 1)
                        logger.info(f"Waiting {delay:.2f} seconds before retry")
                        time.sleep(delay)
                        # Remove the failed request from tracking
                        if self.request_times:
                            self.request_times.pop()
                        self.daily_requests -= 1
                    else:
                        logger.error(
                            f"Failed to embed chunk {i+1} after {self.max_retries} attempts"
                        )
                        raise

                except Exception as e:
                    logger.error(f"Unexpected error embedding chunk {i+1}: {e}")
                    # Remove the failed request from tracking
                    if self.request_times:
                        self.request_times.pop()
                    self.daily_requests -= 1
                    raise

        return embeddings


def get_loader_for_file(file_path: str):
    """
    Returns the appropriate loader for a given file based on its extension.

    Args:
        file_path (str): Path to the file

    Returns:
        Document loader instance appropriate for the file type
    """
    file_extension = file_path.lower().split(".")[-1]

    if file_extension == "pdf":
        return PyPDFLoader(file_path)
    elif file_extension == "txt":
        return TextLoader(file_path, encoding="utf-8")
    elif file_extension == "csv":
        return CSVLoader(file_path)
    elif file_extension in ["doc", "docx"]:
        return UnstructuredWordDocumentLoader(file_path)
    else:
        # Fallback for other file types
        return UnstructuredFileLoader(file_path)


def load_documents_from_directory(
    directory_path: str, supported_extensions: Optional[List[str]] = None
) -> List:
    """
    Load all documents from a directory using appropriate loaders for each file type.

    Args:
        directory_path (str): Path to the directory containing documents
        supported_extensions (List[str], optional): List of file extensions to process.
                                                   If None, processes all supported file types.

    Returns:
        List: List of loaded documents
    """
    if supported_extensions is None:
        supported_extensions = ["pdf", "txt", "csv", "doc", "docx"]

    documents = []
    failed_files = []

    # Ensure the directory exists
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    # Get all files in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # Skip directories
        if os.path.isdir(file_path):
            continue

        # Check if file extension is supported
        file_extension = filename.lower().split(".")[-1]
        if file_extension not in supported_extensions:
            logger.info(f"Skipping unsupported file type: {filename}")
            continue

        try:
            logger.info(f"Loading document: {filename}")
            loader = get_loader_for_file(file_path)
            file_documents = loader.load()

            # Add source metadata to each document
            for doc in file_documents:
                doc.metadata["source"] = filename
                doc.metadata["file_path"] = file_path

            documents.extend(file_documents)
            logger.info(
                f"Successfully loaded {len(file_documents)} pages/chunks from {filename}"
            )

        except Exception as e:
            logger.error(f"Failed to load {filename}: {str(e)}")
            failed_files.append((filename, str(e)))

    logger.info(f"Total documents loaded: {len(documents)}")
    if failed_files:
        logger.warning(f"Failed to load {len(failed_files)} files:")
        for filename, error in failed_files:
            logger.warning(f"  - {filename}: {error}")

    return documents


def process_uploaded_file(file_path: str):
    """
    Process and upload an uploaded file to Pinecone
    """
    loader = get_loader_for_file(file_path)
    file_documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(file_documents)

    text_contents = [doc.page_content for doc in texts]

    rate_limited_embeddings = RateLimitedOpenAIEmbeddings(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        requests_per_minute=60,  # Adjust based on your plan
        requests_per_day=100000,  # Adjust based on your plan
        max_retries=5,
        base_delay=1.0,
    )
    try:
        embeddings = rate_limited_embeddings.embed_documents(text_contents)

        regular_embeddings = OpenAIEmbeddings(
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )

        PineconeVectorStore.from_documents(
            texts, regular_embeddings, index_name=os.environ.get("INDEX_NAME")
        )

    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise


def main():
    load_dotenv()

    # Load documents from the entire directory
    documents_directory = "rag_data"
    logger.info(f"Loading documents from directory: {documents_directory}")

    # You can specify which file types to process
    supported_extensions = [
        "pdf",
        "txt",
        "csv",
        "doc",
        "docx",
    ]  # Add or remove as needed

    documents = load_documents_from_directory(
        directory_path=documents_directory, supported_extensions=supported_extensions
    )

    if not documents:
        logger.error(
            "No documents were loaded. Please check the directory path and file types."
        )
        return

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    print(f"Split {len(documents)} documents into {len(texts)} chunks")

    # Extract text content for embedding
    text_contents = [doc.page_content for doc in texts]

    # Initialize rate-limited embeddings
    # Adjust these values based on your OpenAI plan:
    # - Free tier: 3 requests/minute, 200 requests/day
    # - Pay-as-you-go: 60 requests/minute, 100,000 requests/day
    # - Higher tiers: Check your specific limits
    rate_limited_embeddings = RateLimitedOpenAIEmbeddings(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        requests_per_minute=60,  # Adjust based on your plan
        requests_per_day=100000,  # Adjust based on your plan
        max_retries=5,
        base_delay=1.0,
    )

    try:
        # Get embeddings with rate limiting
        print("Starting embedding process with rate limiting...")
        embeddings = rate_limited_embeddings.embed_documents(text_contents)

        print(f"Successfully generated {len(embeddings)} embeddings")

        regular_embeddings = OpenAIEmbeddings(
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )

        # Create a mock document list with embeddings for Pinecone
        # This is a workaround since we generated embeddings separately
        from langchain_core.documents import Document

        # For now, we'll use the regular embeddings for Pinecone
        # In a production scenario, you might want to store the pre-computed embeddings
        print("Creating Pinecone vector store...")
        PineconeVectorStore.from_documents(
            texts, regular_embeddings, index_name=os.environ.get("INDEX_NAME")
        )

        print("Successfully created Pinecone vector store!")

    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise


if __name__ == "__main__":
    main()
