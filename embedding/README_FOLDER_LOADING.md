# Document Folder Loading Functionality

This document describes the enhanced document ingestion functionality that allows loading entire folders of documents with support for multiple file types.

## Features

- **Multi-file type support**: PDF, TXT, CSV, DOC, DOCX, and other unstructured files
- **Automatic file type detection**: Uses appropriate loaders based on file extensions
- **Error handling**: Gracefully handles failed file loads and continues processing
- **Metadata preservation**: Adds source file information to each document
- **Flexible filtering**: Can specify which file types to process
- **Comprehensive logging**: Detailed logging of the loading process

## Supported File Types

| Extension       | Loader                         | Description                   |
| --------------- | ------------------------------ | ----------------------------- |
| `.pdf`          | PyPDFLoader                    | PDF documents                 |
| `.txt`          | TextLoader                     | Plain text files              |
| `.csv`          | CSVLoader                      | Comma-separated values        |
| `.doc`, `.docx` | UnstructuredWordDocumentLoader | Microsoft Word documents      |
| Other           | UnstructuredFileLoader         | Fallback for other file types |

## Usage

### Basic Usage

```python
from ingestion import load_documents_from_directory

# Load all supported file types from a directory
documents = load_documents_from_directory("rag_data")
```

### Filter by File Type

```python
# Load only PDF files
documents = load_documents_from_directory(
    directory_path="rag_data",
    supported_extensions=['pdf']
)

# Load PDF and text files only
documents = load_documents_from_directory(
    directory_path="rag_data",
    supported_extensions=['pdf', 'txt']
)
```

### Individual File Loading

```python
from ingestion import get_loader_for_file

# Load a specific file
loader = get_loader_for_file("rag_data/document.pdf")
documents = loader.load()
```

## Function Reference

### `load_documents_from_directory(directory_path, supported_extensions=None)`

Loads all documents from a directory using appropriate loaders for each file type.

**Parameters:**

- `directory_path` (str): Path to the directory containing documents
- `supported_extensions` (List[str], optional): List of file extensions to process.
  If None, processes all supported file types.

**Returns:**

- List of loaded documents with metadata

**Raises:**

- `FileNotFoundError`: If the directory doesn't exist

### `get_loader_for_file(file_path)`

Returns the appropriate loader for a given file based on its extension.

**Parameters:**

- `file_path` (str): Path to the file

**Returns:**

- Document loader instance appropriate for the file type

## Document Metadata

Each loaded document includes the following metadata:

- `source`: The filename of the source document
- `file_path`: The full path to the source file
- Additional metadata specific to the file type (e.g., page numbers for PDFs)

## Error Handling

The system provides comprehensive error handling:

- **File not found**: Logs warning and continues with other files
- **Unsupported file types**: Logs info and skips the file
- **Loading errors**: Logs error details and continues processing
- **Directory not found**: Raises FileNotFoundError

## Example Output

```
INFO:__main__:Loading documents from directory: rag_data
INFO:__main__:Loading document: CV-1.pdf
INFO:__main__:Successfully loaded 1 pages/chunks from CV-1.pdf
INFO:__main__:Loading document: Candide_CoverLetter.pdf
INFO:__main__:Successfully loaded 1 pages/chunks from Candide_CoverLetter.pdf
INFO:__main__:Loading document: finalReport.pdf
INFO:__main__:Successfully loaded 1 pages/chunks from finalReport.pdf
INFO:__main__:Total documents loaded: 3
```

## Integration with Main Ingestion Pipeline

The main `ingestion.py` script has been updated to use the new folder loading functionality:

```python
# Load documents from the entire directory
documents_directory = "rag_data"
documents = load_documents_from_directory(
    directory_path=documents_directory,
    supported_extensions=['pdf', 'txt', 'csv', 'doc', 'docx']
)
```

## Running Examples

To see the functionality in action, run the example script:

```bash
cd embedding
python example_folder_loading.py
```

## Dependencies

Make sure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

The following additional packages are required for the new functionality:

- `unstructured`: For handling various file types
- `python-docx`: For Microsoft Word documents
- `pandas`: For CSV files
- `langchain-community`: For additional document loaders

## Best Practices

1. **File Organization**: Keep your documents organized in a dedicated directory
2. **File Naming**: Use descriptive filenames as they become part of the document metadata
3. **File Types**: Use supported file types for best results
4. **Error Monitoring**: Check logs for any failed file loads
5. **Memory Management**: For large directories, consider processing files in batches

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Install required packages from requirements.txt
2. **Permission errors**: Ensure read access to the document directory
3. **Corrupted files**: Check logs for specific file loading errors
4. **Memory issues**: Process smaller batches of files for very large directories

### Getting Help

If you encounter issues:

1. Check the logs for detailed error messages
2. Verify file permissions and paths
3. Ensure all dependencies are installed
4. Test with a single file first before processing entire directories
