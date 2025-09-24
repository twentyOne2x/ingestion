"""Read PDF files using PyMuPDF library."""
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.Llama_index_sandbox.custom_pymupdfreader.readers.base import BaseReader
from llama_index.core.schema import Document


class PyMuPDFReader(BaseReader):
    """Read PDF files using PyMuPDF library."""

    def load_data(
        self,
        file_path: Union[Path, str],
        metadata: bool = True,
        extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        """Loads list of documents from PDF file and also accepts extra information in dict format."""
        return self.load(file_path, metadata=metadata, extra_info=extra_info)

    def load(
        self,
        file_path: Union[Path, str],
        metadata: bool = True,
        extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        """Loads list of documents from PDF file and also accepts extra information in dict format.

        Args:
            file_path (Union[Path, str]): file path of PDF file (accepts string or Path).
            metadata (bool, optional): if metadata to be included or not. Defaults to True.
            extra_info (Optional[Dict], optional): extra information related to each document in dict format. Defaults to None.

        Raises:
            TypeError: if extra_info is not a dictionary.
            TypeError: if file_path is not a string or Path.

        Returns:
            List[Document]: list of documents.
        """
        try:
            import pymupdf as fitz
        except ImportError:
            try:
                import fitz
            except ImportError:
                raise ImportError(
                    "PyMuPDF is not installed. Please install it with: pip install pymupdf"
                )

        # check if file_path is a string or Path
        if not isinstance(file_path, str) and not isinstance(file_path, Path):
            raise TypeError("file_path must be a string or Path.")

        # Convert Path to string if necessary
        if isinstance(file_path, Path):
            file_path = str(file_path)

        # open PDF file
        doc = fitz.open(file_path)

        # if extra_info is not None, check if it is a dictionary
        if extra_info:
            if not isinstance(extra_info, dict):
                raise TypeError("extra_info must be a dictionary.")

        documents = []

        # if metadata is True, add metadata to each document
        if metadata:
            if not extra_info:
                extra_info = {}
            extra_info["total_pages"] = len(doc)
            extra_info["file_path"] = file_path

            # Create documents with text as string (not bytes)
            for page_num, page in enumerate(doc):
                page_text = page.get_text()  # Don't encode to bytes

                # Create metadata for this page
                page_metadata = dict(
                    extra_info,
                    **{
                        "source": f"{page_num + 1}",
                        "page_number": page_num + 1,
                    }
                )

                # Create Document with text as positional argument or named parameter
                documents.append(
                    Document(
                        text=page_text,  # Pass as string, not bytes
                        metadata=page_metadata  # Use 'metadata' instead of 'extra_info' if that's what Document expects
                    )
                )
        else:
            for page in doc:
                page_text = page.get_text()  # Don't encode to bytes
                documents.append(
                    Document(
                        text=page_text,
                        metadata=extra_info or {}  # Use 'metadata' instead of 'extra_info'
                    )
                )

        # Close the document
        doc.close()

        return documents