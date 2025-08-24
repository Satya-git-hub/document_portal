from __future__ import annotations
from pathlib import Path
from typing import Iterable, List
from fastapi import UploadFile
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredPowerPointLoader, UnstructuredMarkdownLoader, UnstructuredExcelLoader, CSVLoader, UnstructuredImageLoader
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException
from sqlalchemy import create_engine, inspect, text

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".ppt", ".pptx", ".md", ".csv", ".xlsx", ".db", ".sqlite"}

def load_sql_database(connection_string: str) -> List[Document]:
    """
    Load data from any SQL database using SQLAlchemy.
    """
    engine = create_engine(connection_string)
    inspector = inspect(engine)

    docs: List[Document] = []
    with engine.connect() as conn:
        for table_name in inspector.get_table_names():
            result = conn.execute(text(f"SELECT * FROM {table_name}"))
            col_names = result.keys()
            rows = result.fetchall()
            text_data = f"Table: {table_name}\nColumns: {', '.join(col_names)}\nRows:\n"
            for row in rows:
                text_data += f"{row}\n"
            docs.append(Document(page_content=text_data))
    return docs

def load_documents(paths: Iterable[Path]) -> List[Document]:
    """Load docs using appropriate loader based on extension or database URL."""
    docs: List[Document] = []
    try:
        for p in paths:
            ext = p.suffix.lower()
            if ext == ".pdf":
                loader = PyPDFLoader(str(p))
                docs.extend(loader.load())
            elif ext == ".docx":
                loader = Docx2txtLoader(str(p))
                docs.extend(loader.load())
            elif ext == ".txt":
                loader = TextLoader(str(p), encoding="utf-8")
                docs.extend(loader.load())
            elif ext in [".ppt", ".pptx"]:
                loader = UnstructuredPowerPointLoader(str(p))
                docs.extend(loader.load())
            elif ext == ".md":
                loader = UnstructuredMarkdownLoader(str(p))
                docs.extend(loader.load())
            elif ext == ".csv":
                loader = CSVLoader(file_path=str(p), encoding="utf-8")
                docs.extend(loader.load())
            elif ext == ".xlsx":
                loader = UnstructuredExcelLoader(str(p))
                docs.extend(loader.load())
            elif ext in [".db", ".sqlite"]:
                connection_url = f"sqlite:///{p}"
                docs.extend(load_sql_database(connection_url))
            else:
                log.warning("Unsupported extension skipped", path=str(p))
        log.info("Documents loaded", count=len(docs))
        return docs
    except Exception as e:
        log.error("Failed loading documents", error=str(e))
        raise DocumentPortalException("Error loading documents", e) from e

def concat_for_analysis(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("file_path") or "unknown"
        parts.append(f"\n--- SOURCE: {src} ---\n{d.page_content}")
    return "\n".join(parts)

def concat_for_comparison(ref_docs: List[Document], act_docs: List[Document]) -> str:
    left = concat_for_analysis(ref_docs)
    right = concat_for_analysis(act_docs)
    return f"<<REFERENCE_DOCUMENTS>>\n{left}\n\n<<ACTUAL_DOCUMENTS>>\n{right}"

# ---------- Helpers ----------
class FastAPIFileAdapter:
    """Adapt FastAPI UploadFile -> .name + .getbuffer() API"""
    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename
    def getbuffer(self) -> bytes:
        self._uf.file.seek(0)
        return self._uf.file.read()

def read_pdf_via_handler(handler, path: str) -> str:
    if hasattr(handler, "read_pdf"):
        return handler.read_pdf(path)  # type: ignore
    if hasattr(handler, "read_"):
        return handler.read_(path)  # type: ignore
    raise RuntimeError("DocHandler has neither read_pdf nor read_ method.")