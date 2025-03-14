import os
import argparse
import logging
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader,
    CSVLoader
)
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.document_loaders.sql_database import SQLDatabaseLoader

# Set up logging
logging.basicConfig(level=logging.INFO)

load_dotenv()


# === 1. Loader functions per file type ===

def load_pdf(path):
    return PyPDFLoader(path).load()


def load_txt(path):
    return TextLoader(path).load()


def load_docx(path):
    return UnstructuredWordDocumentLoader(path).load()


def load_html(path):
    return UnstructuredHTMLLoader(path).load()


def load_csv(path):
    return CSVLoader(file_path=path).load()


def load_sqlite_db(path, query="SELECT * FROM sqlite_master"):
    db = SQLDatabase.from_uri(f"sqlite:///{path}")
    return SQLDatabaseLoader(db=db, query=query).load()


# === 2. File extension → loader mapping ===
LOADER_MAP = {
    ".pdf": load_pdf,
    ".txt": load_txt,
    ".docx": load_docx,
    ".html": load_html,
    ".csv": load_csv,
    ".db": load_sqlite_db
}


# === 3. Load all documents from folder ===
def load_all_documents(folder_path, preview=True):
    all_docs = []
    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext in LOADER_MAP:
            full_path = os.path.join(folder_path, filename)
            logging.info(f" Loading {filename}")
            try:
                docs = LOADER_MAP[ext](full_path)
                for doc in docs:
                    doc.metadata["source"] = filename
                    doc.metadata["file_type"] = ext
                all_docs.extend(docs)

                if preview:
                    for i, doc in enumerate(docs[:3]):
                        logging.info(f" Preview [{filename}] - Chunk {i + 1}:\n{doc.page_content[:300]}...\n")

            except Exception as e:
                logging.error(f" Failed to load {filename}: {str(e)}")
    return all_docs


# === 4. Deduplicate chunks ===
def deduplicate_chunks(chunks):
    seen = {}
    unique_chunks = []
    for chunk in chunks:
        content = chunk.page_content.strip()
        if content not in seen:
            seen[content] = chunk.metadata  # Store metadata if needed
            unique_chunks.append(chunk)
    return unique_chunks


# === 5. Main ingestion logic ===
def main():
    parser = argparse.ArgumentParser(description=" Multi-format RAG Ingestion Pipeline")
    parser.add_argument("--folder", default="data", help="Folder with input files")
    parser.add_argument("--db", default="rag_chroma_db", help="Chroma persist directory")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Size of each text chunk")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Overlap size for text chunks")

    args = parser.parse_args()
    logging.info(f" Scanning folder: {args.folder}")

    docs = load_all_documents(args.folder)
    logging.info(f" Raw documents loaded: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks = splitter.split_documents(docs)
    logging.info(f" Chunks generated: {len(chunks)}")

    chunks = deduplicate_chunks(chunks)
    logging.info(f" Deduplicated chunks: {len(chunks)}")

    vectorstore = Chroma.from_documents(documents=chunks, embedding=OpenAIEmbeddings(), persist_directory=args.db)
    logging.info(f" ChromaDB saved to: '{args.db}'")


if __name__ == "__main__":
    main()