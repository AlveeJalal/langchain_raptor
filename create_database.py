from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import os
import shutil

CHROMA_PATH = "chroma"
DATA_PATH = "data/docs"
MAX_CHUNK_SIZE = 300
OVERLAP_SIZE = 100


class ChunkNode:
    def __init__(self, document: Document, children=None):
        self.document = document
        self.children = children or []


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    root_nodes = split_text(documents)
    save_to_chroma(root_nodes)


def load_documents():
    loader = DirectoryLoader(DATA_PATH)
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    root_nodes = []
    for document in documents:
        text = document.page_content
        root_node = split_text_recursive(text, document.metadata)
        root_nodes.append(root_node)
    return root_nodes


def split_text_recursive(text: str, metadata: dict) -> ChunkNode:
    if len(text) <= MAX_CHUNK_SIZE:
        return ChunkNode(Document(page_content=text, metadata=metadata))

    chunks = []
    start_index = 0
    while start_index < len(text):
        end_index = start_index + MAX_CHUNK_SIZE
        # Find the end of the current sentence
        while end_index < len(text) and text[end_index] not in ['.', '!', '?']:
            end_index += 1
        # If end of sentence not found, take the next MAX_CHUNK_SIZE characters
        if end_index == len(text):
            end_index = start_index + MAX_CHUNK_SIZE
        # Extract the chunk
        chunk_text = text[start_index:end_index].strip()
        if chunk_text:  # Ensure non-empty chunks
            chunk_node = split_text_recursive(chunk_text, metadata)
            chunks.append(chunk_node)
        start_index = end_index

    return ChunkNode(Document(page_content="", metadata=metadata), chunks)



def save_to_chroma(root_nodes: list[ChunkNode]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Flatten the tree structure to a list of documents
    flat_chunks = flatten_tree(root_nodes)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        flat_chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(flat_chunks)} chunks to {CHROMA_PATH}.")


def flatten_tree(root_nodes: list[ChunkNode]) -> list[Document]:
    flat_chunks = []

    def traverse(node):
        flat_chunks.append(node.document)
        for child in node.children:
            traverse(child)

    for root_node in root_nodes:
        traverse(root_node)

    return flat_chunks


if __name__ == "__main__":
    main()
