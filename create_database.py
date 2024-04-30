from langchain.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import os
import shutil

CHROMA_PATH = "chroma"
DATA_PATH = "data/docs"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    hierarchies = generate_hierarchical_structure(documents)
    save_to_chroma(hierarchies)


def load_documents():
    loader = DirectoryLoader(DATA_PATH)
    documents = loader.load()
    return documents


def generate_hierarchical_structure(documents: list[Document]):
    hierarchies = []
    for doc in documents:
        hierarchy = parse_hierarchy(doc)
        hierarchies.append(hierarchy)
    return hierarchies


def parse_hierarchy(document: Document):
    # Split the document into paragraphs or sections.
    paragraphs = document.content.split("\n\n")  # Assuming paragraphs are separated by double newlines.

    # Construct the hierarchical structure.
    hierarchy = {
        "title": document.title,  # Assuming the document title represents the root of the hierarchy.
        "content": paragraphs[0],  # First paragraph as content of the root node.
        "children": []  # Placeholder for child nodes.
    }

    # Recursively create child nodes for each paragraph or section.
    for paragraph in paragraphs[1:]:
        hierarchy["children"].append({
            "content": paragraph,
            "children": []  # Placeholder for child nodes.
        })

    return hierarchy


def save_to_chroma(hierarchies):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma(OpenAIEmbeddings())
    for hierarchy in hierarchies:
        traverse_hierarchy(hierarchy, db)
    db.persist(CHROMA_PATH)
    print(f"Saved chunks to {CHROMA_PATH}.")


def traverse_hierarchy(hierarchy, db):
    # Add the content of the current node to the database.
    db.add_document(hierarchy["content"])

    # Recursively traverse child nodes.
    for child in hierarchy["children"]:
        traverse_hierarchy(child, db)


if __name__ == "__main__":
    main()
