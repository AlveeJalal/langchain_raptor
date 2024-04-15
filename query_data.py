import argparse
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Initialize the model
    model = ChatOpenAI()

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting...")
            break

        # Split user input by comma to get multiple queries
        queries = [query.strip() for query in user_input.split(",")]

        for query in queries:
            if not query:
                continue

            # Search the DB only if user inputs a question
            if "?" in query:
                results = db.similarity_search_with_relevance_scores(query, k=3)
                if len(results) == 0 or results[0][1] < 0.7:
                    print(f"Unable to find matching results for query: {query}")
                    continue

                context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
                prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
                prompt = prompt_template.format(context=context_text, question=query)
                print(prompt)

                response_text = model.predict(prompt)

                sources = [doc.metadata.get("source", None) for doc, _score in results]
                formatted_response = f"Response: {response_text}\nSources: {sources}"
                print(formatted_response)
            else:
                # If it's not a question, just chat freely
                response_text = model.predict(query)
                print("Alvee's Assistant:", response_text)


if __name__ == "__main__":
    main()
