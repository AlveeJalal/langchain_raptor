import streamlit as st
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
    st.title("Alvee's Assistant")

    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Initialize the model
    model = ChatOpenAI()

    user_input = st.text_input("You:", key="input_id")
    

    if st.button("Submit"):
        if user_input is None:
            st.write("Please enter a query.")
            return

        if user_input.lower() == "exit":
            st.write("Exiting...")
            st.stop()

        # Split user input by comma to get multiple queries
        queries = [query.strip() for query in user_input.split(",")]

        for query in queries:
            if not query:
                continue

            # Search the DB only if user inputs a question
            if "?" in query:
                results = db.similarity_search_with_relevance_scores(query, k=3)
                if len(results) == 0 or results[0][1] < 0.7:
                    st.write(f"Unable to find matching results for query: {query}")
                    continue

                context_tree = [doc for doc, _score in results]
                response_text, context_text = generate_response(context_tree, query, model)

                sources = [doc.metadata.get("source", None) for doc in context_tree]
                formatted_response = f"Context: {context_text}\nResponse: {response_text}\nSources: {sources}"
                st.write(formatted_response)
            



            else:
                # If it's not a question, just chat freely
                response_text = model.predict(query)
                st.write("Alvee's Assistant:", response_text)

    if st.button("Yay Response"):
        response_text2 = model.predict("This was a great answer, thanks!")
        st.write("Alvee's Assistant:", response_text2)

    elif st.button("Nay Response"):
        response_text2 = model.predict("I was not satisfied with your answer.")
        st.write("Alvee's Assistant:", response_text2)

               

    if st.button("Regenerate"):
        if user_input is None:
            st.write("Please enter a query.")
            return

        if user_input.lower() == "exit":
            st.write("Exiting...")
            st.stop()

        # Split user input by comma to get multiple queries
        queries = [query.strip() for query in user_input.split(",")]

        for query in queries:
            if not query:
                continue

            # Search the DB only if user inputs a question
            if "?" in query:
                results = db.similarity_search_with_relevance_scores(query, k=3)
                if len(results) == 0 or results[0][1] < 0.7:
                    st.write(f"Unable to find matching results for query: {query}")
                    continue

                context_tree = [doc for doc, _score in results]
                response_text, context_text = generate_response(context_tree, query, model)

                sources = [doc.metadata.get("source", None) for doc in context_tree]
                formatted_response = f"Context: {context_text}\nResponse: {response_text}\nSources: {sources}"
                st.write(formatted_response)
            else:
                # If it's not a question, just chat freely
                response_text = model.predict(query)
                st.write("Alvee's Assistant:", response_text)

def generate_response(context_tree, query, model):
    # Generate response recursively using context_tree
    context_text = "\n\n---\n\n".join([doc.page_content for doc in context_tree])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)
    response_text = model.predict(prompt)
    return response_text, context_text


if __name__ == "__main__":
    main()
