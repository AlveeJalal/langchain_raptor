# Langchain RAPTOR Tutorial

Install dependencies.

```python
pip install -r requirements.txt
```

Create the Chroma DB.

```python
python create_database.py
```

Query the Chroma DB.

```python
python query_data.py "How does Alice meet the Mad Hatter?"
```

With Streamlit: 
```
python -m streamlit run query_data.py 
```
Then run the page on the specified localhost link provided in the terminal. Then just query in the text field and hit the "Submit" button. 

You'll also need to set up an OpenAI account (and set the OpenAI key in your environment variable) for this to work.
