# DALVEC

DALVEC is a Python library for managing embeddings and database operations. It leverages SentenceTransformer for generating embeddings and provides functionalities for storing and querying these embeddings in a database.

## Features

- Load or download SentenceTransformer models for embedding texts.
- Create and manage embeddings in a SQLite database.
- Batch processing of embeddings.
- Query embeddings with cosine similarity.

## Installation

To install DALVEC, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/dalvec.git
cd dalvec
pip install -r requirements.txt
```

## Usage

Initializing DALVEC

Create an instance of the DALVEC class by specifying the database file, database folder, and the SentenceTransformer model name.

```python
from dalvec import DALVEC

# Initialize DALVEC
dalvec = DALVEC(db_file='storage_embedding.db', dbfolder='./databaseembedding', stmodel='ricardo-filho/bert-base-portuguese-cased-nli-assin-2')

```

### Create an Embedding
Create an embedding for a given context and store it in the database.

```python
context = "This is a sample context."
dalvec.create(context)
``` 
### Batch Embedding
Generate and store embeddings for a batch of contexts.


```python
contexts = ["This is the first context.", "Here is the second context."]
dalvec.batch_embedding(contexts)
``` 

### Querying Embeddings
Query the database for embeddings similar to the input text.
```python
text = "sample query"
threshold = 0.8
results = dalvec.query(text, threshold)

for result in results:
    print(f"ID: {result['id']}, Context: {result['context']}, Similarity: {result['similarity']}")
```
