#!/usr/bin/env python
# -*- Coding: utf-8 -*-
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydal import Field, DAL
from io import BytesIO
import numpy as np
import os
import pickle


from .customembedding import CustonEmbedding

from typing import List, Dict


class DALVEC:
    """
    DALVEC class for managing embeddings and database operations.

    This class initializes a connection to a database, sets up an embedding model,
    and provides methods for creating and querying embeddings.

    Attributes:
        _database (DAL): The database connection object.
        _db_file (str): The file path for the database.
        _dbfolder (str): The folder path for the database.
        _embedding_function (CustonEmbedding): The embedding function using SentenceTransformer.

    Methods:
        embed_single_chunk(i, chunk, embedding_function):
            Encodes a single chunk and logs execution.

        embed_chunk(chunks: List[List[str]], embedding_function) -> Dict[int, List[float]]:
            Parallelized version using ThreadPoolExecutor.

        database:
            Returns the database connection object.

        embedding_function:
            Returns the embedding function.

        tables():
            Creates the tables in the database.

        create(context: str):
            Creates an embedding for a given context and stores it in the database.

        get_vector_from_blob(blob_vector):
            Deserializes the embedding vector from a BLOB.

        batch_embedding(contexts: List[str]):
            Generates and stores embeddings for a batch of contexts.

        query(text: str, threshold: float):
            Queries the database for embeddings similar to the input text.
    """

    def __init__(self, db_file: str = 'storage_embedding.db', dbfolder: str = './databaseembedding', stmodel: str = "ricardo-filho/bert-base-portuguese-cased-nli-assin-2"):
        """
        Initializes the DALVEC class.

        Args:
            db_file (str): The file path for the database. Defaults to 'storage_embedding.db'.
            dbfolder (str): The folder path for the database. Defaults to './databaseembedding'.
            stmodel (str): The name of the SentenceTransformer model to be used. Defaults to "ricardo-filho/bert-base-portuguese-cased-nli-assin-2".
        """
        os.makedirs(dbfolder, exist_ok=True)

        # Initialize the database connection
        self._database = DAL(
            f"sqlite://{db_file}", db_codec='UTF-8', folder=dbfolder, pool_size=1)
        self._db_file = db_file
        self._dbfolder = dbfolder

        # Configure the embedding model
        MODELDIR = os.path.join(self._dbfolder, "model")
        os.makedirs(MODELDIR, exist_ok=True)

        # Create an instance of the CustonEmbedding class
        self._embedding_function = CustonEmbedding(MODELDIR, stmodel)

        # Create the tables
        self.tables()

    @staticmethod
    def embed_single_chunk(i, chunk, embedding_function):
        """
        Encodes a single chunk and logs execution.

        Args:
            i (int): The index of the chunk.
            chunk (list): The chunk to be encoded.
            embedding_function (CustonEmbedding): The embedding function.

        Returns:
            tuple: The index and the embedding vector.
        """
        embedding = embedding_function.embed_query(" ".join(chunk))
        return i, embedding

    @staticmethod
    def embed_chunk(chunks: List[List[str]], embedding_function) -> Dict[int, List[float]]:
        """
        Parallelized version using ThreadPoolExecutor.

        Args:
            chunks (List[List[str]]): A list of chunks to be encoded.
            embedding_function (CustonEmbedding): The embedding function.

        Returns:
            Dict[int, List[float]]: A dictionary mapping chunk indices to embedding vectors.
        """
        qemb = {}

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(
                DALVEC.embed_single_chunk, i, chunk, embedding_function): i for i, chunk in enumerate(chunks)}

            for future in as_completed(futures):
                i, embedding = future.result()
                if embedding is not None:
                    qemb[i] = embedding
        return qemb

    @property
    def database(self):
        """
        Returns the database connection object.

        Returns:
            DAL: The database connection object.
        """
        return self._database

    @property
    def embedding_function(self):
        """
        Returns the embedding function.

        Returns:
            CustonEmbedding: The embedding function.
        """
        return self._embedding_function

    def tables(self):
        """
        Creates the tables in the database.
        """
        print("Starting table creation...")

        # Check if the table already exists before attempting to create it
        if 'embedding' not in self.database.tables():
            self.database.define_table("embedding",
                                       Field("context", "string"),
                                       Field("vector", "blob"))
            self.database.commit()

    def create(self, context: str):
        """
        Creates an embedding for a given context and stores it in the database.

        Args:
            context (str): The context to be embedded.
        """
        embedding = self.embedding_function.embed_query(context)
        embedding_blob = pickle.dumps(embedding)
        self.database.embedding.insert(
            vector=embedding_blob,
            context=context
        )
        self.database.commit()

    def get_vector_from_blob(self, blob_vector):
        """
        Deserializes the embedding vector from a BLOB.

        Args:
            blob_vector (bytes): The BLOB containing the serialized embedding vector.

        Returns:
            numpy.ndarray: The deserialized embedding vector.
        """
        vector = pickle.loads(blob_vector)
        return np.array(vector)

    def batch_embedding(self, contexts: List[str]):
        """
        Generates and stores embeddings for a batch of contexts.

        Args:
            contexts (List[str]): A list of contexts to be embedded.
        """
        qemb = DALVEC.embed_chunk(contexts, self.embedding_function)

        # Store embeddings in the database
        for i, context in enumerate(contexts):
            embedding_blob = pickle.dumps(qemb[i])
            self.database.embedding.insert(
                vector=embedding_blob,
                context=context
            )

        self.database.commit()

    def query(self, text: str, threshold: float):
        """
        Queries the database for embeddings similar to the input text.

        Args:
            text (str): The input text to query.
            threshold (float): The similarity threshold.

        Returns:
            list: A list of dictionaries containing the id, context, and similarity of matching embeddings.
        """
        rows = self.database().select(self.database.embedding.id,
                                      self.database.embedding.vector,
                                      self.database.embedding.context)
        query_embedding = self.embedding_function.embed_query(text)

        similarities = np.array([
            np.ceil(self.embedding_function.cosine_similarity(
                query_embedding, self.get_vector_from_blob(row.vector)) * 100) / 100
            for row in rows
        ])

        idx = np.argsort(similarities)[::-1]
        sorted_rows = [rows[i] for i in idx]
        sorted_similarities = similarities[idx]
        result = [{
            "id": row.id,
            "context": row.context,
            "similarity": sim
        }
            for row, sim in zip(sorted_rows, sorted_similarities)
            if sim >= threshold
        ]
        return result
