from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from typing import List
import numpy as np
import os


from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from typing import List
import numpy as np
import os


class CustonEmbedding(Embeddings):
    """
    Custom embedding class that uses SentenceTransformer to generate embeddings for texts.

    This class loads a SentenceTransformer model from a specified path or downloads it if not found.
    It provides methods for embedding documents and queries, as well as calculating cosine similarity
    between embeddings.

    Attributes:
        _stransform (SentenceTransformer): The SentenceTransformer model used for generating embeddings.

    Methods:
        __call__(texts):
            Makes the class compatible as an embedding function for Chroma.

        cosine_similarity(a, b):
            Computes the cosine similarity between two embedding vectors.

        stransform:
            Returns the SentenceTransformer model.

        encode(*args, **kwds):
            Encodes the given texts using the SentenceTransformer model.

        embed_documents(documents: List[str]) -> List[List[float]]:
            Generates embeddings for a list of documents.

        embed_query(query: str) -> List[float]:
            Generates an embedding for a single query.
    """

    def __init__(self, model_path: str, stmodel: str):
        """
        Initializes the CustonEmbedding class.

        Args:
            model_path (str): The path where the SentenceTransformer model is stored.
            stmodel (str): The name of the SentenceTransformer model to be used.
        """
        fpath = os.path.join(model_path, stmodel)
        try:
            self._stransform = SentenceTransformer(fpath)
            print(f"Model loaded: {stmodel}")
        except:
            print("Model not found, downloading...")
            self._stransform = SentenceTransformer(stmodel)
            self._stransform.save(fpath)

    def __call__(self, texts):
        """
        Makes the class compatible as an embedding function for Chroma.

        Args:
            texts (list or str): A list of strings or a single string to embed.

        Returns:
            list: Embedding vectors for the input texts.
        """
        if isinstance(texts, str):
            texts = [texts]
        return self.encode(texts, convert_to_tensor=False)

    def cosine_similarity(self, a, b):
        """
        Computes the cosine similarity between two embedding vectors.

        Args:
            a (numpy.ndarray): The first embedding vector.
            b (numpy.ndarray): The second embedding vector.

        Returns:
            float: The cosine similarity between the two vectors.
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    @property
    def stransform(self):
        """
        Returns the SentenceTransformer model.

        Returns:
            SentenceTransformer: The SentenceTransformer model.
        """
        return self._stransform

    def encode(self, *args, **kwds):
        """
        Encodes the given texts using the SentenceTransformer model.

        Args:
            *args: Variable length argument list.
            **kwds: Arbitrary keyword arguments.

        Returns:
            numpy.ndarray: Embedding vectors for the input texts.
        """
        return self._stransform.encode(*args, **kwds)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of documents.

        Args:
            documents (List[str]): A list of documents to embed.

        Returns:
            List[List[float]]: A list of embedding vectors for the input documents.
        """
        return [self.encode(d).tolist() for d in documents]

    def embed_query(self, query: str) -> List[float]:
        """
        Generates an embedding for a single query.

        Args:
            query (str): The query to embed.

        Returns:
            List[float]: The embedding vector for the input query.
        """
        return self.encode([query])[0].tolist()
