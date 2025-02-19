#!/usr/bin/env python
# -*- Coding: utf-8 -*-
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import KMeans

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

        # Check if the table already exists before attempting to create it
        if 'embedding' not in self.database.tables():
            self.database.define_table("embedding",
                                       Field("context", "string"),
                                       Field("vector", "blob")
                                       )

            self.database.define_table("kmeans",
                                       Field("kmeans", "blob"),
                                       Field("version", "integer")
                                       )

            self.database.define_table("clusters",
                                       Field("label", "integer"),
                                       Field("centroid", "blob"),
                                       Field("kmeans_id", "reference kmeans")
                                       )

            self.database.define_table("vector_cluster",
                                       Field("vector_id",
                                             "reference embedding"),
                                       Field("cluster_id", "reference clusters")
                                       )

            self.database.commit()

    def create_embedding(self, context: str, version: int = None):
        """
        Creates an embedding for a given context and stores it in the database.

        Args:
            context (str): The context to be embedded.
        """
        embedding = self.embedding_function.embed_query(context)
        embedding_blob = pickle.dumps(embedding)
        vector_id = self.database.embedding.insert(
            vector=embedding_blob,
            context=context
        )

        if self.database.clusters.isempty():
            return vector_id

        return self._insert_cluster(version, embedding, vector_id)

    def _insert_cluster(self, version, embedding, vector_id):
        if version is None:
            kmeans_loaded = self.database.select(self.database.kmeans.ALL,
                                                 orderby=~self.database.kmeans.version).first()
        else:
            query = self.database(self.database.kmeans.version == version)
            if query.isempty():
                raise ValueError(f"Version {version} of kmeans not found.")
            kmeans_loaded = query.select(self.database.kmeans.ALL,
                                         orderby=~self.database.kmeans.version).first()

        kmeans = pickle.loads(kmeans_loaded.kmeans)

        label = kmeans.predict(embedding)

        cluster = self.database(
            (self.database.clusters.label == label)
            &
            (self.database.clusters.kmeans_id == kmeans_loaded.id)
        ).select(self.database.clusters.ALL).first()

        self.database.vector_cluster.insert(
            vector_id=vector_id,
            cluster_id=cluster.id
        )

        self.database.commit()
        return vector_id

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

    def _batch_embedding(self, contexts: List[str]):
        qemb = DALVEC.embed_chunk(contexts, self.embedding_function)
        vector_ids = [
            self.database.embedding.insert(
                vector=pickle.dumps(qemb[i]),
                context=context
            ) for i, context in enumerate(contexts)
        ]

        self.database.commit()
        return vector_ids

    def batch_embedding(self, contexts: List[str], version: int = None):
        """
        Generates and stores embeddings for a batch of contexts.

        Args:
            contexts (List[str]): A list of contexts to be embedded.
        """

        if self.database.clusters.isempty():
            return self._batch_embedding(contexts)

        qemb = DALVEC.embed_chunk(contexts, self.embedding_function)

        if version is None:
            kmeans_loaded = self.database.select(self.database.kmeans.ALL,
                                                 orderby=~self.database.kmeans.version).first()
        else:
            query = self.database(self.database.kmeans.version == version)
            if query.isempty():
                raise ValueError(f"Version {version} of kmeans not found.")
            kmeans_loaded = query.select(self.database.kmeans.ALL,
                                         orderby=~self.database.kmeans.version).first()

        kmeans = pickle.loads(kmeans_loaded.kmeans)

        vector_ids = []

        # Store embeddings in the database
        for i, context in enumerate(contexts):
            embedding_blob = pickle.dumps(qemb[i])
            label = kmeans.predict(qemb[i])

            vector_id = self.database.embedding.insert(
                vector=embedding_blob,
                context=context
            )
            vector_ids.append(vector_id)

            cluster = self.database(
                (self.database.clusters.label == label)
                &
                (self.database.clusters.kmeans_id == kmeans_loaded.id)
            ).select(self.database.clusters.ALL).first()

            self.database.vector_cluster.insert(
                vector_id=vector_id,
                cluster_id=cluster.id
            )

        self.database.commit()
        return vector_ids

    def clustering(self, n_clusters: int = 3):
        """
        Realiza o agrupamento (clustering) dos vetores de embeddings armazenados no banco de dados utilizando o algoritmo K-Means.

        Args:
            n_clusters (int): Número de clusters a serem formados. O padrão é 3.

        Raises:
            ValueError: Se o número de vetores no banco de dados for menor que 20.
        """
        # Verifica se há vetores suficientes no banco de dados para realizar o clustering
        if self.database.embedding.count() < 20:
            raise ValueError("Number of vectors in Database is too low.")

        # Seleciona os vetores e seus IDs do banco de dados
        vectors = self.database.select(
            self.database.embedding.vector,
            self.database.embedding.id
        )

        # Converte os vetores selecionados para um array numpy
        vectors_array = np.array([
            self.get_vector_from_blob(vec.vector)
            for vec in vectors
        ])

        # Verifica a versão atual do K-Means no banco de dados
        total = self.database.kmeans.count()

        if total == 0:
            version = 0
        else:
            version = self.database.select(
                self.database.kmeans.version,
                orderby=~self.database.kmeans.version
            ).first().version + 1

        # Executa o algoritmo K-Means para agrupar os vetores
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(vectors_array)
        clusters_id = {}

        # Insere o modelo K-Means treinado no banco de dados
        kmeans_id = self.database.kmeans.insert(
            kmeans=pickle.dumps(kmeans),
            version=version
        )

        # Insere os centróides dos clusters no banco de dados
        for label, centroid in enumerate(kmeans.cluster_centers_):
            if label not in clusters_id:
                clusters_id[label] = self.database.clusters.insert(
                    label=label,
                    centroid=pickle.dumps(centroid),
                    kmeans_id=kmeans_id
                )

        # Armazena as associações dos vetores com os clusters no banco de dados
        for i, vector in enumerate(vectors):
            self.database.vector_clusters.insert(
                vector_id=vector.id,
                cluster_id=clusters_id[kmeans.labels_[i]]
            )

        # Confirma as alterações no banco de dados
        self.database.commit()

    def query_all(self, text: str, threshold: float):
        vectors = self.database.select(self.database.embedding.ALL)
        vectors_array = np.vstack([self.get_vector_from_blob(row.vector)
                                   for row in vectors])  # (n, m)
        query_embedding = np.array(
            self.embedding_function.embed_query(text)).reshape(1, -1)  # (1, m)
        # Calcula as similaridades entre a embedding da consulta e os vetores encontrados
        similarities = self.calc_similarities(query_embedding, vectors_array)

        # Ordena os índices dos vetores pela similaridade em ordem decrescente
        idx = np.argsort(similarities)[::-1]

        # Ordena as linhas dos vetores e as similaridades de acordo com os índices ordenados
        sorted_rows = [vectors[i] for i in idx]
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

    def query(self, text: str, threshold: float):
        """
        Queries the database for embeddings similar to the input text.

        Args:
            text (str): The input text to query.
            threshold (float): The similarity threshold.

        Returns:
            list: A list of dictionaries containing the id, context, and similarity of matching embeddings.
        """

        if self.database.clusters.isempty():
            return self.query_all(text, threshold)

        query_embedding = np.array(
            self.embedding_function.embed_query(text)).reshape(1, -1)  # (1, m)

        # Busca os clusters
        rows_clusters = self.database().select(self.database.clusters.ALL)

        # Constrói uma matriz de centróides a partir dos clusters encontrados
        centroids = np.vstack([self.get_vector_from_blob(row.centroid)
                               for row in rows_clusters])

        # Calcula as similaridades entre a embedding da consulta e os centróides
        centroid_similarities = self.calc_similarities(
            query_embedding, centroids)

        # Filtra os IDs dos clusters baseado no threshold de similaridade
        filtered_cluster_ids = [
            rows_clusters[i].id  # Pega o ID do cluster
            for i, sincenter in enumerate(centroid_similarities)
            if sincenter >= threshold - np.ceil(100 * threshold * 0.3) / 100
        ]
        if len(filtered_cluster_ids) == 0:
            return []

        # Busca os IDs dos vetores que pertencem aos clusters filtrados
        vector_ids = self.database(
            self.database.vector_clusters.cluster_id.belongs(
                filtered_cluster_ids)  # Usando belongs()
        ).select(self.database.vector_clusters.vector_id)

        # Busca os vetores da tabela `embedding` que correspondem aos `vector_ids` filtrados
        vectors_from_cluster = self.database(
            self.database.embedding.id.belongs(
                [vector_id.vector_id for vector_id in vector_ids])  # Usando belongs() também aqui
        ).select()

        # Constrói uma matriz de vetores a partir dos vetores encontrados
        vectors = np.vstack([self.get_vector_from_blob(row.vector)
                            for row in vectors_from_cluster])  # (n, m)

        if vectors.size == 0:
            return []

        # Calcula as similaridades entre a embedding da consulta e os vetores encontrados
        similarities = self.calc_similarities(query_embedding, vectors)

        # Ordena os índices dos vetores pela similaridade em ordem decrescente
        idx = np.argsort(similarities)[::-1]

        # Ordena as linhas dos vetores e as similaridades de acordo com os índices ordenados
        sorted_rows = [vectors_from_cluster[i] for i in idx]
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

    def calc_similarities(self, query_embedding, vectors):

        # Normaliza cada vetor
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        query_embedding = query_embedding / \
            np.linalg.norm(query_embedding)  # Normaliza o vetor de consulta

        similarities = np.dot(vectors, query_embedding.T).flatten()  # (n,)

        similarities = np.ceil(similarities * 100) / 100
        return similarities
