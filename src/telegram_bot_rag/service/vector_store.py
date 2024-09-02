import logging.config
import uuid

import chromadb
from chromadb import EmbeddingFunction
from omegaconf import OmegaConf
from transformers import pipeline

logging_config = OmegaConf.to_container(OmegaConf.load("./src/telegram_bot_rag/conf/logging_config.yaml"), resolve=True)
logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str):
        self.model = pipeline(
            "feature-extraction",
            model=model_name
        )
    def __call__(self, input_document: str):
        batch_embeddings = self.model(input_document)[0][0]
        return batch_embeddings


class VectorStore:
    def __init__(self, embedding_model_name: str):
        self.client = chromadb.Client()
        self.embedding_func = SentenceTransformerEmbeddingFunction(embedding_model_name)
    def upsert_documents(
            self,
            documents_text: list[str],
            documents_name: list[str],
            collection_name: str
    ) -> None:
        collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_func
        )

        embeddings = [self.embedding_func(document_text)[0] for document_text in documents_text]
        logging.info(f"Upserting {len(documents_text)} documents to collection {collection_name}")
        collection.upsert(
            ids=[str(uuid.uuid4()) for _ in range(len(documents_text))],
            metadatas=[{"name": name} for name in documents_name],
            documents=documents_text,
            embeddings=embeddings,
        )

    def query(self, question: str, n_results: int, collection_name: str):
        collection = self.client.get_or_create_collection(collection_name, embedding_function=self.embedding_func)
        return collection.query(query_texts=question, n_results=n_results)

    def get_document_names(self, collection_name: str) -> list[str]:
        collection = self.client.get_or_create_collection(collection_name)
        collection_content = collection.get()
        document_names = list(set(item["name"] for item in collection_content["metadatas"]))
        return document_names
