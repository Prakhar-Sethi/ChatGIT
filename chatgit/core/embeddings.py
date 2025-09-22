"""Embedding model loader for ChatGIT"""

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import os


def load_embedding_model(
    model_name: str = None, device: str = "cpu"
) -> HuggingFaceBgeEmbeddings:
    """Load embedding model for vector search
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to run model on ('cpu' or 'cuda')
    
    Returns:
        Initialized embedding model
    """
    if model_name is None:
        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return embedding_model
