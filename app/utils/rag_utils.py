"""
RAG (Retrieval-Augmented Generation) utility functions.
Handles document indexing, chunking, searching, and embedding operations.
"""

import uuid
import fitz  # PyMuPDF for PDF extraction
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue


# Global state for RAG system
_embedding_model = None
document_metadata = {}


def get_embedding_model(model_name: str):
    """
    Lazy-load and return the embedding model.
    
    Args:
        model_name: Name of the SentenceTransformer model to load
        
    Returns:
        SentenceTransformer model instance
    """
    global _embedding_model
    if _embedding_model is None:
        print(f"Loading embedding model: {model_name}...")
        _embedding_model = SentenceTransformer(model_name)
        print("Embedding model loaded.")
    return _embedding_model


def init_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int = 384):
    """
    Initialize Qdrant collection if it doesn't exist.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection to create
        vector_size: Dimension of the embedding vectors (default: 384 for all-MiniLM-L6-v2)
    """
    collections = client.get_collections().collections
    if not any(c.name == collection_name for c in collections):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"Created Qdrant collection: {collection_name}")


def extract_text_from_pdf(file_path) -> str:
    """
    Extract text from a PDF file using PyMuPDF.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    try:
        doc = fitz.open(str(file_path))
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence or paragraph boundary
        if end < text_len:
            # Look for last period, newline, or space
            for sep in ['\n\n', '\n', '. ', ' ']:
                last_sep = chunk.rfind(sep)
                if last_sep > chunk_size // 2:  # Only break if we're past halfway
                    chunk = chunk[:last_sep + len(sep)]
                    end = start + len(chunk)
                    break
        
        chunks.append(chunk.strip())
        start = end - overlap
        
        # Prevent infinite loop
        if start >= text_len - overlap:
            break
    
    return [c for c in chunks if c]  # Filter empty chunks


def index_document(
    client: QdrantClient,
    collection_name: str,
    embedding_model_name: str,
    user_id: str,
    doc_id: str,
    filename: str,
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> int:
    """
    Index a document's text chunks into Qdrant.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the Qdrant collection
        embedding_model_name: Name of the embedding model
        user_id: ID of the user uploading the document
        doc_id: Unique document ID
        filename: Original filename
        text: Document text content
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        Number of chunks indexed
    """
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    if not chunks:
        return 0
    
    model = get_embedding_model(embedding_model_name)
    embeddings = model.encode(chunks, show_progress_bar=False)
    
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point_id = str(uuid.uuid4())
        points.append(PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload={
                "user_id": user_id,
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": i,
                "text": chunk,
            }
        ))
    
    client.upsert(collection_name=collection_name, points=points)
    
    # Store metadata
    document_metadata[doc_id] = {
        "filename": filename,
        "user_id": user_id,
        "chunk_count": len(chunks),
        "text_length": len(text),
    }
    
    return len(chunks)


def search_documents(
    client: QdrantClient,
    collection_name: str,
    embedding_model_name: str,
    user_id: str,
    query: str,
    top_k: int = 5
) -> list[dict]:
    """
    Search for relevant document chunks for a user's query.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the Qdrant collection
        embedding_model_name: Name of the embedding model
        user_id: ID of the user
        query: Search query
        top_k: Number of results to return
        
    Returns:
        List of search results with text, filename, score, and chunk_index
    """
    model = get_embedding_model(embedding_model_name)
    query_embedding = model.encode([query], show_progress_bar=False)[0]
    
    # Use query_points for newer qdrant-client versions, fallback to search for older versions
    try:
        results = client.query_points(
            collection_name=collection_name,
            query=query_embedding.tolist(),
            query_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
            limit=top_k,
        ).points
    except AttributeError:
        # Fallback for older qdrant-client versions
        results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            query_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
            limit=top_k,
        )
    
    return [
        {
            "text": hit.payload.get("text", ""),
            "filename": hit.payload.get("filename", ""),
            "score": hit.score,
            "chunk_index": hit.payload.get("chunk_index", 0),
        }
        for hit in results
    ]


def get_user_documents(user_id: str) -> list[dict]:
    """
    Get list of documents indexed for a user.
    
    Args:
        user_id: ID of the user
        
    Returns:
        List of document metadata dictionaries
    """
    return [
        {"doc_id": doc_id, **meta}
        for doc_id, meta in document_metadata.items()
        if meta.get("user_id") == user_id
    ]


def delete_document(client: QdrantClient, collection_name: str, user_id: str, doc_id: str) -> bool:
    """
    Delete a document and its chunks from the index.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the Qdrant collection
        user_id: ID of the user
        doc_id: Document ID to delete
        
    Returns:
        True if successful, False otherwise
    """
    if doc_id not in document_metadata:
        return False
    
    meta = document_metadata[doc_id]
    if meta.get("user_id") != user_id:
        return False
    
    # Delete points with matching doc_id
    client.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        ),
    )
    
    del document_metadata[doc_id]
    return True


def perform_rag_search(
    client: QdrantClient,
    collection_name: str,
    embedding_model_name: str,
    user_id: str,
    query: str,
    top_k: int = 5
) -> str:
    """
    Perform RAG search and return formatted context.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the Qdrant collection
        embedding_model_name: Name of the embedding model
        user_id: ID of the user
        query: Search query
        top_k: Number of results to retrieve
        
    Returns:
        Formatted context string for LLM
    """
    results = search_documents(client, collection_name, embedding_model_name, user_id, query, top_k)
    
    if not results:
        return ""
    
    parts = ["**[Document Context]**\n"]
    for i, r in enumerate(results, 1):
        score_pct = int(r["score"] * 100)
        parts.append(f"\n**[{i}. {r['filename']} (relevance: {score_pct}%)]**\n{r['text']}\n")
    
    return "\n".join(parts)
