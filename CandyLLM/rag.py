"""
🍭 CandyLLM Vector Database & RAG System
Comprehensive Retrieval-Augmented Generation with multiple vector store backends
"""

import os
import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json
import pickle
import hashlib

@dataclass
class Document:
    """Document representation for vector storage"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if not self.id:
            # Generate ID from content hash
            self.id = hashlib.md5(self.content.encode()).hexdigest()

@dataclass
class SearchResult:
    """Search result from vector store"""
    document: Document
    score: float
    rank: int

class VectorStore(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store"""
        pass
    
    @abstractmethod
    async def search(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[SearchResult]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    async def delete(self, document_ids: List[str]) -> bool:
        """Delete documents by ID"""
        pass
    
    @abstractmethod
    async def update_document(self, document: Document) -> bool:
        """Update a document"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        pass

class SimpleVectorStore(VectorStore):
    """
    Simple in-memory vector store for development and testing
    Uses basic cosine similarity for search
    """
    
    def __init__(self, embedding_function=None):
        self.documents: Dict[str, Document] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.embedding_function = embedding_function or self._default_embedding
    
    def _default_embedding(self, text: str) -> np.ndarray:
        """Simple hash-based embedding (for testing only)"""
        # Convert text to vector using simple hash method
        hash_bytes = hashlib.md5(text.encode()).digest()
        return np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32)
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the store"""
        added_ids = []
        
        for doc in documents:
            # Generate embedding if not provided
            if doc.embedding is None:
                if callable(self.embedding_function):
                    doc.embedding = await self._get_embedding(doc.content)
                else:
                    doc.embedding = self.embedding_function(doc.content)
            
            # Store document and embedding
            self.documents[doc.id] = doc
            self.embeddings[doc.id] = doc.embedding
            added_ids.append(doc.id)
        
        return added_ids
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text (async version)"""
        # For simple store, just call sync version
        return self.embedding_function(text)
    
    async def search(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[SearchResult]:
        """Search for similar documents"""
        if not self.embeddings:
            return []
        
        # Get query embedding
        query_embedding = await self._get_embedding(query)
        
        # Calculate similarities
        similarities = []
        for doc_id, doc_embedding in self.embeddings.items():
            document = self.documents[doc_id]
            
            # Apply filters
            if filter_dict:
                if not self._matches_filter(document.metadata, filter_dict):
                    continue
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((doc_id, similarity))
        
        # Sort by similarity and take top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        similarities = similarities[:k]
        
        # Create search results
        results = []
        for rank, (doc_id, score) in enumerate(similarities):
            document = self.documents[doc_id]
            results.append(SearchResult(document=document, score=score, rank=rank))
        
        return results
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria"""
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    async def delete(self, document_ids: List[str]) -> bool:
        """Delete documents by ID"""
        try:
            for doc_id in document_ids:
                if doc_id in self.documents:
                    del self.documents[doc_id]
                    del self.embeddings[doc_id]
            return True
        except Exception:
            return False
    
    async def update_document(self, document: Document) -> bool:
        """Update a document"""
        try:
            if document.embedding is None:
                document.embedding = await self._get_embedding(document.content)
            
            self.documents[document.id] = document
            self.embeddings[document.id] = document.embedding
            return True
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            "total_documents": len(self.documents),
            "total_embeddings": len(self.embeddings),
            "store_type": "SimpleVectorStore",
            "embedding_dimension": len(next(iter(self.embeddings.values()))) if self.embeddings else 0
        }

class ChromaVectorStore(VectorStore):
    """ChromaDB vector store implementation"""
    
    def __init__(self, collection_name: str = "candyllm", persist_directory: Optional[str] = None):
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("ChromaDB not installed. Run: pip install chromadb")
        
        self.collection_name = collection_name
        
        if persist_directory:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.client = chromadb.EphemeralClient()
        
        self.collection = self.client.get_or_create_collection(name=collection_name)
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to ChromaDB"""
        ids = [doc.id for doc in documents]
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Add embeddings if provided
        embeddings = []
        for doc in documents:
            if doc.embedding is not None:
                embeddings.append(doc.embedding.tolist())
            else:
                embeddings.append(None)
        
        if any(emb is not None for emb in embeddings):
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=[emb for emb in embeddings if emb is not None]
            )
        else:
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
        
        return ids
    
    async def search(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[SearchResult]:
        """Search ChromaDB"""
        where_filter = filter_dict if filter_dict else None
        
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where_filter
        )
        
        search_results = []
        for i in range(len(results['ids'][0])):
            doc_id = results['ids'][0][i]
            content = results['documents'][0][i]
            metadata = results['metadatas'][0][i] or {}
            distance = results['distances'][0][i]
            
            # Convert distance to similarity score (ChromaDB uses distance)
            score = 1.0 - distance
            
            document = Document(id=doc_id, content=content, metadata=metadata)
            search_results.append(SearchResult(document=document, score=score, rank=i))
        
        return search_results
    
    async def delete(self, document_ids: List[str]) -> bool:
        """Delete documents from ChromaDB"""
        try:
            self.collection.delete(ids=document_ids)
            return True
        except Exception:
            return False
    
    async def update_document(self, document: Document) -> bool:
        """Update document in ChromaDB"""
        try:
            # Delete and re-add (ChromaDB doesn't have direct update)
            await self.delete([document.id])
            await self.add_documents([document])
            return True
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ChromaDB statistics"""
        count = self.collection.count()
        return {
            "total_documents": count,
            "store_type": "ChromaVectorStore",
            "collection_name": self.collection_name
        }

class RAGSystem:
    """
    Retrieval-Augmented Generation System
    Combines vector search with LLM generation
    """
    
    def __init__(self, vector_store: VectorStore, llm_wrapper=None):
        self.vector_store = vector_store
        self.llm_wrapper = llm_wrapper
        self.system_prompt = """You are a helpful assistant. Use the provided context to answer questions accurately. If the context doesn't contain enough information to answer the question, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
    
    async def add_documents(self, documents: List[Union[Document, str, Dict]]) -> List[str]:
        """Add documents to the RAG system"""
        processed_docs = []
        
        for doc in documents:
            if isinstance(doc, str):
                # Simple string document
                processed_docs.append(Document(id="", content=doc, metadata={}))
            elif isinstance(doc, dict):
                # Dictionary with content and metadata
                content = doc.get('content', doc.get('text', ''))
                metadata = {k: v for k, v in doc.items() if k not in ['content', 'text']}
                doc_id = doc.get('id', '')
                processed_docs.append(Document(id=doc_id, content=content, metadata=metadata))
            elif isinstance(doc, Document):
                processed_docs.append(doc)
        
        return await self.vector_store.add_documents(processed_docs)
    
    async def search(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[SearchResult]:
        """Search for relevant documents"""
        return await self.vector_store.search(query, k, filter_dict)
    
    async def answer(self, question: str, k: int = 5, filter_dict: Optional[Dict] = None, **llm_kwargs) -> Dict[str, Any]:
        """Answer a question using RAG"""
        # Search for relevant documents
        search_results = await self.search(question, k, filter_dict)
        
        if not search_results:
            return {
                "answer": "I don't have enough information to answer this question.",
                "sources": [],
                "context_used": ""
            }
        
        # Prepare context
        context_parts = []
        sources = []
        
        for result in search_results:
            context_parts.append(f"[Source {result.rank + 1}]: {result.document.content}")
            sources.append({
                "id": result.document.id,
                "content": result.document.content[:200] + "..." if len(result.document.content) > 200 else result.document.content,
                "metadata": result.document.metadata,
                "score": result.score,
                "rank": result.rank
            })
        
        context = "\n\n".join(context_parts)
        
        # Generate answer using LLM
        if self.llm_wrapper:
            prompt = self.system_prompt.format(context=context, question=question)
            
            try:
                answer = self.llm_wrapper.answer(prompt, **llm_kwargs)
            except Exception as e:
                answer = f"Error generating answer: {e}"
        else:
            answer = "No LLM wrapper provided for answer generation."
        
        return {
            "answer": answer,
            "sources": sources,
            "context_used": context,
            "question": question
        }
    
    async def stream_answer(self, question: str, k: int = 5, filter_dict: Optional[Dict] = None, **llm_kwargs):
        """Stream answer using RAG (if LLM supports streaming)"""
        # Search for relevant documents
        search_results = await self.search(question, k, filter_dict)
        
        if not search_results:
            yield "I don't have enough information to answer this question."
            return
        
        # Prepare context
        context_parts = []
        for result in search_results:
            context_parts.append(f"[Source {result.rank + 1}]: {result.document.content}")
        
        context = "\n\n".join(context_parts)
        prompt = self.system_prompt.format(context=context, question=question)
        
        # Stream answer if supported
        if self.llm_wrapper and hasattr(self.llm_wrapper, 'stream'):
            try:
                async for chunk in self.llm_wrapper.stream(prompt, **llm_kwargs):
                    yield chunk
            except Exception as e:
                yield f"Error generating answer: {e}"
        else:
            # Fallback to regular answer
            result = await self.answer(question, k, filter_dict, **llm_kwargs)
            yield result["answer"]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        vector_stats = self.vector_store.get_stats()
        return {
            **vector_stats,
            "rag_system": True,
            "llm_wrapper": self.llm_wrapper is not None
        }

class DocumentLoader:
    """Utility class for loading documents from various sources"""
    
    @staticmethod
    def load_text_file(file_path: str, chunk_size: int = 1000, overlap: int = 100) -> List[Document]:
        """Load and chunk a text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return DocumentLoader.chunk_text(text, chunk_size, overlap, metadata={"source": file_path})
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100, metadata: Optional[Dict] = None) -> List[Document]:
        """Split text into chunks with overlap"""
        if metadata is None:
            metadata = {}
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence or word boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_space = chunk.rfind(' ')
                
                if last_period > len(chunk) * 0.8:  # Break at sentence if near end
                    end = start + last_period + 1
                    chunk = text[start:end]
                elif last_space > len(chunk) * 0.8:  # Break at word if near end
                    end = start + last_space
                    chunk = text[start:end]
            
            chunk_metadata = {
                **metadata,
                "chunk_index": len(chunks),
                "chunk_start": start,
                "chunk_end": end
            }
            
            chunks.append(Document(
                id="",  # Will be auto-generated
                content=chunk.strip(),
                metadata=chunk_metadata
            ))
            
            start = end - overlap
        
        return chunks
    
    @staticmethod
    def load_directory(directory_path: str, pattern: str = "*.txt", chunk_size: int = 1000) -> List[Document]:
        """Load all files matching pattern from directory"""
        from glob import glob
        
        documents = []
        file_paths = glob(os.path.join(directory_path, pattern))
        
        for file_path in file_paths:
            try:
                file_docs = DocumentLoader.load_text_file(file_path, chunk_size)
                documents.extend(file_docs)
            except Exception as e:
                print(f"⚠️  Error loading {file_path}: {e}")
        
        return documents

# Factory function for creating vector stores
def create_vector_store(store_type: str = "simple", **kwargs) -> VectorStore:
    """Create a vector store of the specified type"""
    
    if store_type.lower() == "simple":
        return SimpleVectorStore(**kwargs)
    elif store_type.lower() == "chroma":
        return ChromaVectorStore(**kwargs)
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")

# Factory function for creating RAG systems
def create_rag_system(vector_store_type: str = "simple", llm_wrapper=None, **vector_store_kwargs) -> RAGSystem:
    """Create a complete RAG system"""
    vector_store = create_vector_store(vector_store_type, **vector_store_kwargs)
    return RAGSystem(vector_store, llm_wrapper)
