"""
Vector Database Replacement untuk ChromaDB
Compatible dengan Streamlit Cloud
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import pickle
import json
from sentence_transformers import SentenceTransformer
import faiss
import firebase_admin
from firebase_admin import firestore

class StreamlitVectorDB:
    """
    In-memory vector database replacement untuk ChromaDB
    Menggunakan FAISS + Firebase untuk persistence
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize vector database with embedding model"""
        self.model_name = model_name
        self.embeddings = None
        self.documents = []
        self.metadata = []
        self.index = None
        self.dimension = 384  # Dimension untuk all-MiniLM-L6-v2
        
        # Initialize embedding model
        self._load_model()
        
    def _load_model(self):
        """Load embedding model dengan caching"""
        try:
            # Cache model di session state
            if f'embedding_model_{self.model_name}' not in st.session_state:
                with st.spinner('Loading embedding model...'):
                    st.session_state[f'embedding_model_{self.model_name}'] = SentenceTransformer(self.model_name)
            
            self.model = st.session_state[f'embedding_model_{self.model_name}']
            st.success("✅ Embedding model loaded successfully")
            
        except Exception as e:
            st.error(f"❌ Error loading embedding model: {e}")
            # Fallback ke model yang lebih ringan
            self.model = None
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents ke vector database"""
        try:
            if self.model is None:
                st.error("❌ Embedding model not available")
                return False
                
            # Generate embeddings
            with st.spinner('Generating embeddings...'):
                embeddings = self.model.encode(documents)
            
            # Store documents and metadata
            self.documents.extend(documents)
            if metadata:
                self.metadata.extend(metadata)
            else:
                self.metadata.extend([{} for _ in documents])
            
            # Create or update FAISS index
            if self.index is None:
                self.index = faiss.IndexFlatL2(self.dimension)
            
            self.index.add(embeddings.astype('float32'))
            
            # Save to Firebase (optional persistence)
            self._save_to_firebase()
            
            st.success(f"✅ Added {len(documents)} documents to vector database")
            return True
            
        except Exception as e:
            st.error(f"❌ Error adding documents: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Search similar documents"""
        try:
            if self.model is None or self.index is None:
                st.warning("⚠️ Vector database not initialized")
                return []
            
            # Generate query embedding
            query_embedding = self.model.encode([query])
            
            # Search in FAISS index
            distances, indices = self.index.search(query_embedding.astype('float32'), k)
            
            # Prepare results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.documents):
                    results.append({
                        'document': self.documents[idx],
                        'metadata': self.metadata[idx],
                        'distance': float(distance),
                        'similarity': 1 / (1 + distance)  # Convert distance to similarity
                    })
            
            return results
            
        except Exception as e:
            st.error(f"❌ Error in similarity search: {e}")
            return []
    
    def _save_to_firebase(self):
        """Save vector database to Firebase (optional)"""
        try:
            if 'firebase_app' in st.session_state:
                db = firestore.client()
                
                # Save documents and metadata
                doc_ref = db.collection('vector_db').document('documents')
                doc_ref.set({
                    'documents': self.documents,
                    'metadata': self.metadata,
                    'timestamp': firestore.SERVER_TIMESTAMP
                })
                
        except Exception as e:
            st.warning(f"⚠️ Could not save to Firebase: {e}")
    
    def load_from_firebase(self):
        """Load vector database from Firebase"""
        try:
            if 'firebase_app' in st.session_state:
                db = firestore.client()
                
                doc_ref = db.collection('vector_db').document('documents')
                doc = doc_ref.get()
                
                if doc.exists:
                    data = doc.to_dict()
                    self.documents = data.get('documents', [])
                    self.metadata = data.get('metadata', [])
                    
                    # Rebuild FAISS index
                    if self.documents and self.model:
                        embeddings = self.model.encode(self.documents)
                        self.index = faiss.IndexFlatL2(self.dimension)
                        self.index.add(embeddings.astype('float32'))
                        
                        st.success("✅ Vector database loaded from Firebase")
                        return True
                        
        except Exception as e:
            st.warning(f"⚠️ Could not load from Firebase: {e}")
            
        return False

# Alternative: Pinecone Integration
class PineconeVectorDB:
    """
    Pinecone vector database integration
    Requires PINECONE_API_KEY in secrets
    """
    
    def __init__(self):
        try:
            import pinecone
            
            # Initialize Pinecone
            pinecone.init(
                api_key=st.secrets["PINECONE_API_KEY"],
                environment=st.secrets.get("PINECONE_ENVIRONMENT", "us-east1-gcp")
            )
            
            # Create or connect to index
            index_name = "audit-system-vectors"
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    index_name,
                    dimension=384,
                    metric="cosine"
                )
            
            self.index = pinecone.Index(index_name)
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            
            st.success("✅ Pinecone vector database initialized")
            
        except Exception as e:
            st.error(f"❌ Error initializing Pinecone: {e}")
            self.index = None
            self.model = None
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents to Pinecone"""
        try:
            if self.index is None or self.model is None:
                return False
                
            # Generate embeddings
            embeddings = self.model.encode(documents)
            
            # Prepare vectors for Pinecone
            vectors = []
            for i, (doc, embed) in enumerate(zip(documents, embeddings)):
                vector_id = f"doc_{len(vectors)}"
                vector_data = {
                    'id': vector_id,
                    'values': embed.tolist(),
                    'metadata': {
                        'document': doc,
                        **(metadata[i] if metadata else {})
                    }
                }
                vectors.append(vector_data)
            
            # Upload to Pinecone
            self.index.upsert(vectors)
            
            st.success(f"✅ Added {len(documents)} documents to Pinecone")
            return True
            
        except Exception as e:
            st.error(f"❌ Error adding documents to Pinecone: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Search similar documents in Pinecone"""
        try:
            if self.index is None or self.model is None:
                return []
                
            # Generate query embedding
            query_embedding = self.model.encode([query])[0]
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=k,
                include_metadata=True
            )
            
            # Format results
            formatted_results = []
            for match in results['matches']:
                formatted_results.append({
                    'document': match['metadata']['document'],
                    'metadata': match['metadata'],
                    'similarity': match['score']
                })
            
            return formatted_results
            
        except Exception as e:
            st.error(f"❌ Error in Pinecone search: {e}")
            return []

# Main function untuk initialize vector database
def initialize_vector_db(db_type: str = "streamlit"):
    """
    Initialize vector database based on type
    
    Args:
        db_type: "streamlit" (FAISS+Firebase) or "pinecone"
    """
    
    if db_type == "pinecone":
        return PineconeVectorDB()
    else:
        return StreamlitVectorDB()

# Usage example
if __name__ == "__main__":
    # Initialize vector database
    vector_db = initialize_vector_db("streamlit")
    
    # Add sample documents
    documents = [
        "Audit report for Q1 2024 financial statements",
        "Risk assessment for internal controls",
        "Compliance review for regulatory requirements"
    ]
    
    metadata = [
        {"type": "financial", "quarter": "Q1", "year": 2024},
        {"type": "risk", "department": "internal_controls"},
        {"type": "compliance", "department": "regulatory"}
    ]
    
    vector_db.add_documents(documents, metadata)
    
    # Search similar documents
    results = vector_db.similarity_search("financial audit report", k=3)
    
    for result in results:
        print(f"Document: {result['document']}")
        print(f"Similarity: {result['similarity']:.3f}")
        print("---")