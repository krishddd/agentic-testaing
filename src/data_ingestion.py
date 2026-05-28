import os
from typing import List, Optional, Any, Dict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader

from src.logging import logger
from src.chroma_vectorstore import ChromaDBVectorStoreManager 
from src.validators import save_documents


class DataIngestion:
    def __init__(
        self,
        config_dict,
        embedding_model: Any = None,
    ):
        self.embedding_model = embedding_model
        self.collection_name = config_dict['collection_name']
        self.db_vector_path = config_dict['db_vector_path']
        self.db_type = config_dict['db_type']
        self.data_path = config_dict['data_path']
        
        # Advanced chunking configuration
        self.chunk_size = config_dict.get('chunk_size', 1500)  # Optimized for long documents
        self.chunk_overlap = config_dict.get('chunk_overlap', 300)  # 20% overlap
        self.enable_semantic_chunking = config_dict.get('enable_semantic_chunking', False)

        print(f"[INIT] Config loaded: {config_dict}")
        print(f"[INIT] Chunking strategy: size={self.chunk_size}, overlap={self.chunk_overlap}")

    def load_and_split_data(self) -> List[Document]:
        """Load PDFs and split into optimized chunks for long documents"""
        print("[LOAD] Checking if data path exists...")

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path '{self.data_path}' does not exist.")

        logger.info("Loading documents...")
        print(f"[LOAD] Loading documents from: {self.data_path}")

        # Load documents
        if os.path.isdir(self.data_path):
            loader = PyPDFDirectoryLoader(self.data_path)
            print("[LOAD] Directory mode enabled - loading multiple PDFs.")
        else:
            loader = PyPDFLoader(self.data_path)
            print("[LOAD] Single PDF file mode.")

        documents = loader.load()
        print(f"[LOAD] {len(documents)} pages loaded from PDF(s).")

        if not documents:
            raise ValueError("No documents found in the specified path.")

        logger.info(f"Loaded {len(documents)} pages successfully!")

        # Analyze document characteristics
        total_chars = sum(len(doc.page_content) for doc in documents)
        avg_page_length = total_chars / len(documents) if documents else 0
        print(f"[ANALYSIS] Total characters: {total_chars:,}")
        print(f"[ANALYSIS] Average page length: {avg_page_length:.0f} characters")

        # Choose optimal splitting strategy based on content
        text_chunks = self._split_documents_intelligently(documents, avg_page_length)
        
        # Add enhanced metadata
        text_chunks = self._enrich_chunk_metadata(text_chunks)

        # Save chunks for validation
        save_documents(text_chunks, "document_data.pkl")
        logger.info(f"Split documents into {len(text_chunks)} chunks.")
        print(f"[SPLIT] {len(text_chunks)} chunks created and saved to document_data.pkl")

        return text_chunks

    def _split_documents_intelligently(
        self, 
        documents: List[Document], 
        avg_page_length: float
    ) -> List[Document]:
        """
        Intelligently split documents based on content type and length
        """
        print("[SPLIT] Analyzing optimal chunking strategy...")

        # Strategy 1: For very long research papers (avg page > 3000 chars)
        if avg_page_length > 3000:
            print("[SPLIT] Detected long-form content - using large chunk strategy")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,  # ~400-500 tokens - captures complex arguments
                chunk_overlap=400,  # 20% overlap for continuity
                length_function=len,
                separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
                keep_separator=True
            )
        
        # Strategy 2: For medium-length technical documents (1500-3000 chars)
        elif avg_page_length > 1500:
            print("[SPLIT] Detected technical content - using medium chunk strategy")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,  # ~300-375 tokens
                chunk_overlap=300,
                length_function=len,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
                keep_separator=True
            )
        
        # Strategy 3: For shorter documents or PDFs with lots of whitespace
        else:
            print("[SPLIT] Detected concise content - using standard chunk strategy")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,  # ~240-300 tokens
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
                keep_separator=True
            )

        logger.info("Splitting documents into chunks...")
        text_chunks = text_splitter.split_documents(documents)
        
        # Quality check
        avg_chunk_size = sum(len(chunk.page_content) for chunk in text_chunks) / len(text_chunks)
        print(f"[SPLIT] Average chunk size: {avg_chunk_size:.0f} characters")
        
        return text_chunks

    def _enrich_chunk_metadata(self, chunks: List[Document]) -> List[Document]:
        """
        Add comprehensive metadata to each chunk for better retrieval
        """
        print("[METADATA] Enriching chunks with metadata...")
        
        for i, chunk in enumerate(chunks):
            # Get existing metadata
            source_file = chunk.metadata.get('source', 'unknown')
            page_num = chunk.metadata.get('page', 'unknown')
            
            # Add enhanced metadata
            chunk.metadata.update({
                'chunk_id': i,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chunk_size': len(chunk.page_content),
                'source_file': os.path.basename(source_file) if source_file != 'unknown' else source_file,
                'page_number': page_num,
                # Add document type detection
                'doc_type': self._detect_doc_type(chunk.page_content),
                # Add content preview for debugging
                'preview': chunk.page_content[:100] + '...' if len(chunk.page_content) > 100 else chunk.page_content
            })
        
        print(f"[METADATA] Enhanced {len(chunks)} chunks with rich metadata")
        return chunks

    def _detect_doc_type(self, content: str) -> str:
        """
        Detect document type based on content patterns
        """
        content_lower = content.lower()
        
        # Research paper indicators
        if any(keyword in content_lower for keyword in ['abstract', 'introduction', 'methodology', 'conclusion', 'references']):
            return 'research_paper'
        
        # Technical documentation indicators
        elif any(keyword in content_lower for keyword in ['function', 'class', 'implementation', 'algorithm', 'procedure']):
            return 'technical_doc'
        
        # FAQ/Q&A indicators
        elif '?' in content and content.count('?') > 2:
            return 'faq'
        
        else:
            return 'general'

    def initialize_vectorstore(self, text_chunks: List[Document]) -> Optional[Any]:
        """Initialize vector store with processed chunks"""
        print("[VECTORSTORE] Initializing vector store...")

        if not text_chunks:
            raise ValueError("No text chunks available for vector store initialization.")

        chroma_manager = ChromaDBVectorStoreManager(
            embeddings=self.embedding_model, db_path=self.db_vector_path
        )
        print(f"[VECTORSTORE] Creating collection: {self.collection_name}")
        
        vectorstore = chroma_manager.create_collection(
            text_chunks, collection_name=self.collection_name
        )

        logger.info(f"Vector store '{self.collection_name}' initialized successfully.")
        print(f"[VECTORSTORE] Vector store '{self.collection_name}' initialized with {len(text_chunks)} chunks.")
        
        # Print summary statistics
        self._print_ingestion_summary(text_chunks)
        
        return vectorstore

    def _print_ingestion_summary(self, chunks: List[Document]):
        """Print comprehensive summary of data ingestion"""
        print("\n" + "="*60)
        print("DATA INGESTION SUMMARY")
        print("="*60)
        print(f"Total Chunks Created: {len(chunks)}")
        
        # Chunk size distribution
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        print(f"Chunk Size - Min: {min(chunk_sizes)}, Max: {max(chunk_sizes)}, Avg: {sum(chunk_sizes)/len(chunk_sizes):.0f}")
        
        # Document types
        doc_types = {}
        for chunk in chunks:
            doc_type = chunk.metadata.get('doc_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        print(f"Document Types: {doc_types}")
        
        # Source files
        sources = set(chunk.metadata.get('source_file', 'unknown') for chunk in chunks)
        print(f"Source Files: {len(sources)} unique file(s)")
        for source in sources:
            print(f"  - {source}")
        
        print("="*60 + "\n")

    def ingestion_main(self) -> Optional[Any]:
        """Main data ingestion pipeline with error handling"""
        print("[MAIN] Starting optimized data ingestion pipeline...")
        print("[MAIN] Pipeline optimized for: PDFs, Research Papers, Long Documents")

        try:
            text_chunks = self.load_and_split_data()
            print(f"[MAIN] Proceeding with {len(text_chunks)} optimized chunks for vector store.")
            return self.initialize_vectorstore(text_chunks)

        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            print(f"[ERROR] Ingestion failed: {e}")
            raise
