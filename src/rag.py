import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from src.logging import logger
from src.model import Model
from src.retriever import RetrieverManager
from src.configuration import ConfigurationManager

config_manager = ConfigurationManager()
config_dict = config_manager.configurations()

class RAG:
    """
    Enhanced RAG pipeline with comprehensive answer generation.
    """
    def __init__(self, model, retriever):
        """
        Initializes the RAG pipeline components.
        """
        self.retriever = retriever
        self.model = model
        self.chain_type = 'retriever'
        
        # ENHANCED PROMPT: Explicitly requests comprehensive answers
        self.prompt = PromptTemplate.from_template("""You are an expert assistant specialized in providing comprehensive, detailed answers.

Your task is to answer the question using ONLY the information provided in the context below. 

CRITICAL INSTRUCTIONS:
1. **Be Comprehensive**: Include ALL relevant information from the context
2. **Preserve Structure**: If the context contains tables, lists, or structured information, maintain that organization
3. **Include Specifics**: Mention specific technical terms, mechanism names, categories, and details
4. **Use Sections**: Organize your answer with clear headings and subsections when appropriate
5. **Cite Sources**: When referencing specific information, note which document it comes from
6. **No Summarization**: Provide the FULL detail available - do not condense or shorten
7. **Stay Accurate**: Only use information explicitly stated in the context

CONTEXT:
{context}

QUESTION: {input}

COMPREHENSIVE ANSWER:""")

        self.chain = self._create_retrieval_chain()

    def _create_retrieval_chain(self):
        """
        Creates the document retrieval and generation chain.
        """
        if self.chain_type == 'retriever':
            logger.info('Creating enhanced document retrieval chain...')
            document_chain = create_stuff_documents_chain(self.model, self.prompt)
            retrieval_chain = create_retrieval_chain(self.retriever, document_chain)
            return retrieval_chain
        else:
            logger.error("Unsupported chain type specified: %s", self.chain_type)
            return None

    def invoke(self, input_dict: dict) -> dict:
        """
        Invokes the RAG chain with enhanced logging.
        """
        if not self.chain:
            raise ValueError("The RAG chain has not been initialized properly.")
        
        logger.info(f"Processing query: {input_dict.get('input', '')[:100]}...")
        original_result = self.chain.invoke(input_dict)

        # Enhanced logging of retrieved context
        print("\n" + "="*80)
        print("RETRIEVED CONTEXT DOCUMENTS")
        print("="*80)
        
        for i, doc in enumerate(original_result.get("context", [])):
            source = doc.metadata.get('source', 'unknown')
            page = doc.metadata.get('page', '?')
            rerank_score = doc.metadata.get('rerank_score', 'N/A')
            
            print(f"\n[Document {i+1}]")
            print(f"  Source: {source}")
            print(f"  Page: {page}")
            print(f"  Rerank Score: {rerank_score}")
            print(f"  Content Preview: {doc.page_content[:300]}...")
            print("-" * 80)

        logger.info(f"Generated answer length: {len(original_result['answer'])} characters")
        return original_result


def main():
    """
    Main function to set up and run the enhanced RAG pipeline.
    """
    logger.info("Setting up the enhanced RAG pipeline...")
    model_obj = Model(config_dict)
    
    embedding = model_obj.load_ollama_embedding()
    llm_model = model_obj.load_ollama_model()

    # Use enhanced retriever with reranking
    retriever_manager = RetrieverManager(embeddings=embedding, config_dict=config_dict)
    
    # Get enhanced retriever: retrieve 15 docs, rerank to top 8
    retriever = retriever_manager.get_retriever(
        k=15,  # Retrieve more initially
        enable_reranking=True,  # Enable cross-encoder reranking
        top_n_rerank=8  # Keep top 8 after reranking
    )
    
    rag_pipeline = RAG(model=llm_model, retriever=retriever)
    
    logger.info("Enhanced RAG pipeline is ready with reranking enabled.")
    return rag_pipeline


if __name__ == "__main__":
    rag_chain = main()
    
    # Test question
    question = "What security and privacy mechanisms are essential for robust autonomous agent deployment?"
    
    print("\n" + "="*80)
    print(f"QUESTION: {question}")
    print("="*80)
    
    response = rag_chain.invoke({"input": question})
    
    print("\n" + "="*80)
    print("FINAL ANSWER")
    print("="*80)
    print(response["answer"])
    print("\n" + "="*80)