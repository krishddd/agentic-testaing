import fitz  # PyMuPDF
import os
import re
import json
import nltk
import pandas as pd
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from src.logging import logger

nltk.download('punkt')

class TestDataGeneration:
    def __init__(self, model):
        self.folder_path = "dataset"  # Folder containing your PDFs
        self.chunk_size = 1024
        self.model = model
        logger.info(f"Initialized TestDataGeneration with folder: {self.folder_path}")

    def get_document_chunks(self):
        try:
            documents = []
            for file in os.listdir(self.folder_path):
                if file.endswith(".pdf"):
                    file_path = os.path.join(self.folder_path, file)
                    doc = fitz.open(file_path)
                    for i, page in enumerate(doc):
                        text = page.get_text()
                        if text.strip():
                            documents.append(Document(
                                page_content=text,
                                metadata={"source": file, "page": i + 1}
                            ))
                    doc.close()

            text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size, separator=" ", chunk_overlap=100)
            split_docs = text_splitter.split_documents(documents)
            
            logger.info(f"Successfully split PDF(s) into {len(split_docs)} chunks.")
            return [doc.page_content for doc in split_docs]

        except Exception as e:
            logger.error(f"Error in get_document_chunks: {e}")
            raise

    def load_qa_generation_chain(self):
        try:
            # Define prompt template
            prompt_template = """
Generate a precise, specific question that directly captures the most important information from the given paragraph.

Guidelines:
1. Formulate a clear, focused question that highlights the key insight or main point of the paragraph.
2. Avoid vague questions like "What is this paragraph about?"
3. The question should be specific, concise, and directly extractable from the paragraph's content.
4. Create a comprehensive answer that:
- Uses information directly from the paragraph
- Provides detailed context
- Includes multiple relevant details
- Is at least 3-4 sentences long
5. If the paragraph lacks sufficient information to answer the question, respond with "I don't know."

Input Paragraph: {content}

Output:
Question: [Your specific, focused question]
Answer: [Detailed, comprehensive response]
            """

            prompt = PromptTemplate(input_variables=["content"], template=prompt_template)
            chain = prompt | self.model
            logger.info("QA generation chain loaded successfully.")
            return chain

        except Exception as e:
            logger.error(f"Error loading QA generation chain: {e}")
            raise

    def post_process_qa(self, generated_text: str) -> list:
        lines = generated_text.strip().splitlines()
        lines = [line.strip() for line in lines if line.strip()]

        qa_pairs = []
        i = 0
        while i < len(lines):
            if lines[i].lower().startswith("question:"):
                question = re.sub(r'^Question:\s*', '', lines[i], flags=re.IGNORECASE).strip()
                if i + 1 < len(lines) and lines[i+1].lower().startswith("answer:"):
                    answer = re.sub(r'^Answer:\s*', '', lines[i+1], flags=re.IGNORECASE).strip()
                    if question and answer and "i don't know" not in answer.lower():
                        qa_pairs.append({"question": question, "ground_truth_answer": answer})
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        
        return qa_pairs

    def generator(self, test_size=20, output_path="generation_evaluation_dataset.json"):
        logger.info("Starting test data generation...")
        try:
            chunks = self.get_document_chunks()
            qa_chain = self.load_qa_generation_chain()

            all_qa_pairs = []
            for context in chunks:
                if len(all_qa_pairs) >= test_size:
                    break
                
                result = qa_chain.invoke({"content": context})
                
                # Check if result is a string before post-processing
                if isinstance(result, str):
                    processed_qa = self.post_process_qa(result)
                    if processed_qa:
                        all_qa_pairs.extend(processed_qa)
                elif hasattr(result, 'content') and isinstance(result.content, str): # Handling for AIMessage
                    processed_qa = self.post_process_qa(result.content)
                    if processed_qa:
                        all_qa_pairs.extend(processed_qa)

            # Ensure we only have `test_size` items
            final_qa_pairs = all_qa_pairs[:test_size]

            # Save to JSON file
            with open(output_path, 'w') as f:
                json.dump(final_qa_pairs, f, indent=4)
                
            logger.info(f"Test data with {len(final_qa_pairs)} QA pairs saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error during test data generation: {e}")
            raise