from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.node_parser import MarkdownNodeParser
from src.Parser import PdfParser
from src.VectorDatabase import PineconeVectordatabase
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
from typing import List

class RAGIndex():
    def __init__(self,llm , llamaparse_api_key ,pinecone_api_key, embed_model, embed_dim) -> None:
        self.llm = llm
        self.embed_model = embed_model
        self.db = PineconeVectordatabase(pinecone_api_key,embed_dim)
        self.parser = PdfParser(llamaparse_api_key)
        self.vector_store = None
        self.ref_doc_ids = []

    def create_nodes_from_text(self,extracted_pages):
        nodes = []
        node_ids = []
        for page_num, text in extracted_pages.items():
            # Create a Document object for each page
            embedding = self.embed_model.get_text_embedding(text)
            doc = Document(
                text=text,
                metadata={
                    "page_number": page_num
                },
                embedding=embedding
            )
            nodes.append(doc)
            node_ids.append(doc.id_)
        return node_ids,nodes

    def create_nodes_from_markdown_pages(self, extracted_pages):
        # Create a MarkdownNodeParser
        markdown_parser = MarkdownNodeParser()
            
        # Parse documents into nodes
        nodes = []
        node_ids = []

        for page_num, doc in extracted_pages.items():
            doc.metadata={"page_number": page_num}
            doc.embedding = self.embed_model.get_text_embedding(doc.text)
            page_nodes = markdown_parser.get_nodes_from_documents([doc])
            for node in page_nodes:
                node.metadata["page_number"] = doc.metadata["page_number"]
                node_ids.append(node.id_)
            nodes.extend(page_nodes)
        
        return node_ids,nodes
    
    def create_vectorstore(self, index_name):
        if self.vector_store is None:
            pinecone_index = self.db.create_index(index_name)
            self.vector_store = PineconeVectorStore(pinecone_index=pinecone_index,index_name=index_name)

    def clear_vectorstore(self, delete_index = False):
        if self.vector_store:
            # Delete all documents from the vector store
            self.vector_store.delete_nodes(ids=self.ref_doc_ids)
        
        if delete_index:
            self.db.delete_index(self.vector_store.index_name)

    def add_data_to_vectorstore(self, pdf_path):
        pages_without_tables, pages_with_tables = self.parser(pdf_path)
        nodes = []
        if pages_with_tables:
            md_node_ids, md_nodes = self.create_nodes_from_markdown_pages(pages_with_tables)
            nodes.extend(md_nodes)
            self.ref_doc_ids.append(md_node_ids)
        if pages_without_tables:
            text_node_ids, text_nodes = self.create_nodes_from_text(pages_without_tables)
            nodes.extend(text_nodes)
            self.ref_doc_ids.append(text_node_ids)
        
        try:
            self.vector_store.add(nodes)
        except Exception as e:
            error = f"Unexpected Error as {e}"

    def retrieve_context(self, query):
        # Instantiate VectorStoreIndex object from your vector_store object
        vector_index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store,embed_model=self.embed_model)

        # Grab 5 search results
        retriever = VectorIndexRetriever(index=vector_index,embed_model=self.embed_model, similarity_top_k=5)

        # Query vector DB
        answer = retriever.retrieve(query)

        # Inspect results
        return [i.get_content() for i in answer]

    def generate_response(self, query):
        vector_index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store,embed_model=self.embed_model)
        query_engine = vector_index.as_query_engine(llm=self.llm)
        response = query_engine.query(query)
        return response
