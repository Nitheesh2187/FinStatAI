import nest_asyncio
from dotenv import load_dotenv
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from src.Index import RAGIndex

nest_asyncio.apply()

load_dotenv()


embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
llm = Ollama(model="llama3.2:3b",request_timeout=420.0, temperature= 0.75)

index = RAGIndex(
    llm=llm,
    llamaparse_api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    embed_model=embed_model,
    embed_dim=384)

index.create_vectorstore("fin-statement-index")

index.add_data_to_vectorstore("DATA/Large Cap/Annual_Report_2024-250-261.pdf")

questions = [
    "Who is Shri Vinay M. Tonse?",
    "What is the capital reserves deductions during the year as at 31.03.2024?",
    "What is the Minority Interest on the date of balance sheet as at 31.03.2024?",
    "What is the total deposits as at 31.03.2024?",
    "How much is the Borrowings in India from capital instruments?",
    "What is the Balances with Reserve Bank of India in both current and other accounts as of year 2023"
]

for ques in questions:
    print("Question: ",ques)
    print("Retrived Documents:")
    print(index.retrieve_context(ques))
    print("Response:")
    print(index.generate_response(ques))
    print()