import streamlit as st
import os
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Initialize MongoDB connection
@st.cache_resource
def init_mongodb():
    try:
        client = MongoClient(os.getenv("MONGODB_URI"))
        db = client[os.getenv("DB_NAME", "kpmg")]
        collection = db[os.getenv("COLLECTION_NAME", "kpmg_doc")]
        return collection
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {str(e)}")
        return None

# Initialize embedding model
@st.cache_resource
def init_embedding_model():
    return SentenceTransformer('multi-qa-mpnet-base-cos-v1')

# Initialize LLM
@st.cache_resource
def init_llm():
    try:
        huggingfacehub_api_token = os.getenv("HUGGINGFACE_API_KEY")
        if not huggingfacehub_api_token:
            raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")
        
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",
            huggingfacehub_api_token=huggingfacehub_api_token,
            model_kwargs={"temperature": 0.7, "max_length": 512}
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize LLM: {str(e)}")
        return None

def get_embeddings(text: str, model) -> List[float]:
    return model.encode(text).tolist()

def semantic_search(query: str, collection, embedding_model, top_k=5):
    """
    Performs semantic search using MongoDB's $vectorSearch operator
    
    Args:
        query (str): The search query
        collection: MongoDB collection
        embedding_model: SentenceTransformer model
        k (int): Number of results to return
        
    Returns:
        List[Dict]: List of matching documents with scores
    """
    # Generate embedding for query
    query_embedding = get_embeddings(query, embedding_model)
    
    top_k = 5

    # Define the search pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "kpmg_vector_index",  # Your vector index name
                "queryVector": query_embedding,
                "path": "embedding",
                #"exact": True,
                "limit": top_k,
                "numCandidates": top_k * 10  # Optional: increases accuracy
                }
        },
        {
            "$project": {
                "filename": 1,
                "chunk_id": 1,
                "chunk_text": 1,
                "metadata": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    
    try:
        results = list(collection.aggregate(pipeline))
        return results
    except Exception as e:
        st.error(f"Vector search failed: {str(e)}")
        return []

def generate_response(context: List[Dict], query: str, llm) -> str:
    # Extract text from context
    context_text = "\n".join([r["chunk_text"] for r in context])
    
    prompt_template = """Based on the following context, answer the question.
    Context: {context}
    Question: {query}
    Answer: """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "query"]
    )
    
    formatted_prompt = prompt.format(context=context_text, query=query)
    
    try:
        response = llm.invoke(formatted_prompt)
        return response
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "Failed to generate response"

def main():
    st.title("Siemens-Healthineers Document Search")
    st.write("Ask questions about Siemens Healthineers documents using semantic search")
    
    # Initialize components
    collection = init_mongodb()
    embedding_model = init_embedding_model()
    llm = init_llm()
    
    # Check initialization results
    if collection is None or llm is None:
        st.error("Failed to initialize required components")
        return
    
    # Query input
    query = st.text_input("Enter your question:")
    
    if st.button("Search"):
        if query:
            with st.spinner("Searching documents..."):
                results = semantic_search(query, collection, embedding_model, top_k=5)
                
                if results:
                    #context = "\n".join([r["text"] for r in results])
                    
                    with st.spinner("Generating answer..."):
                        response = generate_response(results, query, llm)
                        
                        # Display response
                        st.write("### Answer")
                        st.write(response)
                        
                        # Display sources
                        st.write("### Sources")
                        for idx, result in enumerate(results, 1):
                            with st.expander(f"Source {idx}"):
                                st.write(f"**Text:** {result['chunk_text']}")
                                #st.write(f"**Metadata:** {result['metadata']}")
                                st.write(f"**Relevance Score:** {result['score']:.2f}")
                else:
                    st.warning("No relevant documents found")
        else:
            st.warning("Please enter a question")

if __name__ == "__main__":
    main()
