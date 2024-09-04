from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import torch
import faiss

def retrieve_relevant_passages(question, components, k=250):
    corpus = components['corpus']
    index = components['index']
    retriever = components['retriever']

    k = min(k, len(corpus))

    print("Retrieving relevant passages for question")
    
    question_embedding = retriever.encode(question, convert_to_tensor=True)
    question_embedding = question_embedding.unsqueeze(0) if question_embedding.dim() == 1 else question_embedding
    
    _, top_k_indices = index.search(question_embedding.cpu().numpy(), k)
    return [corpus[idx] for idx in top_k_indices[0]]

def generate_input(question, context):
    print("Generating a context for question")
    
    input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
    return input_text

def rag_pipeline(question, components, k):
    print("Running the RAG pipeline...")
    
    retrieved_passages = retrieve_relevant_passages(question, components, k)
    context = " ".join([passage for passage in retrieved_passages])

    return context

    # input = generate_input(question, context)

    # return {
    #     "question": question,
    #     "input": input,
    #     "retrieved_passages": retrieved_passages
    # }
