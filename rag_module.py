from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import torch
import faiss

# # Initialize global variables
# datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", "trec", 
#             "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

# longbench_data = {}
# corpus = []
# input_context_mapping = []

# print("Loading the LongBench datasets...")
# for dataset in datasets:
#     print(f"Loading dataset: {dataset}")
#     longbench_data[dataset] = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')

# print("Loading the LLaMA 3 model and tokenizer...")
# model_name = "meta-llama/Meta-Llama-3-8B"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# print("Loading SentenceTransformer model for retrieval...")
# retriever = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# print("Preparing the corpus from the LongBench dataset...")
# for dataset_name, dataset_data in longbench_data.items():
#     print(f"Processing dataset: {dataset_name}")
#     for i, item in enumerate(dataset_data):
#         corpus.append(item['context'])
#         input_context_mapping.append({
#             "input": item['input'],
#             "answers": item['answers'],
#             "dataset": item['dataset'],
#             "id": item['_id']
#         })

# print(f"Corpus prepared with {len(corpus)} items.")
# corpus_embeddings = retriever.encode(corpus, convert_to_tensor=True)

# embedding_dim = corpus_embeddings.shape[1]
# print("Building FAISS index for fast retrieval...")
# index = faiss.IndexFlatL2(embedding_dim)
# index.add(corpus_embeddings.cpu().numpy())

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
