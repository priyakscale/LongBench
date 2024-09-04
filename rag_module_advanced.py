from openai import OpenAI
import faiss
import numpy as np
import time

# Initialize OpenAI client
client = OpenAI(api_key="sk-svcacct-3hvIVW8x7bx59eegJCv6T3BlbkFJevtK2uhr7pu2GA1f8f5m")

def make_gpt_request_with_retry(prompt: str, client, max_retries: int = 5, initial_wait: int = 2):
    retry_count = 0
    wait_time = initial_wait
    
    while retry_count < max_retries:
        try:
            # Make the GPT request
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "system", "content": prompt}],
            )
            return response
        
        except OpenAI.error.RateLimitError:
            retry_count += 1
            if retry_count >= max_retries:
                raise Exception("Max retries reached. Could not complete the request due to rate limits.")
            
            print(f"Rate limit hit. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            wait_time *= 2  # Exponential backoff
        
    raise Exception("Request failed due to too many rate limit errors.")

def get_openai_embeddings(texts, model="text-embedding-3-small"):
    print(f"Getting embeddings for {len(texts)} texts")
    
    embeddings = []
    for text in texts:
        text = text.replace("\n", " ")  # Clean the text to remove newlines
        response = client.embeddings.create(input=[text], model=model)
        embeddings.append(response.data[0].embedding)

    return np.array(embeddings)

def create_faiss_index(embeddings):
    print("Building FAISS index for fast retrieval...")
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    return index

def retrieve_relevant_passages(question, components, k=10):
    corpus = components['corpus']
    index = components['index']

    # Ensure k does not exceed the number of corpus items
    k = min(k, len(corpus))

    print(f"Retrieving top {k} relevant passages for question")
    question_embedding = get_openai_embeddings([question])[0].reshape(1, -1)
    
    _, top_k_indices = index.search(question_embedding, k)
    return [corpus[idx] for idx in top_k_indices[0]]

def rerank_with_gpt4(question, passages):
    print("Re-ranking passages with GPT-4")
    prompt = f"Given the question: '{question}', rank these passages by relevance as best as you can, even if full context is unavailable, without including the word 'Passage' or the passage numbers in your response.:\n\n"
    for i, passage in enumerate(passages):
        prompt += f"Passage {i + 1}: {passage}\n\n"
    prompt += "Return the passages ranked by relevance as best as you can. Only provide the ranked content without any ranking indicators or labels."

    # response = client.chat.completions.create(
    #     model="gpt-4",
    #     messages=[{"role": "system", "content": prompt}],
    # )
    response = make_gpt_request_with_retry(prompt=prompt, client=client)

    ranked_passages = response.choices[0].message.content.split("\n")
    return ranked_passages

def rag_pipeline(question, components, k):
    print("Running the RAG pipeline...")

    retrieved_passages = retrieve_relevant_passages(question, components, k)
    ranked_passages = rerank_with_gpt4(question, retrieved_passages)

    context = " ".join([passage for passage in ranked_passages])
    return context

# Initial setup: prepare corpus and build index
def prepare_rag_components(json_obj):
    input_text = json_obj.get('input', '')
    context = json_obj.get('context', '')
    
    # Split context into chunks
    chunk_size = 200
    corpus = [context[i:i+chunk_size] for i in range(0, len(context), chunk_size)]
    
    # Get OpenAI embeddings and create FAISS index
    embeddings = get_openai_embeddings(corpus)
    index = create_faiss_index(embeddings)

    components = {
        'corpus': corpus,
        'index': index,
    }
    
    return components

# Example JSON input
json_obj = {
    'input': 'What is the main idea of the document?',
    'context': '''
        Following the previous steps
        of long context scaling, it is vital to also align the
        model with instruction-following data to ensure
        that it can interact with various user requests in a
        chat interface (Wang et al., 2023). This phase, often
        referred to as supervised fine-tuning or instructiontuning, has been extensively studied in short context scenarios (Wang et al., 2022; Taori et al., 2023;
        Wang et al., 2023; Tunstall et al., 2023). However, the introduction of long sequences presents
        unique challenges in terms of data, training methods, and evaluation for alignment. Xiong et al.
        (2023) proposes generating long instruction data
        by concatenating short instruction data, yet their
        dataset and model weight are not open-sourced.
        On the other hand, while Chen et al. (2023b) has
        made their long instruction data, LongAlpaca-12k,
        available and employed LoRA (Hu et al., 2022) for
        efficient fine-tuning, it lacks in-depth discussion
        and comparative analysis of the influence of data
        and training methodologies. Our work aims to find
        an optimal solution for supervised (full parameter)
        fine-tuning on long context with full attention, by
        tuning data, training methods, and evaluating the
        aligned models on a wide range of tasks.
        '''  # your long context string
}

# if __name__ == "__main__":
#     print("Providing a RAG input instead of full long context")
#     components = prepare_rag_components(json_obj)
#     k = 10
    
#     # Generate the context using RAG
#     context = rag_pipeline(json_obj['input'], components, k)
#     json_obj['context'] = context

#     print(context)

