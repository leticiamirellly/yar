from db import getContext
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings

local_model_directory="."
model = AutoModelForCausalLM.from_pretrained("local_model_directory")
tokenizer = AutoTokenizer.from_pretrained("local_model_directory")

class QueryAgent:
    def __init__(self):
        embedding_model_name = "thenlper/gte-base"
        self.embed_model = HuggingFaceEmbeddings(
                    model_name=embedding_model_name,
                    model_kwargs={"device": "cuda"},
                    encode_kwargs={"device": "cuda", "batch_size": 100})
    

    def __call__(self, query, num_chunks=5, stream=True):
        embedding_model_name = "thenlper/gte-base"
        def chunk_query(section):
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", " ", ""],
                chunk_size=128,
                chunk_overlap=50,
                length_function=len)
            chunks = text_splitter.create_documents(
                texts=[section], 
            )
            return [{"text": chunk.page_content} for chunk in chunks]

        def generate_response(
        model, tokenizer, prompt, temperature=0.1, max_length=2000
        ):
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                temperature=temperature,
                do_sample=True
            )
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
            store_context = getContext()
            embedding_query_actor = store_context

            embedding_queries_results = embedding_query_actor(embedding_model.embed_query(query))

            response = generate_response(model, tokenizer, embedding_queries_results[0])
            return response

        except TypeError as e:
            raise ValueError("Batch items must be dictionaries with a 'text' key.") from e