from pathlib import Path
import ray
import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup
from functools import partial
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ray.data import ActorPoolStrategy
import torch
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from ray.util import ActorPool
from ray.runtime_env import RuntimeEnv

runtime_env = RuntimeEnv(
    pip=["emoji"],
    env_vars={"TF_WARNINGS": "none"})

RuntimeEnv(conda={
    "channels": ["defaults"],
    "run_options": ["--cap-drop SYS_ADMIN","--log-level=debug"]})

# wget -e robots=off --recursive --no-clobber --page-requisites   --html-extension --convert-links --restrict-file-names=windows   --domains jestjs.io --no-parent --accept=html   -P $EFS_DIR https://jestjs.io/docs/getting-started

# export EFSDIR=/desired/directory

if ray.is_initialized():
    ray.shutdown()
    
# Inicializa o Ray
ray.init(ignore_reinit_error=True, log_to_driver=False)

# def print_cluster_resources():
#     cluster_resources = ray.cluster_resources()
#     available_resources = ray.available_resources()

#     print("Total Cluster Resources:", cluster_resources)
#     print("Available Resources:", available_resources)

# print_cluster_resources()

try:
    EFS_DIR = '/desired/output/directory'
    DOCS_DIR = Path(EFS_DIR, "jestjs.io/docs/")
    sample_html_fp = Path(DOCS_DIR, "getting-started.tmp.html")

    ds = ray.data.from_items([{"path": sample_html_fp}])
    chunk_size = 300
    chunk_overlap = 50
    
    # Função para extrair textos de um arquivo HTML
    def extract_texts(batch):
        path = batch['path']
        all_texts = []
        try:
            with open(path, 'r', encoding='utf-8') as fp:
                soup = BeautifulSoup(fp, 'html.parser')
            texts = [p.get_text(strip=True) for p in soup.find_all('p')]
            all_texts = [{'text': text} for text in texts]
        except Exception as e:
            print(f"Failed to process {path}: {str(e)}")
        return all_texts
    
    sections_ds = ds.flat_map(extract_texts)
    sections = sections_ds.take_all()
    
    sample_section = sections_ds.take(1)[0]


    def chunk_section(section, chunk_size, chunk_overlap):
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len)
        chunks = text_splitter.create_documents(
            texts=[sample_section["text"]], 
        )
        return [{"text": chunk.page_content} for chunk in chunks]
    
    chunks_ds = sections_ds.flat_map(partial(
        chunk_section, 
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap))

    @ray.remote(num_gpus=1)
    class EmbedChunks:
        def __init__(self, model_name):
            self.embed_model = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": "cuda"},
                    encode_kwargs={"device": "cuda", "batch_size": 100})
          
        def process(self, batch):
            embeddings = self.embed_model.embed_documents(batch["text"])
            return {"text": batch["text"], "embeddings": embeddings}
        
    embed_chunks_actor = EmbedChunks.remote(model_name="thenlper/gte-base")
    
    def process_batch(batch):
        return ray.get(embed_chunks_actor.process.remote(batch))
    
    embedded_chunks = chunks_ds.map_batches(
        process_batch,
        batch_size=100,  # Adjusted based on GPU memory and task complexity
        num_gpus=1,
        concurrency=2
    )
    
    try:
        print("Loading...")
        print(f"Processed {embedded_chunks.count()} chunks.")
    except Exception as e:
        print("ERROR: An issue occurred during processing...")
        print("Error:", e)

except Exception as e:
    print(f"Initialization failed: {str(e)}")

