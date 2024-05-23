import os
from pinecone import Pinecone, ServerlessSpec
from pinecone_datasets import load_dataset

dataset = load_dataset("langchain-python-docs-text-embedding-ada-002")
# we drop sparse_values as they are not needed for this example
dataset.documents.drop(["metadata"], axis=1, inplace=True)
dataset.documents.rename(columns={"blob": "metadata"}, inplace=True)

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT")
)

index_name = "gpt-4-langchain-docs-fast"

if index_name not in pc.list_indexes():
    # if does not exist, create index
    pc.create_index(
        name=index_name,
        dimension=1536,  # dimensionality of text-embedding-ada-002
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

for batch in dataset.iter_documents(batch_size=100):
    index.upsert(batch)
