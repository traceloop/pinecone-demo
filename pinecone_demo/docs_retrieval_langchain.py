import os
from pinecone import Pinecone
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from traceloop.sdk import Traceloop

Traceloop.init(disable_batch=True)

index_name = "gpt-4-langchain-docs-fast"
model_name = "text-embedding-ada-002"

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT")
)
index = pc.Index(index_name)

embeddings = OpenAIEmbeddings(
    model=model_name,
)

vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

questions_llm = ChatOpenAI(model_name="gpt-3.5-turbo")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)

qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
).with_config({"run_name": "langchain_qa_bot"})

for i in range(100):
    prompt = PromptTemplate.from_template(
        "Write a question a developer might ask about a framework for developing LLM applications."
    )
    output_parser = StrOutputParser()
    question = prompt | questions_llm | output_parser
    print(qa.invoke(question.invoke({})))
