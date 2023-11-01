import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import sentry_sdk
from sentry_sdk.integrations.opentelemetry import SentrySpanProcessor, SentryPropagator
from traceloop.sdk import Traceloop

sentry_sdk.init(
    dsn="https://0b61b297b1e1a9aa6287f392dc96aa34@o4506150677970945.ingest.sentry.io/4506150679150592",
    enable_tracing=True,
    sample_rate=1.0,
    # set the instrumenter to use OpenTelemetry instead of Sentry
    instrumenter="otel",
)

# Traceloop.init(
#     disable_batch=True, processor=SentrySpanProcessor(), propagator=SentryPropagator()
# )

Traceloop.init(disable_batch=True)

from opentelemetry import trace
from opentelemetry.propagate import set_global_textmap
from sentry_sdk.integrations.opentelemetry import SentrySpanProcessor, SentryPropagator

provider = trace.get_tracer_provider()
provider.add_span_processor(SentrySpanProcessor())
set_global_textmap(SentryPropagator())

index_name = "gpt-4-langchain-docs-fast"
model_name = "text-embedding-ada-002"

index = pinecone.Index(index_name)

embed = OpenAIEmbeddings(
    model=model_name,
)

vectorstore = Pinecone(index, embed.embed_query, "text")

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)

qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
)

print(qa.run("how do I build an agent with LangChain?"))
