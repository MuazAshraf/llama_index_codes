"""## Agentic RAG"""

from llama_index.core import SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

documents = SimpleDirectoryReader(input_files=["data\\testing.pdf"]).load_data()

from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)

from llama_index.core import SummaryIndex, VectorStoreIndex

summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes)

summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
vector_query_engine = vector_index.as_query_engine()



summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description=(
        "This Agent is useful to summarize the pdf in a simplied way by giving intuitions"
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "This Agent is useful to answer to the user queries within the pdf and retrieve the relevant piece of information"
    ),
)

query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        summary_tool,
        vector_tool,
    ],
    verbose=True
)

response = query_engine.query("tell me who is Ryan in the pdf")

print(response)

#response.source_nodes

response2 = query_engine.query("summarize the pdf")

print(response2)


