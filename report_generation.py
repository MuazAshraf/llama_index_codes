import asyncio
from llama_index.core import set_global_tokenizer
import tiktoken
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext
from dotenv import load_dotenv
import os

load_dotenv('.env')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = os.getenv('PINECONE_API_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
pc = Pinecone(api_key=PINECONE_API_KEY)
set_global_tokenizer(tiktoken.encoding_for_model("gpt-4o-mini").encode)


pinecone_index = pc.Index("testing")
documents = SimpleDirectoryReader(input_files=["data\\testing.pdf"]).load_data()

### setup embedding/LLM model
embed_model = OpenAIEmbedding()
llm = OpenAI(model="gpt-4o-mini")

Settings.embed_model = embed_model
Settings.llm = llm

vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace="testing")

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
#Define File Retriever
doc_retriever = index.as_retriever(
    retrieval_mode="files_via_content",
    files_top_k=1
)

nodes = doc_retriever.retrieve("Give me a summary of Tesla in 2019") 

#Define chunk retriever
#The chunk-level retriever does vector search with a final reranked set of rerank_top_n=5.
chunk_retriever = index.as_retriever(
    retrieval_mode="chunks",
    rerank_top_n=5
)

#Define Retriever Tools
#Wrap these with Python functions into tool objects - these will directly be used by the LLM.
from llama_index.core.tools import FunctionTool
from llama_index.core.schema import NodeWithScore
from typing import List

# function tools
def chunk_retriever_fn(query: str) -> List[NodeWithScore]:
    """Retrieves a small set of relevant document chunks from the corpus.

    ONLY use for research questions that want to look up specific facts from the knowledge corpus,
    and don't need entire documents.

    """
    return chunk_retriever.retrieve(query)

def doc_retriever_fn(query: str) -> float:
    """Document retriever that retrieves entire documents from the corpus.

    ONLY use for research questions that may require searching over entire research reports.

    Will be slower and more expensive than chunk-level retrieval but may be necessary.
    """
    return doc_retriever.retrieve(query)

chunk_retriever_tool = FunctionTool.from_defaults(fn=chunk_retriever_fn)
doc_retriever_tool = FunctionTool.from_defaults(fn=doc_retriever_fn)

#Build a Report Generation Workflow
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Tuple
import pandas as pd
from IPython.display import display, Markdown


class TextBlock(BaseModel):
    """Text block."""

    text: str = Field(..., description="The text for this block.")


class TableBlock(BaseModel):
    """Image block."""

    caption: str = Field(..., description="Caption of the table.")
    col_names: List[str] = Field(..., description="Names of the columns.")
    rows: List[Tuple] = Field(
        ...,
        description=(
            "List of rows. Each row is a data entry tuple, "
            "where each element of the tuple corresponds positionally to the column name."
        )
    )

    def to_df(self) -> pd.DataFrame:
        """To dataframe."""
        df = pd.DataFrame(self.rows, columns=self.col_names)
        df.style.set_caption(self.caption)
        return df


class ReportOutput(BaseModel):
    """Data model for a report.

    Can contain a mix of text and table blocks. Use table blocks to present any quantitative metrics and comparisons.

    """

    blocks: List[TextBlock | TableBlock] = Field(
        ..., description="A list of text and table blocks."
    )

    def render(self) -> None:
        """Render as formatted text within a jupyter notebook."""
        for b in self.blocks:
            if isinstance(b, TextBlock):
                display(Markdown(b.text))
            else:
                display(b.to_df())


report_gen_system_prompt = """\
You are a report generation assistant tasked with producing a well-formatted report given parsed context.
You will be given context from one or more reports that take the form of parsed text + tables
You are responsible for producing a report with interleaving text and tables - in the format of interleaving text and "table" blocks.

Make sure the report is detailed with a lot of textual explanations especially if tables are given.

You MUST output your response as a tool call in order to adhere to the required output format. Do NOT give back normal text.

Here is an example of a toy valid tool call - note the text and table block:
```
{
    "blocks": [
        {
            "text": "A report on cities"
        },
        {
            "caption": "Comparison of CityA vs. CityB",
            "col_names": [
              "",
              "Population",
              "Country",
            ],
            "rows": [
              [
                "CityA",
                "1,000,000",
                "USA"
              ],
              [
                "CityB",
                "2,000,000",
                "Mexico"
              ]
            ]
        }
    ]
}
```
"""

report_gen_llm = OpenAI(model="gpt-4o-mini", max_tokens=2048, system_prompt=report_gen_system_prompt)
report_gen_sllm = report_gen_llm.as_structured_llm(output_cls=ReportOutput)

from llama_index.core.workflow import Workflow

from typing import Any, List
from operator import itemgetter

from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.structured_llm import StructuredLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage
from llama_index.core.tools.types import BaseTool
from llama_index.core.tools import ToolSelection
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, Context, step
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import TreeSummarize, CompactAndRefine
from llama_index.core.workflow import Event


class InputEvent(Event):
    input: List[ChatMessage]


class ChunkRetrievalEvent(Event):
    tool_call: ToolSelection


class DocRetrievalEvent(Event):
    tool_call: ToolSelection


class ReportGenerationEvent(Event):
    pass


class ReportGenerationAgent(Workflow):
    """Report generation agent."""

    def __init__(
        self,
        chunk_retriever_tool: BaseTool,
        doc_retriever_tool: BaseTool,
        llm: FunctionCallingLLM | None = None,
        report_gen_sllm: StructuredLLM | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.chunk_retriever_tool = chunk_retriever_tool
        self.doc_retriever_tool = doc_retriever_tool

        self.llm = llm or OpenAI()
        self.summarizer = CompactAndRefine(llm=self.llm)
        assert self.llm.metadata.is_function_calling_model

        self.report_gen_sllm = report_gen_sllm or self.llm.as_structured_llm(
            ReportOutput, system_prompt=report_gen_system_prompt
        )
        self.report_gen_summarizer = TreeSummarize(llm=self.report_gen_sllm)

        self.memory = ChatMemoryBuffer.from_defaults(llm=llm)
        self.sources = []

    @step(pass_context=True)
    async def prepare_chat_history(self, ctx: Context, ev: StartEvent) -> InputEvent:
        # clear sources
        self.sources = []

        ctx.data["stored_chunks"] = []
        ctx.data["query"] = ev.input

        # get user input
        user_input = ev.input
        user_msg = ChatMessage(role="user", content=user_input)
        self.memory.put(user_msg)

        # get chat history
        chat_history = self.memory.get()
        return InputEvent(input=chat_history)

    @step(pass_context=True)
    async def handle_llm_input(
        self, ctx: Context, ev: InputEvent
    ) -> ChunkRetrievalEvent | DocRetrievalEvent | ReportGenerationEvent | StopEvent:
        chat_history = ev.input

        response = await self.llm.achat_with_tools(
            [self.chunk_retriever_tool, self.doc_retriever_tool],
            chat_history=chat_history,
        )
        self.memory.put(response.message)

        tool_calls = self.llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )
        for tool_call in tool_calls:
            if self._verbose:
                print(f"Tool call: {tool_call}")
        if not tool_calls:
            # all the content should be stored in the context, so just pass along input
            return ReportGenerationEvent(input=ev.input)

        for tool_call in tool_calls:
            if tool_call.tool_name == self.chunk_retriever_tool.metadata.name:
                return ChunkRetrievalEvent(tool_call=tool_call)
            elif tool_call.tool_name == self.doc_retriever_tool.metadata.name:
                return DocRetrievalEvent(tool_call=tool_call)
            else:
                return StopEvent(result={"response": "Invalid tool."})

    @step(pass_context=True)
    async def handle_retrieval(
        self, ctx: Context, ev: ChunkRetrievalEvent | DocRetrievalEvent
    ) -> InputEvent:
        """Handle retrieval.

        Store retrieved chunks, and go back to agent reasoning loop.

        """
        query = ev.tool_call.tool_kwargs["query"]
        if isinstance(ev, ChunkRetrievalEvent):
            retrieved_chunks = self.chunk_retriever_tool(query).raw_output
        else:
            retrieved_chunks = self.doc_retriever_tool(query).raw_output
        ctx.data["stored_chunks"].extend(retrieved_chunks)

        # synthesize an answer given the query to return to the LLM.
        response = self.summarizer.synthesize(query, nodes=retrieved_chunks)
        self.memory.put(
            ChatMessage(
                role="tool",
                content=str(response),
                additional_kwargs={
                    "tool_call_id": ev.tool_call.tool_id,
                    "name": ev.tool_call.tool_name,
                },
            )
        )

        # send input event back with updated chat history
        return InputEvent(input=self.memory.get())

    @step(pass_context=True)
    async def generate_report(
        self, ctx: Context, ev: ReportGenerationEvent
    ) -> StopEvent:
        """Generate report."""
        # given all the context, generate query
        response = self.report_gen_summarizer.synthesize(
            ctx.data["query"], nodes=ctx.data["stored_chunks"]
        )

        return StopEvent(result={"response": response})
    
agent = ReportGenerationAgent(
    chunk_retriever_tool,
    doc_retriever_tool,
    llm=llm,
    report_gen_sllm=report_gen_sllm,
    verbose=True,
    timeout=120.0,
)

async def run_agent():
    ret = await agent.run(
        input="Tell me about the top-level assets and liabilities for Tesla in 2021, and compare it against those of Apple in 2021. Which company is doing better?"
    )
    return ret

if __name__ == "__main__":
    result = asyncio.run(run_agent())
    print(result["response"].response.render())