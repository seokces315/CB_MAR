import os
from crewai_tools import LlamaIndexTool
from llama_index.embeddings.openai import OpenAIEmbedding
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import ReActAgent
from langchain.memory import ConversationBufferMemory
from llama_index.core.langchain_helpers.agents import IndexToolConfig
from llama_index.core.langchain_helpers.agents import LlamaIndexTool as LlamaIndexTool2
from dotenv import load_dotenv
from llama_index.core import Document

load_dotenv()

# API 키 직접 지정
os.environ["OPENAI_API_KEY"] = ""

embed_model = OpenAIEmbedding()
client = qdrant_client.QdrantClient(location=":memory:")
vector_store = QdrantVectorStore(client=client, collection_name="test_store")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

llm = OpenAI(model="gpt-4", temperature=0.1)

Settings.llm = llm
Settings.embed_model = embed_model


def load_data(files_directory):
    reader = SimpleDirectoryReader(input_dir=files_directory, recursive=True)
    docs = reader.load_data()
    if docs:
        # Execute pipeline and time the process
        index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
        return index
    else:
        return None


def load_single_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    doc = Document(text=content)
    index = VectorStoreIndex.from_documents([doc])
    return index


index = load_single_file("docs_cardio.txt")

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,
)

response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize",
)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# 먼저 QueryEngineTool 직접 구성
query_engine_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata={
        "name": "RAG Tool",
        "description": "Use this to retrieve from indexed medical documents.",
    },
)
# 그런 다음 LlamaIndexTool에 넘김
tool = LlamaIndexTool(
    name="RAG Tool",
    description="Tool to query the clinical documents.",
    llama_index_tool=query_engine_tool,
)

# tool_config = IndexToolConfig(
#     query_engine=query_engine,
#     name=f"Vector Index",
#     description=f"useful for when you want to answer queries about the document",
#     tool_kwargs={"return_direct": True},
#     memory=memory,
# )

# # create the tool
# tool = LlamaIndexTool.from_tool_config(tool_config)


def generate_report(class_mapping, predicted_class, concepts, contribution):
    prompt = f"""
        Generate a detailed medical report based on the below information and knowledge from the documents.

        Disease: {class_mapping[predicted_class]}

        The disease was detected based on the analysis of chest x-ray images. The concepts and their corresponding contributions to the classification are as follows:

        Concepts: {concepts}
        Contributions: {contribution}

        The higher values of contributions indicate the importance of the concepts in the classification of the disease. Negative values of contribution means the concept does not contribute for the classification.
        The disease has been detected based on the presence of these concepts.

        For the report, focus on the concepts that have the highest contributions to the classification of the disease. Discuss the top 10 contributing concepts and their relevance to the disease, and where are they found.
        Also discuss the relevance of these concepts in the diagnosis and treatment of the disease.

        The findings based on the concepts should be explained in detail in medical terminology.

        In the report, also provide a brief summary of the disease, including its symptoms, causes, medical diagnosis, treatment options, and prognosis. Make sure to write the report in proper clinical and medical terms and then explain all medical terms in a way that a person with no medical background can understand. Also, include information about any relevant clinical trials or research studies, and any potential risks or complications associated with the disease.

        Finally, provide a section on prevention and self-care measures that a person with the disease can take to improve their health and well-being.
    """

    report = tool.run(prompt)
    return report
