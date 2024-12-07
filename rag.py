

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_text_splitters import RecursiveCharacterTextSplitter

import faiss
import bs4
import os
from dotenv import load_dotenv
load_dotenv()



CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDINGS_DIMENSION = os.getenv("EMBEDDINGS_DIMENSION", 3072)

NUMBER_OF_CHUNKS_TO_RETRIEVE = 2
CHUNK_SIZE = 500
CHUNK_OVERLAP = 150

llm = ChatOpenAI(model=CHAT_MODEL)



embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)


try:
    EMBEDDINGS_DIMENSION = int(EMBEDDINGS_DIMENSION)
except ValueError:
    EMBEDDINGS_DIMENSION = 3072

vector_store = FAISS(
        embedding_function=embeddings,
        index=faiss.IndexFlatL2(EMBEDDINGS_DIMENSION),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )



# seperate different types of RAGs for different types of documents (currently only pdf)

def load_pdf_based_rag(content: str):
    # Load and chunk contents of the pdf
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_splits = text_splitter.split_documents([Document(page_content=content)])
    return all_splits

    
def make_vector_store(type:dict):
    
   
    if type["type"] == "pdf":
        all_splits = load_pdf_based_rag(type["content"])
    

    vector_store.add_documents(all_splits)

    






@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=NUMBER_OF_CHUNKS_TO_RETRIEVE)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs






# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1] # Reverse the string to get the most recent messages first

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)

    # Prompt Engineering Part
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}" # Context
    )

    # Current conversation messages (excluding AI tool calls)
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}



###### Manual Graph Building with Agent Only being able to perform one tool call ######

graph_builder = StateGraph(MessagesState)
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

memory = MemorySaver() # Helps in letting the llm remember what was said in the previous conversation (including the retrived context)
graph = graph_builder.compile(checkpointer=memory)

# Specify an ID for the thread
config = {"configurable": {"thread_id": "abc123"}} # used for memory saving (not too important for demonstration purposes)


def answer_question_in_console(question: str):
    

    for step in graph.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
        config=config,
    ):
        step["messages"][-1].pretty_print()

###########################################################################

### Using the prebuilt agent with the graph already built, can perform multiple tool calls ###

# from langgraph.prebuilt import create_react_agent
# memory = MemorySaver() # Helps in letting the llm remember what was said in the previous conversation (including the retrived context)
# agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory)


# config = {"configurable": {"thread_id": "def234"}} # used for memory saving (not too important for demonstration purposes)



# def answer_question(input_message: str):

#     for event in agent_executor.stream(
#         {"messages": [{"role": "user", "content": input_message}]},
#         stream_mode="values",
#         config=config,
#     ):
#         event["messages"][-1].pretty_print()

###########################################################################

if __name__ == "__main__":
    # example question
    input_message = (
        "What is the standard method for Task Decomposition?\n\n"
        "Once you get the answer, look up common extensions of that method."
    )

    while True:
        input_message = input("Ask a question: ")
        answer_question_in_console(input_message)
        print("\n")


    