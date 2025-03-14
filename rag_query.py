from typing import Union

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

load_dotenv()

# === 1.Load Chroma Vectorstore ===

vectorstore = Chroma(
    persist_directory="rag_chroma_db",
    embedding_function= OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever(search_kwargs={"k":2})

# === 2.Prompt ===

prompt = ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant. Use the provided context to answer"),
    MessagesPlaceholder(variable_name="history"),
    ("human","{question}\n\nContext:\n{context}")
])

# === 3. LLM and output parser ===

llm = ChatOpenAI(
    model = "gpt-4o-mini",
    streaming = True,
    callbacks = [StreamingStdOutCallbackHandler()],
    max_tokens = 500
)

parser = StrOutputParser()

# === 4. Format retrieved docs ===

def format_docs(docs):
    return "\n\n".join([f"---\n{doc.page_content.strip()}" for doc in docs])

# === 5. Chain with retrieval and memory input ===

def inject_context(inputs):
    question = inputs["question"]
    history =  inputs["history"]
    docs = retriever.invoke(question)
    return {
        "question" : question,
        "context"  : format_docs(docs),
        "history"  : history
    }

rag_chain = (
    RunnableLambda(inject_context) |
    prompt |
    llm |
    parser
)

# === 6. Updated memory class with add_message() ===

class Mymemory(BaseChatMessageHistory):
    def __init__(self):
        self.messages = []

    def add_user_message(self, content: str):
        self.messages.append(HumanMessage(content=content))

    def add_ai_message(self, content: str):
        self.messages.append(AIMessage(content=content))

    def add_message(self, message : BaseMessage):
        self.messages.append(message)

    def clear(self):
        self.messages = []

chat_with_memory = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id : Mymemory(),
    input_messages_key= "question",
    history_messages_key= "history"
)

# === 7. Run the chat ===

print("\n Chat with your AI! Type 'exit' to quit.")

session_id = "my_rag_session"

while True:
    question = input("\nYou: ")
    if question.strip().lower() == "exit":
        break
    print("\n AI:", end= " ")
    chat_with_memory.invoke(
         {"question": question},
        config={"configurable": {"session_id":session_id}}
    )
    show = input("\n\n Show sources chunk? (y/n)").strip().lower()
    if show == "y":
        doc = retriever.invoke(question)
        for i, doc in enumerate(doc,1):
            print(f"\n--- Source #{i} ---\n{doc.page_content[:500]}...")