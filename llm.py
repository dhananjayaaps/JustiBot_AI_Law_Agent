from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate

OPENAI_KEY= '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'

# define embedding
embeddings = OpenAIEmbeddings(
    openai_api_key = OPENAI_KEY
)
# define memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
openai = OpenAI(temperature=0, openai_api_key= OPENAI_KEY)
# db3 = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
faiss_db = FAISS.load_local("faiss_index", embeddings)

refine_eq_template = """You are the AI lawyer bot. the following is the conversation between you and human. genarate a question followed by '?' as helpful answer to gather additional information from human.

Current conversation:
{history}"""

# define chain
chat_llm = ConversationalRetrievalChain.from_llm(openai, faiss_db.as_retriever(search_kwargs={"k": 3}), memory=memory, verbose=True)
# refine_qs = ConversationalRetrievalChain.from_llm(openai, faiss_db.as_retriever(search_kwargs={"k": 3}), verbose=True)

def create_db(file):
    # load documents
    loader = PyPDFLoader(file)
    documents = loader.load()
    documents = documents[:16]
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=30)
    docs = text_splitter.split_documents(documents)
    # create vector database from data
    # vectordb = Chroma.from_documents(
    #     documents=docs,
    #     embedding=embeddings,
    #     persist_directory='./chroma_db'
    # )
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss_index")
    # vectordb.persist()

# def refineQuestion(query):
#     query = refine_eq_template.format(history=memory.load_memory_variables({})['history'])
#     memory.save_context({"outputs": query})
#     return refine_qs({"inputs": query}, {"question": query})

# refine_k = 3
# n = 0
def chat(query):
    # global n
    # if n<refine_k:
    #     res = refineQuestion(query)
    #     n=n+1
    #     return res['answer']
    res = chat_llm({"question": query})
    # n=0
    # memory.clear()
    return res['answer']
