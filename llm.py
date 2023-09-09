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

template = """
  You are an AI Lawyer. Conversation between a human and an AI lawyer and related context are given. Use the following pieces of context to answer the question at the end. If you don't know the answer or question is not related to law, just say that you don't know, don't try to make up an answer.
  The laws have sections and subsections. A section start with number and dot (ex: "4.") and subsections are starts numbers with brackets(ex : "(2)")
  you should provide exract sections and its subsections with its numbers followed by your answer. follow below template.
  ANSWER TEMPLATE:
    [sections and its subsections related to question]
    [your answer]
  CONTEXT:
  {context}
  
  QUESTION: 
  {question}

  CHAT HISTORY:
  {chat_history}
  
  ANSWER:
  """

prompt = PromptTemplate(input_variables=["chat_history", "question", "context"], template=template)

# define embedding
embeddings = OpenAIEmbeddings(
    openai_api_key = OPENAI_KEY
)
# define memory
memory = ConversationBufferMemory(memory_key="chat_history", ai_prefix="AI Lawyer", return_messages=True)
openai = OpenAI(temperature=0, openai_api_key= OPENAI_KEY)
# db3 = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
faiss_db = FAISS.load_local("faiss_index", embeddings)


# define chain
chat_llm = ConversationalRetrievalChain.from_llm(openai, faiss_db.as_retriever(search_kwargs={"k": 3}), memory=memory,combine_docs_chain_kwargs={"prompt": prompt}, verbose=True)

def create_db(file):
    # load documents
    loader = PyPDFLoader(file)
    documents = loader.load()
    documents = documents[:16]
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=30)
    docs = text_splitter.split_documents(documents)
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss_index")
    # vectordb.persist()

def get_chat_history():
    return memory.load_memory_variables({})

def chat(question):
    chat_history= get_chat_history()
    res = chat_llm({"question": question, "chat_history": chat_history})
    # n=0
    # memory.clear()
    memory.save_context({"input": question}, {"output": res['answer']})
    return res['answer']


