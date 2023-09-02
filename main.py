from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma

# define embedding
embeddings = OpenAIEmbeddings(
    openai_api_key= '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'
)

def create_db(file):
    # load documents
    loader = PyPDFLoader(file)
    documents = loader.load()
    documents = documents[:1]
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=30)
    docs = text_splitter.split_documents(documents)
    # create vector database from data
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory='./chroma_db'
    )
    vectordb.persist()



if __name__ == "__main__":
    db3 = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    docs = db3.similarity_search("Appointment of Registrar-General and his duties.")
    print(docs[0].page_content)