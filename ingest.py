import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings


PC_IP_ADDRESS = "http://192.168.0.15:11434"

DATA_PATH = "./docs"
DB_PATH = "./chroma_db"
MODEL_NAME = "mistral"



def create_vector_db():
    print("Инициализация процесса создания векторной базы...")


    loader = DirectoryLoader(DATA_PATH, glob='**/*.txt', loader_cls=TextLoader, recursive=True, show_progress=True)

    documents = loader.load()
    print(f"Загружено {len(documents)} текстовых документов.")


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)
    texts = text_splitter.split_documents(documents)
    print(f"Документы нарезаны на {len(texts)} чанков.")


    print("Инициализация модели эмбеддингов...")
    embeddings = OllamaEmbeddings(model=MODEL_NAME, base_url=PC_IP_ADDRESS)
    

    print("Создание эмбеддингов и сохранение в векторную базу (это может занять время)...")
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    

    vectorstore.persist()
    
    print(f"\n✅ Векторная база успешно создана и сохранена в '{DB_PATH}'.")
    print(f"Всего векторов в базе: {vectorstore._collection.count()}")


if __name__ == "__main__":
    create_vector_db()
