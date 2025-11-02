from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# --- НАСТРОЙКИ ---
# !!! ВАЖНО: Замени этот IP на IP твоего ПК !!!
PC_IP_ADDRESS = "http://192.168.0.15:11434" # <-- ЗАМЕНИ НА СВОЙ IP

DB_PATH = "./chroma_db"
MODEL_NAME = "mistral"
# --- КОНЕЦ НАСТРОЕК ---

def debug_retriever():
    print("Запуск отладки ретривера...")

    # 1. Инициализируем эмбеддинги
    embeddings = OllamaEmbeddings(model=MODEL_NAME, base_url=PC_IP_ADDRESS)

    # 2. Загружаем нашу векторную базу
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    print("✅ Векторная база загружена.")

    # 3. Создаем ретривер. По умолчанию он ищет 4 самых похожих документа (k=4)
    retriever = vectorstore.as_retriever(search_type="mmr")
    print("✅ Ретривер создан.")

    # 4. Наш проблемный вопрос
    question = "Где ведется вся документация?"

    print(f"\nИщу документы, релевантные вопросу: '{question}'")
    
    # 5. САМОЕ ГЛАВНОЕ: получаем релевантные документы БЕЗ вызова LLM
    relevant_docs = retriever.get_relevant_documents(question)

    print("\n" + "="*50)
    if not relevant_docs:
        print("❌ Ретривер не нашел НИ ОДНОГО релевантного документа!")
    else:
        print(f"✅ Найдено {len(relevant_docs)} релевантных документов:")
        for i, doc in enumerate(relevant_docs):
            print(f"\n--- Документ #{i+1} ---")
            print(f"Источник: {doc.metadata.get('source', 'Неизвестно')}")
            print(f"Содержимое: \n{doc.page_content}")
    print("\n" + "="*50)


if __name__ == "__main__":
    debug_retriever()
