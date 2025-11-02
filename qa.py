# <<< ИЗМЕНЕНИЕ ЗДЕСЬ: Импортируем Ollama из правильного места
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


PC_IP_ADDRESS = "http://192.168.0.15:11434"

DB_PATH = "./chroma_db"
MODEL_NAME = "mistral"




def main():
    print("Инициализация системы Вопрос-Ответ...")

    llm = Ollama(
        base_url=PC_IP_ADDRESS,
        model=MODEL_NAME
    )
    print(f"✅ Соединение с моделью '{MODEL_NAME}' на сервере установлено.")


    embeddings = OllamaEmbeddings(model=MODEL_NAME, base_url=PC_IP_ADDRESS)


    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    print("✅ Векторная база загружена.")


    retriever = vectorstore.as_retriever(search_type="mmr")
    print("✅ Ретривер готов к работе.")

    # 5. Создаем шаблон промпта.
    #    Это инструкция для LLM, как именно отвечать на вопрос.
    #    {context} - сюда LangChain подставит найденные в базе документы.
    #    {question} - сюда LangChain подставит наш вопрос.
    template = """
    Ты — умный и вежливый ассистент компании.
    Твоя задача — отвечать на вопросы, основываясь ИСКЛЮЧИТЕЛЬНО на предоставленном контексте.
    Если в контексте нет ответа на вопрос, вежливо скажи, что не можешь ответить на основе имеющихся данных.
    Не придумывай ничего от себя.

    Контекст:
    {context}

    Вопрос:
    {question}

    Ответ:
    """
    prompt = PromptTemplate.from_template(template)

    # 6. Собираем все вместе в единую цепочку (QA Chain)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    print("✅ Система готова к приему вопросов.\n" + "="*50)


    question = "Где ведется вся документация?" 

    print(f"💬 Ваш вопрос: {question}\n")
    print("🤔 Думаю над ответом...")

    result = qa_chain({"query": question})
    answer = result["result"]
    source_documents = result["source_documents"]

    print("\n" + "="*50)
    print("🤖 Ответ:")
    print(answer)
    print("\n" + "="*50)

    if source_documents:
        print("📚 Источники:")
        for doc in source_documents:
            print(f"  - {doc.metadata['source']} (часть текста: \"{doc.page_content[:100]}...\")")

if __name__ == "__main__":
    main()
