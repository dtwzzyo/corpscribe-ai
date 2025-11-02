from flask import Flask, request, jsonify
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import logging

# --- НАСТРОЙКИ ---
# Настраиваем логирование, чтобы видеть запросы в консоли
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


PC_IP_ADDRESS = "http://192.168.0.15:11434"

DB_PATH = "./chroma_db"
MODEL_NAME = "mistral"
# --- КОНЕЦ НАСТРОЕК ---


# --- ИНИЦИАЛИЗАЦИЯ RAG-СИСТЕМЫ (делаем один раз при старте) ---
app = Flask(__name__)
qa_chain = None

def initialize_rag_chain():
    """
    Эта функция загружает все компоненты RAG и собирает их в цепочку.
    Она будет вызвана один раз при первом запросе к API.
    """
    global qa_chain
    if qa_chain is not None:
        logging.info("RAG-цепочка уже инициализирована.")
        return

    logging.info("Начинаю инициализацию RAG-цепочки...")
    
    llm = Ollama(base_url=PC_IP_ADDRESS, model=MODEL_NAME)
    embeddings = OllamaEmbeddings(model=MODEL_NAME, base_url=PC_IP_ADDRESS)
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_type="mmr")
    
    template = """
    Ты — умный и вежливый ассистент компании.
    Твоя задача — отвечать на вопросы, основываясь ИСКЛЮЧИТЕЛЬНО на предоставленном контексте.
    Если в контексте нет ответа на вопрос, вежливо скажи, что не можешь ответить на основе имеющихся данных.
    Не придумывай ничего от себя.

    Контекст: {context}
    Вопрос: {question}
    Ответ:
    """
    prompt = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    logging.info("✅ RAG-цепочка успешно инициализирована!")

# --- ЭНДПОИНТ API ---
@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Основной эндпоинт, который принимает вопросы.
    """
    # 1. Инициализируем RAG, если это первый запуск
    if qa_chain is None:
        initialize_rag_chain()

    # 2. Получаем JSON из тела запроса
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "В теле запроса отсутствует ключ 'question'"}), 400
    
    question = data['question']
    logging.info(f"Получен вопрос: '{question}'")

    # 3. Выполняем запрос к RAG-цепочке
    try:
        logging.info("Отправляю запрос в RAG-цепочку...")
        result = qa_chain({"query": question})
        
        answer = result.get("result", "Не удалось получить ответ.")
        source_documents = result.get("source_documents", [])
        
        # Форматируем источники для красивого вывода
        sources = []
        if source_documents:
            for doc in source_documents:
                sources.append({
                    "source": doc.metadata.get('source', 'Неизвестно'),
                    "content_preview": doc.page_content[:100] + "..."
                })

        logging.info(f"Сформирован ответ: '{answer}'")
        # 4. Возвращаем ответ в формате JSON
        return jsonify({
            "answer": answer,
            "sources": sources
        })

    except Exception as e:
        logging.error(f"Произошла ошибка при обработке запроса: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500


# --- ЗАПУСК СЕРВЕРА ---
if __name__ == '__main__':
    # Запускаем сервер. host='0.0.0.0' делает его доступным в локальной сети.
    app.run(host='0.0.0.0', port=5001, debug=True)
