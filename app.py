from flask import Flask, request, jsonify
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import logging
import os
import shutil
from typing import Any

# --- НОВЫЕ ИМПОРТЫ ---
from langchain_deepseek import ChatDeepSeek
from langchain_openai import OpenAIEmbeddings # Используем для OpenAI-совместимых эмбеддингов

# --- Старые импорты для RAG ---
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from flashrank import Ranker, RerankRequest
from pydantic import Field

# --- НАСТРОЙКИ ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DATA_PATH = "./docs"
DB_PATH = "./chroma_db"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
LLM_MODEL_NAME = "deepseek-coder" # Эта модель будет использоваться и для чата, и для эмбеддингов
# --- КОНЕЦ НАСТРОЕК ---

app = Flask(__name__)
qa_chain = None

# --- ЛОГИКА ИНДЕКСАЦИИ (полностью на DeepSeek) ---
def create_vector_db():
    global qa_chain
    qa_chain = None
    logging.info("Начинаю полную переиндексацию базы знаний с помощью DeepSeek...")
    
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
         logging.warning("Папка 'docs' пуста. Векторная база не будет создана.")
         if not os.path.exists(DATA_PATH):
             os.makedirs(DATA_PATH)
         return False

    all_docs = []
    for filename in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, filename)
        if os.path.isfile(file_path):
            try:
                loader = UnstructuredFileLoader(file_path)
                all_docs.extend(loader.load())
            except Exception as e:
                logging.error(f"Ошибка загрузки файла {filename}: {e}")
    
    if not all_docs:
        logging.warning("Не удалось загрузить ни одного документа.")
        return False

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    texts = text_splitter.split_documents(all_docs)
    logging.info(f"Документы нарезаны на {len(texts)} чанков.")

    # <<< ЗАМЕНА OLLAMA НА DEEPSEEK ДЛЯ ЭМБЕДДИНГОВ >>>
    logging.info("Создание эмбеддингов через DeepSeek API...")
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY не найден для создания эмбеддингов!")
        
    embeddings = OpenAIEmbeddings(
        model=LLM_MODEL_NAME,
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com/v1"
    )
    
    vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=DB_PATH)
    vectorstore.persist()
    logging.info(f"✅ Векторная база успешно пересоздана. Всего векторов: {vectorstore._collection.count()}")
    return True

# --- ИНИЦИАЛИЗАЦИЯ RAG (полностью на DeepSeek) ---
def initialize_rag_chain():
    global qa_chain
    if qa_chain is not None: return
    logging.info("Инициализация RAG-цепочки с DeepSeek и Ре-ранкером...")
    
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY не найден!")
    
    llm = ChatDeepSeek(model=LLM_MODEL_NAME, api_key=DEEPSEEK_API_KEY)
    
    embeddings = OpenAIEmbeddings(
        model=LLM_MODEL_NAME,
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com/v1"
    )
    
    if not os.path.exists(DB_PATH): return
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    base_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 20})
    
    class FlashRankReranker(BaseDocumentCompressor):
        ranker: Any = Field(default_factory=lambda: Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./flashrank_models"))
        def compress_documents(self, documents, query, **kwargs):
            passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(documents)]
            rerank_request = RerankRequest(query=query, passages=passages)
            rerank_result = self.ranker.rerank(rerank_request)
            final_docs = []
            for r in rerank_result[:3]: final_docs.append(documents[r['id']])
            return final_docs

    compression_retriever = ContextualCompressionRetriever(base_compressor=FlashRankReranker(), base_retriever=base_retriever)
    
    template = """Ты — умный и вежливый ассистент. Отвечай на вопросы ИСКЛЮЧИТЕЛЬНО на основе предоставленного контекста. Если ответа нет, вежливо скажи, что не можешь ответить на основе имеющихся данных. Не придумывай ничего от себя. Контекст: {context} Вопрос: {question} Ответ:"""
    prompt = PromptTemplate.from_template(template)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=compression_retriever, return_source_documents=True, chain_type_kwargs={"prompt": prompt})
    logging.info("✅ RAG-цепочка успешно инициализирована!")

# --- ЭНДПОИНТЫ (без изменений) ---
@app.route('/documents', methods=['GET'])
def list_documents():
    if not os.path.exists(DATA_PATH): os.makedirs(DATA_PATH)
    return jsonify(os.listdir(DATA_PATH))
@app.route('/upload', methods=['POST'])
def upload_document():
    file = request.files['file']
    file.save(os.path.join(DATA_PATH, file.filename))
    return jsonify({"message": f"Файл '{file.filename}' успешно загружен."}), 201
@app.route('/documents/<path:filename>', methods=['DELETE'])
def delete_document(filename):
    file_path = os.path.join(DATA_PATH, filename)
    if os.path.exists(file_path): os.remove(file_path)
    return jsonify({"message": f"Файл '{filename}' удален."})
@app.route('/rebuild', methods=['POST'])
def rebuild_index():
    try:
        create_vector_db()
        return jsonify({"message": "Процесс пересоздания базы знаний запущен."})
    except Exception as e:
        logging.error(f"Ошибка при пересоздании базы: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
@app.route('/ask', methods=['POST'])
def ask_question():
    if not os.path.exists(DB_PATH): return jsonify({"answer": "База знаний еще не создана. Пожалуйста, загрузите документы и пересоберите базу в панели управления.", "sources": []})
    if qa_chain is None: initialize_rag_chain()
    question = request.get_json().get('question')
    result = qa_chain({"query": question})
    sources = [{"source": doc.metadata.get('source', 'N/A'), "content_preview": doc.page_content[:100] + "..."} for doc in result.get("source_documents", [])]
    return jsonify({"answer": result.get("result", "Нет ответа."), "sources": sources})

if __name__ == '__main__':
    # Оставляем это для локального тестирования, если понадобится
    app.run(host='0.0.0.0', port=5001)
