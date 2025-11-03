import streamlit as st
import requests
import json
import os

# --- (Все функции для API остаются без изменений) ---
API_BASE_URL = "http://127.0.0.1:5001"
def get_documents():
    try:
        response = requests.get(f"{API_BASE_URL}/documents")
        if response.status_code == 200: return response.json()
    except: pass
    return None
def upload_document(file):
    try:
        files = {'file': (file.name, file.getvalue(), file.type)}
        response = requests.post(f"{API_BASE_URL}/upload", files=files)
        return response.json()
    except: return {"error": "Не удалось подключиться к API."}
def delete_document(filename):
    try:
        response = requests.delete(f"{API_BASE_URL}/documents/{filename}")
        return response.json()
    except: return {"error": "Не удалось подключиться к API."}
def rebuild_index():
    try:
        response = requests.post(f"{API_BASE_URL}/rebuild")
        return response.json()
    except: return {"error": "Не удалось подключиться к API."}
def query_api(question):
    try:
        response = requests.post(f"{API_BASE_URL}/ask", json={"question": question})
        if response.status_code == 200: return response.json()
        return {"error": f"Ошибка сервера: {response.status_code}", "details": response.text}
    except: return {"error": "Не удалось подключиться к API."}


# --- (Основная структура приложения - без изменений) ---
st.set_page_config(page_title="CorpScribe AI", page_icon="🤖", layout="wide")
with st.sidebar:
    st.title("Панель Управления")
    page = st.radio("Выберите страницу", ["Чат с Ассистентом", "Управление Базой Знаний"])
    api_status = get_documents()
    if api_status is not None: st.success("✅ API-сервер доступен")
    else: st.error("❌ API-сервер недоступен!")

# --- (Страница чата - без изменений) ---
if page == "Чат с Ассистентом":
    st.title("🤖 CorpScribe AI")
    # ... (весь код чата остается тем же)
    if "messages" not in st.session_state: st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])
    if prompt := st.chat_input("Задайте ваш вопрос..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Думаю..."):
                response_data = query_api(prompt)
                answer = response_data.get("answer", "Произошла ошибка.")
                st.markdown(answer)
                sources = response_data.get("sources", [])
                if sources:
                    with st.expander("Показать источники"):
                        for source in sources: st.info(f"Источник: {source['source']}\n\n>{source['content_preview']}")
        st.session_state.messages.append({"role": "assistant", "content": answer})

elif page == "Управление Базой Знаний":
    st.title("🗂️ Управление Базой Знаний")
    # ... (блок переиндексации без изменений)
    st.subheader("Пересобрать Базу Знаний")
    if st.button("🚀 Запустить пересборку"):
        with st.spinner("Пересобираю базу знаний..."):
            result = rebuild_index()
            st.success(result.get('message', 'Готово!'))
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Загрузить новый документ")
        uploaded_file = st.file_uploader("Выберите .txt или .pdf файл", type=['txt', 'pdf'])
        
        # --- ИЗМЕНЕНИЕ ЗДЕСЬ: Добавили кнопку для подтверждения загрузки ---
        if st.button("📤 Загрузить файл", disabled=(uploaded_file is None)):
            if uploaded_file is not None:
                with st.spinner("Загружаю файл..."):
                    result = upload_document(uploaded_file)
                    if "error" in result:
                        st.error(f"Ошибка: {result['error']}")
                    else:
                        st.success(result.get('message', 'Файл загружен!'))
                        st.info("Не забудьте пересобрать базу.")
                        st.rerun() # Обновляем страницу, чтобы показать новый файл
            else:
                st.warning("Пожалуйста, сначала выберите файл.")
    
    with col2:
        st.subheader("Текущие документы")
        documents = get_documents()
        if documents:
            for doc in documents:
                sub_col1, sub_col2 = st.columns([0.8, 0.2])
                sub_col1.text(doc)
                if sub_col2.button("🗑️", key=f"del_{doc}"):
                    delete_document(doc)
                    st.rerun()
