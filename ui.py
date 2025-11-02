import streamlit as st
import requests
import json


API_URL = "http://127.0.0.1:5001/ask" # URL Flask API




def query_api(question):
    """Отправляет вопрос на API и возвращает ответ."""
    try:
        payload = {"question": question}
        headers = {"Content-Type": "application/json"}
        response = requests.post(API_URL, data=json.dumps(payload), headers=headers)
        
        # Проверяем, успешен ли запрос
        if response.status_code == 200:
            return response.json()
        else:
            # Возвращаем информацию об ошибке, если что-то пошло не так
            return {"error": f"Ошибка сервера: {response.status_code}", "details": response.text}
    except requests.exceptions.ConnectionError:
        return {"error": "Не удалось подключиться к API. Убедитесь, что сервер app.py запущен."}
    except Exception as e:
        return {"error": f"Произошла непредвиденная ошибка: {e}"}



# 1. Заголовок страницы
st.set_page_config(page_title="CorpScribe AI", page_icon="🤖")
st.title("🤖 CorpScribe AI")
st.caption("Ваш умный ассистент по корпоративной базе знаний")

# 2. Инициализация истории чата в сессии
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Отображение сообщений из истории
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Поле для ввода нового сообщения
if prompt := st.chat_input("Задайте ваш вопрос по документам..."):
    # Добавляем сообщение пользователя в историю и отображаем его
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Получаем ответ от ассистента и отображаем его
    with st.chat_message("assistant"):
        # Показываем "индикатор загрузки" пока ждем ответ
        with st.spinner("Думаю..."):
            api_response = query_api(prompt)

            if "error" in api_response:
                # Если API вернул ошибку, показываем ее
                response_text = f"Произошла ошибка: {api_response['error']}"
                if 'details' in api_response:
                    response_text += f"\n\nДетали: `{api_response['details']}`"
            else:
                # Если все хорошо, форматируем и показываем ответ
                answer = api_response.get("answer", "Не удалось получить ответ.")
                sources = api_response.get("sources", [])
                
                response_text = answer
                if sources:
                    response_text += "\n\n**Источники:**\n"
                    for i, source in enumerate(sources, 1):
                        source_info = source.get('source', 'Неизвестно')
                        # Мы используем st.expander, чтобы источники не загромождали чат
                        with st.expander(f"Источник {i}: {source_info}"):
                            st.write(source.get('content_preview', 'Нет предпросмотра.'))
            
            st.markdown(response_text)

    # Добавляем ответ ассистента в историю
    st.session_state.messages.append({"role": "assistant", "content": response_text})
