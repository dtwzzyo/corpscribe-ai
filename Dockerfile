# Шаг 1: Берем стандартный, надежный образ Python 3.9
FROM python:3.9

# Шаг 2: Устанавливаем рабочую директорию
WORKDIR /app

# Шаг 3: Устанавливаем "тяжеловесов" отдельной командой.
RUN pip install --no-cache-dir --prefer-binary \
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Шаг 4: Копируем наш список остальных зависимостей
COPY requirements.txt .

# Шаг 5: Устанавливаем все остальное.
RUN pip install --no-cache-dir -r requirements.txt

# Шаг 6: Копируем весь код приложения
COPY . .

# Шаг 7: Устанавливаем Gunicorn (наш production-сервер для Flask)
RUN pip install gunicorn

# Шаг 8: Открываем порты
EXPOSE 5001
EXPOSE 8501

# Шаг 9: <<< ГЛАВНОЕ ИЗМЕНЕНИЕ: Поочередный запуск >>>
# Создаем маленький скрипт-запускатор
RUN echo '#!/bin/bash' > /app/start.sh && \
    echo 'echo "Starting Gunicorn (Flask API)..."' >> /app/start.sh && \
    echo 'gunicorn --bind 0.0.0.0:5001 --workers 2 app:app &' >> /app/start.sh && \
    echo 'sleep 10' >> /app/start.sh && \
    echo 'echo "Starting Streamlit UI..."' >> /app/start.sh && \
    echo 'streamlit run ui.py --server.port 8501 --server.address 0.0.0.0 --server.fileWatcherType none' >> /app/start.sh && \
    chmod +x /app/start.sh

# Запускаем наш скрипт
CMD ["/app/start.sh"]