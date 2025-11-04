# Шаг 1: Берем стандартный, надежный образ Python 3.9
FROM python:3.9

# Шаг 2: Устанавливаем рабочую директорию
WORKDIR /app

# Шаг 3: Устанавливаем "тяжеловесов" отдельной командой.
# Мы указываем --no-cache-dir, чтобы не засорять образ, и --prefer-binary, чтобы по возможности
# использовать готовые "колеса", а не компилировать с нуля.
# Также указываем URL для скачивания, чтобы pip не тратил время на поиск.
RUN pip install --no-cache-dir --prefer-binary \
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Шаг 4: Копируем наш список остальных зависимостей
COPY requirements.txt .

# Шаг 5: Устанавливаем все остальное. Это пройдет быстро, т.к. torch уже на месте.
RUN pip install --no-cache-dir -r requirements.txt

# Шаг 6: Копируем весь код приложения
COPY . .

# Шаг 7: Открываем порты
EXPOSE 5001
EXPOSE 8501

# Шаг 8: Запускаем приложение
CMD ["/bin/bash", "-c", "python app.py & streamlit run ui.py --server.port 8501 --server.address 0.0.0.0"]