# Шаг 1: Берем "мощный" базовый образ с уже установленным Python и PyTorch
FROM pytorch/pytorch:2.3.1-cpu-py3.9

# Шаг 2: Устанавливаем рабочую директорию
WORKDIR /app

# Шаг 3: Копируем ТОЛЬКО requirements.txt
COPY requirements.txt .

# Шаг 4: Устанавливаем НАШИ библиотеки.
# PyTorch уже есть, так что pip установит только то, чего не хватает. Это гораздо быстрее и легче.
RUN pip install --no-cache-dir -r requirements.txt

# Шаг 5: Копируем весь остальной код
COPY . .

# Шаг 6: Открываем порты
EXPOSE 5001
EXPOSE 8501

# Шаг 7: Запускаем приложение
CMD ["/bin/bash", "-c", "python app.py & streamlit run ui.py --server.port 8501 --server.address 0.0.0.0"]