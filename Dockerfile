# Hugging Face Spaces (Docker SDK) image for the 1% Club Finance Assistant.
# Runs both services from one container: the FastAPI backend (chain/qa_chain.py
# wrapped by backend/main.py) bound to localhost only, and the Streamlit
# frontend bound to the port Spaces expects to be exposed publicly (7860).
# See start.sh for how the two processes are launched together.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x start.sh

ENV PYTHONUNBUFFERED=1
EXPOSE 7860

CMD ["./start.sh"]
