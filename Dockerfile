FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY . .

RUN uv sync

EXPOSE 7860

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
