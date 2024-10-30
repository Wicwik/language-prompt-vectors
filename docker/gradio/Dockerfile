FROM python:3.12-slim

RUN pip install --no-cache-dir gradio
EXPOSE 7860
COPY . .

ENV GRADIO_SERVER_NAME="0.0.0.0"

RUN ls -lad

CMD ["python", "app.py"]