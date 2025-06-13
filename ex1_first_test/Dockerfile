# Utilizza una base Python ufficiale
FROM python:3.11-slim

WORKDIR /app

# Copia requirements e installa dipendenze
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copia i file dell'applicazione
COPY main.py ./
COPY intro.env ./
COPY data ./data

# Comando di default: accetta domanda da CLI
ENTRYPOINT ["python", "main.py"]
