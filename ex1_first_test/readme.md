# AI Agent CLI Container

Questa applicazione permette di interrogare un agente AI tramite CLI, utilizzando documenti PDF forniti dall'utente. L'app gira in un container Docker e accede ai PDF tramite una cartella locale montata come volume.

## Requisiti
- Docker
- Python 3.11 (solo per sviluppo/test locale)

## Variabili d'ambiente
- `DOCS_DIR`: **(obbligatoria)** Path assoluto della cartella locale (host) contenente i PDF, montata come volume nel container.
- Altre variabili d'ambiente richieste per Azure OpenAI devono essere definite in `intro.env` o passate al container.

## Esempio di build e run

1. **Build dell'immagine**

```powershell
docker build -t ai-agent-app ex1_first_test
```

2. **Esecuzione del container con volume e variabili**

```powershell
docker run --rm -e DOCS_DIR=/data \
  -e AZURE_OPENAI_API_KEY=... \
  -e AZURE_OPENAI_ENDPOINT=... \
  -e AZURE_OPENAI_DEPLOYMENT_NAME=... \
  -e AZURE_OPENAI_API_VERSION=... \
  -e AZURE_OPENAI_EMBEDDINGS_ENDPOINT=... \
  -e AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME=... \
  -e AZURE_OPENAI_EMBEDDINGS_API_VERSION=... \
  -v C:\Percorso\Locale\PDF:/data \
  ai-agent-app "La tua domanda qui"
```

## Testing

Per eseguire i test unitari:

```powershell
cd ex1_first_test
pytest
```

## Note
- La cartella locale dei PDF deve essere montata come volume Docker e il path deve essere fornito tramite `DOCS_DIR`.
- Le credenziali e i parametri sensibili vanno gestiti tramite variabili d'ambiente o file `.env`.
- Il vector store SQL non è più richiesto: tutta la ricerca avviene in memoria sui PDF caricati.
