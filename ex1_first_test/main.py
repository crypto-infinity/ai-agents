import os
import argparse
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from typing_extensions import List, TypedDict

# Load environment variables
def load_env():
    load_dotenv("intro.env")

# Setup LLM and Embeddings
def setup_llm_and_embeddings():
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_VERSION"),
    )
    return llm, embeddings

# Setup Vector Store solo in memoria
def setup_vectorstore(embeddings):
    local_vector_store = InMemoryVectorStore(embeddings)
    return local_vector_store

# Load and split documents
# Usa il path assoluto fornito da DOCS_DIR (montato come volume dal comando Docker)
def load_and_split_docs(local_vector_store):
    DIR_PATH = os.getenv("DOCS_DIR")
    if not DIR_PATH:
        raise ValueError("La variabile d'ambiente DOCS_DIR deve essere impostata con il path assoluto della cartella da montare nel container.")
    loader = PyPDFDirectoryLoader(DIR_PATH)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(docs)
    local_vector_store.add_documents(documents=all_splits)
    return all_splits

# Prompt setup
def get_prompt():
    template = """
Use the following pieces of context, User History and Answer History to answer the Question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say \"thanks for asking!\" at the end of the answer.

{context}

Question: {question}

User History: {user_history}

Answer History: {answer_history}

Helpful Answer:
"""
    return PromptTemplate.from_template(template)

# State definition
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    answer_history: List[str]
    user_history: List[str]

def retrieve(state, local_vector_store):
    local_results = local_vector_store.similarity_search_with_score(state["question"])
    # Solo retrieval locale, niente SQL
    all_results = local_results
    seen = set()
    deduped_results = []
    for doc, score in all_results:
        key = doc.page_content.strip()
        if key not in seen:
            seen.add(key)
            deduped_results.append((doc, score))
    deduped_results.sort(key=lambda x: x[1])
    return {"context": [doc for doc, score in deduped_results], "results_with_score": deduped_results}

def generate(state, llm, prompt):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    prompt_text = prompt.format(
        question=state["question"],
        context=docs_content,
        user_history=state["user_history"],
        answer_history=state["answer_history"]
    )
    response = llm.invoke(prompt_text)
    answer_history = state.get("answer_history", [])
    answer_history.append(response.content)
    user_history = state.get("user_history", [])
    user_history.append(state["question"])
    return {"answer": response.content, "answer_history": answer_history, "user_history": user_history}

def main():
    import sys
    load_env()
    llm, embeddings = setup_llm_and_embeddings()
    local_vector_store = setup_vectorstore(embeddings)
    load_and_split_docs(local_vector_store)
    prompt = get_prompt()

    answer_history = []
    user_history = []

    print("Digita la tua domanda (scrivi 'exit', 'esci', 'quit', 'fine' per terminare):")
    while True:
        try:
            user_input = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            print("\nUscita per errore keyboard.")
            break
        if user_input.lower() in {"exit", "esci", "quit", "fine", "stop", "close"}:
            print("Uscita.")
            break
        if not user_input:
            continue
        state = {
            "question": user_input,
            "answer_history": answer_history,
            "user_history": user_history,
            "context": [],
            "answer": ""
        }
        retrieval = retrieve(state, local_vector_store)
        state["context"] = retrieval["context"]
        result = generate(state, llm, prompt)
        print(f'Risposta: {result["answer"]}\n')
        answer_history = result.get("answer_history", [])
        user_history = result.get("user_history", [])

if __name__ == "__main__":
    main()
