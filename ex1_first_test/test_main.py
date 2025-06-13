import os
import pytest
from unittest.mock import MagicMock
from main import load_and_split_docs

class DummyVectorStore:
    def add_documents(self, documents):
        self.docs = documents

class DummyLoader:
    def __init__(self, docs):
        self._docs = docs
    def load(self):
        return self._docs

class DummyTextSplitter:
    def __init__(self, splits):
        self._splits = splits
    def split_documents(self, docs):
        return self._splits

# Patch PyPDFDirectoryLoader e RecursiveCharacterTextSplitter per test isolati
def test_load_and_split_docs(monkeypatch):
    os.environ['DOCS_DIR'] = '/tmp/fakepdfs'
    dummy_docs = ['doc1', 'doc2']
    dummy_splits = ['split1', 'split2', 'split3']
    dummy_vector_store = DummyVectorStore()

    monkeypatch.setattr('main.PyPDFDirectoryLoader', lambda path: DummyLoader(dummy_docs))
    monkeypatch.setattr('main.RecursiveCharacterTextSplitter', lambda **kwargs: DummyTextSplitter(dummy_splits))

    result = load_and_split_docs(dummy_vector_store)
    assert result == dummy_splits
    assert dummy_vector_store.docs == dummy_splits

def test_load_and_split_docs_no_env(monkeypatch):
    if 'DOCS_DIR' in os.environ:
        del os.environ['DOCS_DIR']
    dummy_vector_store = DummyVectorStore()
    with pytest.raises(ValueError):
        load_and_split_docs(dummy_vector_store)
