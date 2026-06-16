from pathlib import Path


def test_vectorstore_exists():
    assert Path("vectorstore/faiss_index/index.faiss").exists()
    assert Path("vectorstore/faiss_index/index.pkl").exists()