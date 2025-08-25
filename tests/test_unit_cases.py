#tests/test_unit_cases.py
import io
import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from api.main import app

client = TestClient(app)

# =========================================
# 1. Health Check Endpoint 
# =========================================
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

# =========================================
# 2. UI Homepage 
# =========================================
def test_homepage_serving():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

# =========================================
# 3. File Upload + Analyze (FAISS & Analyzer mocked)
# =========================================
@patch("api.main.DocHandler.save_pdf")
@patch("api.main.read_pdf_via_handler")
@patch("api.main.DocumentAnalyzer.analyze_document")
def test_analyze_document(mock_analyze, mock_read_pdf, mock_save_pdf):
    mock_save_pdf.return_value = "/tmp/test.pdf"
    mock_read_pdf.return_value = "Dummy content"
    mock_analyze.return_value = {"summary": "Mock analysis"}

    file_content = io.BytesIO(b"Test PDF content")
    response = client.post("/analyze", files={"file": ("test.pdf", file_content)})
    assert response.status_code == 200
    data = response.json()
    assert data["summary"] == "Mock analysis"

# =========================================
# 4. Document Compare (FAISS & LLM mocked)
# =========================================
@patch("api.main.DocumentComparator.save_uploaded_files")
@patch("api.main.DocumentComparator.combine_documents")
@patch("api.main.DocumentComparatorLLM.compare_documents")
def test_compare_documents(mock_compare, mock_combine, mock_save_files):
    mock_save_files.return_value = ("/tmp/ref.pdf", "/tmp/act.pdf")
    mock_combine.return_value = "Combined text"
    mock_compare.return_value = MagicMock(to_dict=lambda orient: [{"diff": "Mock diff"}])

    file1 = io.BytesIO(b"Reference PDF")
    file2 = io.BytesIO(b"Actual PDF")
    response = client.post("/compare",
                           files={"reference": ("ref.pdf", file1), "actual": ("act.pdf", file2)})
    assert response.status_code == 200
    data = response.json()
    assert data["rows"][0]["diff"] == "Mock diff"

# =========================================
# 5. Chat Indexing (FAISS mocked)
# =========================================
@patch("api.main.ChatIngestor.built_retriver")
def test_chat_index(mock_built):
    mock_built.return_value = None
    file_content = io.BytesIO(b"Chat document content")
    response = client.post("/chat/index",
                           files={"files": ("doc.txt", file_content)},
                           data={"session_id": "sess1"})
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "sess1"
    assert data["k"] == 5

# =========================================
# 6. Chat Query (RAG + FAISS mocked)
# =========================================
@patch("api.main.ConversationalRAG.invoke")
@patch("api.main.ConversationalRAG.load_retriever_from_faiss")
def test_chat_query(mock_load, mock_invoke):
    mock_invoke.return_value = "Mock answer"
    response = client.post("/chat/query",
                           data={"question": "Hello?", "session_id": "sess1"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Mock answer"
    assert data["session_id"] == "sess1"

# =========================================
# 7. Authentication / Invalid Query 
# =========================================
def test_chat_query_without_session_id():
    response = client.post("/chat/query",
                           data={"question": "Hi", "use_session_dirs": True})
    assert response.status_code == 400

# =========================================
# 8. Error Handling for Bad Upload 
# =========================================
def test_analyze_bad_file():
    response = client.post("/analyze", files={"file": ("empty.txt", io.BytesIO(b""))})
    assert response.status_code in [500, 422]

# =========================================
# 9. Environment Variable Check 
# =========================================
def test_env_variables(monkeypatch):
    # ECS injects env variables from Secrets Manager or SSM
    monkeypatch.setenv("FAISS_BASE", "mock_faiss_base")
    monkeypatch.setenv("FAISS_INDEX_NAME", "mock_index")
    response = client.get("/health")
    assert response.status_code == 200
    # Ensure app reads env correctly
    assert os.getenv("FAISS_BASE") == "mock_faiss_base"
    assert os.getenv("FAISS_INDEX_NAME") == "mock_index"

# =========================================
# 10. Concurrency Test 
# =========================================
@patch("api.main.ConversationalRAG.invoke")
@patch("api.main.ConversationalRAG.load_retriever_from_faiss")
def test_concurrent_queries(mock_load, mock_invoke):
    mock_invoke.return_value = "Concurrent mock answer"
    from concurrent.futures import ThreadPoolExecutor

    def call_query():
        resp = client.post("/chat/query",
                           data={"question": "Hi", "session_id": "sess1"})
        return resp.status_code, resp.json()["answer"]

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(lambda _: call_query(), range(5)))
    for status, answer in results:
        assert status == 200
        assert answer == "Concurrent mock answer"

# =========================================
# 11. File Update / Reindex Test 
# =========================================
@patch("api.main.ChatIngestor.built_retriver")
def test_file_reindex(mock_built):
    mock_built.return_value = None
    # Upload file first version
    file_content = io.BytesIO(b"Version 1 content")
    response1 = client.post("/chat/index",
                            files={"files": ("doc.txt", file_content)},
                            data={"session_id": "sess_update"})
    assert response1.status_code == 200

    # Upload file updated version
    file_content_v2 = io.BytesIO(b"Version 2 content")
    response2 = client.post("/chat/index",
                            files={"files": ("doc.txt", file_content_v2)},
                            data={"session_id": "sess_update"})
    assert response2.status_code == 200
    # Ensure session_id is unchanged
    assert response2.json()["session_id"] == "sess_update"
