from fastapi.testclient import TestClient

PDF_BYTES = b"%PDF-1.4\n% test\n"
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 100


def logged_in_client():
    from app.main import create_app
    client = TestClient(create_app(start_worker=False))
    client.__enter__()
    client.post("/login", data={"username": "test", "password": "geheim"})
    return client
