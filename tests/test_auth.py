from fastapi.testclient import TestClient


def make_client():
    from app.main import create_app
    return TestClient(create_app())


def test_health_is_public():
    with make_client() as c:
        assert c.get("/health").json() == {"status": "ok"}


def test_index_redirects_to_login():
    with make_client() as c:
        r = c.get("/", follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/login"


def test_login_wrong_password():
    with make_client() as c:
        r = c.post("/login", data={"username": "test", "password": "falsch"})
        assert r.status_code == 401
        assert "falsch" in r.text


def test_login_ok():
    with make_client() as c:
        r = c.post("/login", data={"username": "test", "password": "geheim"}, follow_redirects=False)
        assert r.status_code == 303
        assert c.get("/").status_code == 200
