from types import SimpleNamespace

import pytest

import app.app as liver_app


def test_firebase_auth_request_sends_password_to_firebase_only(monkeypatch):
    monkeypatch.setenv("FIREBASE_API_KEY", "test-api-key")
    posted = {}

    def fake_post(url, *, params, json, timeout):
        posted.update(url=url, params=params, json=json, timeout=timeout)
        return SimpleNamespace(ok=True, json=lambda: {"localId": "user-1", "idToken": "token"})

    monkeypatch.setattr(liver_app.requests, "post", fake_post)
    password = "test-password"
    response = liver_app.firebase_auth_request(
        "accounts:signInWithPassword",
        {"email": "patient@example.com", "password": password, "returnSecureToken": True},
    )

    assert response["localId"] == "user-1"
    assert posted["params"] == {"key": "test-api-key"}
    assert posted["json"]["password"] == password


def test_firebase_auth_request_reports_invalid_credentials(monkeypatch):
    monkeypatch.setenv("FIREBASE_API_KEY", "test-api-key")
    monkeypatch.setattr(
        liver_app.requests,
        "post",
        lambda *args, **kwargs: SimpleNamespace(
            ok=False,
            json=lambda: {"error": {"message": "INVALID_LOGIN_CREDENTIALS"}},
        ),
    )

    with pytest.raises(RuntimeError, match="email or password is incorrect"):
        liver_app.firebase_auth_request(
            "accounts:signInWithPassword",
            {"email": "patient@example.com", "password": "wrong-password"},
        )


def test_demo_session_does_not_use_firebase_refresh(monkeypatch):
    session_state = {}
    monkeypatch.setattr(liver_app.st, "session_state", session_state)
    monkeypatch.setattr(
        liver_app.requests,
        "post",
        lambda *args, **kwargs: pytest.fail("Demo session must not call Firebase"),
    )

    liver_app.save_demo_session("demo@example.com", "Doctor")
    liver_app.refresh_firebase_session()

    assert session_state == {
        "authenticated": True,
        "auth_mode": "demo",
        "user_role": "Doctor",
        "user_email": "demo@example.com",
    }


def test_firebase_user_role_requires_firestore_role(monkeypatch):
    monkeypatch.setenv("FIREBASE_API_KEY", "test-api-key")
    monkeypatch.setenv("FIREBASE_PROJECT_ID", "test-project")
    monkeypatch.setattr(
        liver_app.requests,
        "get",
        lambda *args, **kwargs: SimpleNamespace(
            status_code=200,
            ok=True,
            json=lambda: {"fields": {"role": {"stringValue": "patient"}}},
        ),
    )

    role = liver_app.firebase_user_role({"localId": "user-1", "idToken": "token"})

    assert role == "patient"


def test_normalized_role_accepts_firestore_lowercase(monkeypatch):
    monkeypatch.setattr(liver_app.st, "session_state", {"user_role": "patient"})

    assert liver_app.normalized_role() == "patient"


def test_patient_report_query_is_scoped_to_firebase_uid(monkeypatch):
    monkeypatch.setenv("FIREBASE_API_KEY", "test-api-key")
    monkeypatch.setenv("FIREBASE_PROJECT_ID", "test-project")
    monkeypatch.setattr(
        liver_app.st,
        "session_state",
        {
            "auth_mode": "firebase",
            "user_role": "patient",
            "firebase_uid": "patient-uid",
            "firebase_id_token": "id-token",
        },
    )
    posted = {}

    def fake_post(url, *, headers, json, timeout):
        posted.update(url=url, headers=headers, json=json, timeout=timeout)
        return SimpleNamespace(ok=True, json=lambda: [])

    monkeypatch.setattr(liver_app.requests, "post", fake_post)

    assert liver_app.fetch_firebase_reports() == []
    query = posted["json"]["structuredQuery"]
    assert query["where"]["fieldFilter"]["value"]["stringValue"] == "patient-uid"
    assert posted["headers"]["Authorization"] == "Bearer id-token"