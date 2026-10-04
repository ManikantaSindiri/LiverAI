import os
import sys
import time
import uuid
import html
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

import requests
import streamlit as st

# Add project root to Python path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from rag.clinical_guidelines import CLINICAL_KNOWLEDGE_BASE, ClinicalGuidelinesRAG
from utils.report_parser import analyze_report_text, extract_report_text


def get_firebase_status():
    """Read Firebase web configuration without exposing it in the UI."""
    names = (
        "FIREBASE_API_KEY",
        "FIREBASE_AUTH_DOMAIN",
        "FIREBASE_PROJECT_ID",
        "FIREBASE_STORAGE_BUCKET",
        "FIREBASE_MESSAGING_SENDER_ID",
        "FIREBASE_APP_ID",
    )
    firebase_config = {}
    for name in names:
        value = os.getenv(name)
        if not value:
            try:
                value = st.secrets.get(name)
            except Exception:
                value = None
        firebase_config[name] = value
    return firebase_config, bool(firebase_config["FIREBASE_API_KEY"])


def normalized_role() -> str:
    return str(st.session_state.get("user_role", "")).strip().lower()


def firebase_documents_url() -> str:
    config, configured = get_firebase_status()
    project_id = config.get("FIREBASE_PROJECT_ID")
    if not configured or not project_id:
        raise RuntimeError("Set FIREBASE_API_KEY and FIREBASE_PROJECT_ID to connect patient reports.")
    return f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/(default)/documents"


@st.cache_resource(show_spinner=False)
def demo_report_store() -> List[Dict[str, Any]]:
    """In-process report store for the explicitly non-persistent demo mode."""
    return []


def firebase_report_headers() -> Dict[str, str]:
    token = st.session_state.get("firebase_id_token")
    if not token:
        raise RuntimeError("Your Firebase session has expired. Sign in again to access reports.")
    return {"Authorization": f"Bearer {token}"}


def save_report_record(report: Dict[str, Any]) -> None:
    if st.session_state.get("auth_mode") != "firebase":
        demo_report_store().insert(0, report)
        return

    fields = {
        "patientUid": {"stringValue": report["patient_uid"]},
        "patientEmail": {"stringValue": report["patient_email"]},
        "filename": {"stringValue": report["filename"]},
        "createdAt": {"timestampValue": report["created_at"]},
        "keyPoints": {"stringValue": "\n".join(report["key_points"])},
        "plainLanguage": {"stringValue": "\n".join(report["plain_language"])},
        "sourceText": {"stringValue": report["source_text"]},
    }
    response = requests.post(
        f"{firebase_documents_url()}/reports",
        params={"documentId": report["id"]},
        headers=firebase_report_headers(),
        json={"fields": fields},
        timeout=20,
    )
    if not response.ok:
        raise RuntimeError("Could not save this report to Firestore. Check the report security rules and try again.")


def firestore_string(fields: Dict[str, Any], name: str) -> str:
    value = fields.get(name, {})
    return value.get("stringValue", value.get("timestampValue", ""))


def fetch_firebase_reports() -> List[Dict[str, Any]]:
    base_url = firebase_documents_url()
    headers = firebase_report_headers()
    if normalized_role() == "patient":
        response = requests.post(
            f"{base_url}:runQuery",
            headers=headers,
            json={
                "structuredQuery": {
                    "from": [{"collectionId": "reports"}],
                    "where": {
                        "fieldFilter": {
                            "field": {"fieldPath": "patientUid"},
                            "op": "EQUAL",
                            "value": {"stringValue": st.session_state.get("firebase_uid", "")},
                        }
                    },
                    "limit": 100,
                }
            },
            timeout=20,
        )
    else:
        response = requests.get(
            f"{base_url}/reports",
            params={"pageSize": 100},
            headers=headers,
            timeout=20,
        )
    if not response.ok:
        raise RuntimeError("Could not load reports from Firestore. Check the report security rules and try again.")

    payload = response.json()
    documents = (
        [row["document"] for row in payload if row.get("document")]
        if normalized_role() == "patient"
        else payload.get("documents", [])
    )
    reports = []
    for document in documents:
        fields = document.get("fields", {})
        reports.append(
            {
                "id": document.get("name", "").rsplit("/", 1)[-1],
                "patient_uid": firestore_string(fields, "patientUid"),
                "patient_email": firestore_string(fields, "patientEmail"),
                "filename": firestore_string(fields, "filename"),
                "created_at": firestore_string(fields, "createdAt"),
                "key_points": firestore_string(fields, "keyPoints").splitlines(),
                "plain_language": firestore_string(fields, "plainLanguage").splitlines(),
                "source_text": firestore_string(fields, "sourceText"),
            }
        )
    return reports


def load_report_records() -> List[Dict[str, Any]]:
    if st.session_state.get("auth_mode") == "firebase":
        return fetch_firebase_reports()
    return list(demo_report_store())


def firebase_auth_request(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Call Firebase Identity Toolkit; passwords are sent to Firebase, never persisted here."""
    firebase_config, configured = get_firebase_status()
    if not configured:
        raise RuntimeError("Firebase is not configured. Add FIREBASE_API_KEY to the app environment or Streamlit secrets.")

    try:
        response = requests.post(
            f"https://identitytoolkit.googleapis.com/v1/{endpoint}",
            params={"key": firebase_config["FIREBASE_API_KEY"]},
            json=payload,
            timeout=15,
        )
    except requests.RequestException as exc:
        raise RuntimeError("Could not reach Firebase Authentication. Check the network and try again.") from exc

    result = response.json()
    if response.ok:
        return result

    error_code = result.get("error", {}).get("message", "")
    messages = {
        "EMAIL_NOT_FOUND": "No Firebase account exists for this email address.",
        "INVALID_PASSWORD": "The email or password is incorrect.",
        "INVALID_LOGIN_CREDENTIALS": "The email or password is incorrect.",
        "EMAIL_EXISTS": "An account already exists for this email address.",
        "WEAK_PASSWORD": "Choose a stronger password (at least 6 characters).",
        "USER_DISABLED": "This account has been disabled. Contact your administrator.",
        "TOO_MANY_ATTEMPTS_TRY_LATER": "Too many attempts. Wait a while and try again.",
        "OPERATION_NOT_ALLOWED": "Enable Email/Password in Firebase Authentication sign-in providers.",
        "INVALID_EMAIL": "Enter a valid email address.",
    }
    raise RuntimeError(messages.get(error_code, "Firebase Authentication failed. Check your Firebase setup and try again."))


def save_firebase_session(user: Dict[str, Any], role: str) -> None:
    """Keep only Firebase tokens and identity in this Streamlit session, never the password."""
    st.session_state["authenticated"] = True
    st.session_state["auth_mode"] = "firebase"
    st.session_state["user_role"] = role
    st.session_state["user_email"] = user.get("email", "")
    st.session_state["firebase_uid"] = user.get("localId")
    st.session_state["firebase_id_token"] = user.get("idToken")
    st.session_state["firebase_refresh_token"] = user.get("refreshToken")
    st.session_state["firebase_token_expires_at"] = time.time() + int(user.get("expiresIn", 3600))


def save_demo_session(email: str, role: str) -> None:
    """Start a local demo session without validating or storing credentials."""
    st.session_state["authenticated"] = True
    st.session_state["auth_mode"] = "demo"
    st.session_state["user_role"] = role
    st.session_state["user_email"] = email


def authenticate_user(email: str, password: str, expected_role: str) -> None:
    _, firebase_configured = get_firebase_status()
    if not email.strip() or not password:
        raise RuntimeError("Enter your email address and password.")
    if not firebase_configured:
        save_demo_session(email.strip(), expected_role)
        return

    user = firebase_auth_request(
        "accounts:signInWithPassword",
        {"email": email.strip(), "password": password, "returnSecureToken": True},
    )
    verify_firebase_email(user)
    actual_role = firebase_user_role(user)
    if actual_role != expected_role.lower():
        raise RuntimeError(f"This Firebase account is registered as {actual_role}, not {expected_role.lower()}.")
    save_firebase_session(user, actual_role)


def verify_firebase_email(user: Dict[str, Any]) -> None:
    account = firebase_auth_request("accounts:lookup", {"idToken": user["idToken"]})
    accounts = account.get("users", [])
    if not accounts or not accounts[0].get("emailVerified"):
        raise RuntimeError("Verify your email address using the link Firebase sent before signing in.")


def firebase_user_role(user: Dict[str, Any]) -> str:
    firebase_config, configured = get_firebase_status()
    project_id = firebase_config.get("FIREBASE_PROJECT_ID")
    if not configured or not project_id:
        raise RuntimeError("Set FIREBASE_API_KEY and FIREBASE_PROJECT_ID to enable role-verified sign-in.")

    url = (
        f"https://firestore.googleapis.com/v1/projects/{project_id}"
        f"/databases/(default)/documents/users/{user['localId']}"
    )
    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {user['idToken']}"},
            timeout=15,
        )
    except requests.RequestException as exc:
        raise RuntimeError("Could not reach Firestore to verify your account role.") from exc

    if response.status_code == 404:
        raise RuntimeError("No app role is assigned to this account. Ask your administrator to provision it.")
    if not response.ok:
        raise RuntimeError("Could not verify your account role. Check the Firestore rules and setup.")

    role = response.json().get("fields", {}).get("role", {}).get("stringValue", "").lower()
    if role not in {"doctor", "patient"}:
        raise RuntimeError("This account has no valid app role. Ask your administrator to provision it.")
    return role


def create_firebase_patient_profile(user: Dict[str, Any]) -> None:
    firebase_config, configured = get_firebase_status()
    project_id = firebase_config.get("FIREBASE_PROJECT_ID")
    if not configured or not project_id:
        raise RuntimeError("Set FIREBASE_API_KEY and FIREBASE_PROJECT_ID before creating accounts.")

    url = (
        f"https://firestore.googleapis.com/v1/projects/{project_id}"
        f"/databases/(default)/documents/users"
    )
    try:
        response = requests.post(
            url,
            params={"documentId": user["localId"]},
            headers={"Authorization": f"Bearer {user['idToken']}"},
            json={"fields": {"role": {"stringValue": "patient"}}},
            timeout=15,
        )
    except requests.RequestException as exc:
        raise RuntimeError("Could not create the patient profile in Firestore.") from exc

    if response.status_code == 409:
        if firebase_user_role(user) == "patient":
            return
        raise RuntimeError("An account profile already exists with a different role. Contact your administrator.")
    if not response.ok:
        raise RuntimeError("Could not create the patient profile. Check Firestore setup and security rules.")


def refresh_firebase_session() -> None:
    if st.session_state.get("auth_mode") != "firebase":
        return

    expires_at = st.session_state.get("firebase_token_expires_at", 0)
    if not st.session_state.get("authenticated") or time.time() < expires_at - 60:
        return

    firebase_config, configured = get_firebase_status()
    refresh_token = st.session_state.get("firebase_refresh_token")
    if not configured or not refresh_token:
        st.session_state["authenticated"] = False
        return

    try:
        response = requests.post(
            "https://securetoken.googleapis.com/v1/token",
            params={"key": firebase_config["FIREBASE_API_KEY"]},
            data={"grant_type": "refresh_token", "refresh_token": refresh_token},
            timeout=15,
        )
        response.raise_for_status()
        tokens = response.json()
    except (requests.RequestException, ValueError):
        st.session_state["authenticated"] = False
        return

    st.session_state["firebase_id_token"] = tokens["id_token"]
    st.session_state["firebase_refresh_token"] = tokens["refresh_token"]
    st.session_state["firebase_token_expires_at"] = time.time() + int(tokens["expires_in"])


def send_firebase_email_action(request_type: str, email: str = "", id_token: str = "") -> None:
    payload = {"requestType": request_type}
    if email:
        payload["email"] = email
    if id_token:
        payload["idToken"] = id_token
    firebase_auth_request("accounts:sendOobCode", payload)


def init_demo_data() -> List[Dict[str, Any]]:
    return [
        {
            "id": "PT-1024",
            "name": "John Doe",
            "age": 63,
            "gender": "Male",
            "mrn": "MRN-20481",
            "study_date": "2026-09-21",
            "condition": "Hepatocellular carcinoma follow-up",
            "diagnosis": "Segment V lesion, 46.7 mm, high-risk imaging",
            "risk": "High Risk",
            "doctor": "Dr. Sarah Mitchell",
        },
        {
            "id": "PT-2087",
            "name": "Robert Johnson",
            "age": 58,
            "gender": "Male",
            "mrn": "MRN-31021",
            "study_date": "2026-09-14",
            "condition": "Liver lesion surveillance",
            "diagnosis": "No lesion detected in prior report",
            "risk": "Low Risk",
            "doctor": "Dr. Sarah Mitchell",
        },
        {
            "id": "PT-3076",
            "name": "Maria Lopez",
            "age": 71,
            "gender": "Female",
            "mrn": "MRN-11899",
            "study_date": "2026-09-05",
            "condition": "HCC screening",
            "diagnosis": "Sub-centimeter lesion under review",
            "risk": "Moderate Risk",
            "doctor": "Dr. Sarah Mitchell",
        },
    ]


def build_patient_summary(patient: Dict[str, Any], metrics: Dict[str, Any], radiomics: Dict[str, Any], location: Dict[str, Any], lirads: Dict[str, Any]) -> Dict[str, Any]:
    rag = ClinicalGuidelinesRAG()
    query = (
        f"{patient['condition']} {patient['diagnosis']} {location['segment_name']} "
        f"{metrics['tumor_volume_cm3']:.2f} cm3 lesion {lirads['lirads_category']}"
    )
    evidence = rag.retrieve_guidelines(query, top_k=3)
    recommendation = (
        f"Patient: {patient['name']} ({patient['id']})\n"
        f"Age: {patient['age']} | MRN: {patient['mrn']}\n"
        f"Current study: {patient['study_date']}\n"
        f"Diagnosis: {patient['diagnosis']}\n"
        f"AI risk category: {lirads['lirads_category']} ({lirads['malignancy_probability_percent']}%)\n"
        f"Longest axis: {metrics['recist_max_diameter_mm']:.2f} mm\n"
        f"Tumor volume: {metrics['tumor_volume_cm3']:.2f} cm³\n"
        f" guideline evidence: {len(evidence)} relevant clinical references\n"
        f"recommendation: {lirads['clinical_recommendation']}"
    )

    return {
        "recommendation": recommendation,
        "evidence": evidence,
    }


def build_clinical_summary(metrics, radiomics, location, lirads):
    findings = (
        f"AI Risk Category: {lirads['lirads_category']}\n"
        f"Lesion location: {location['segment_name']}\n"
        f"Longest axis: {metrics['recist_max_diameter_mm']:.2f} mm\n"
        f"Mean lesion HU: {radiomics['lesion_mean_hu']:.2f} HU vs mean liver HU: {radiomics['liver_mean_hu']:.2f} HU\n"
        f"Attenuation difference: {radiomics['hu_attenuation_deficit']:.2f} HU\n"
        f"Tumor volume: {metrics['tumor_volume_cm3']:.2f} cm³ | Tumor burden: {metrics['tumor_burden_percentage']:.2f}%\n"
        f"Description: {lirads['lirads_description']}"
    )

    return {
        "summary_title": "AI Clinical Report",
        "findings": findings,
        "recommendation": lirads["clinical_recommendation"],
        "risk_percent": float(lirads["malignancy_probability_percent"]),
    }


SOURCE_LABELS = {
    "AASLD": "AASLD",
    "EASL": "EASL",
    "NCCN": "NCCN",
    "BCLC": "BCLC",
    "LI-RADS": "LI-RADS",
    "Milan": "Milan criteria",
}


def guideline_source(entry: Dict[str, Any]) -> str:
    identifier = entry.get("id", "")
    title = entry.get("guideline", "")
    if "LIRADS" in identifier.upper() or "LI-RADS" in title.upper():
        return SOURCE_LABELS["LI-RADS"]
    for source in ("AASLD", "EASL", "NCCN", "BCLC"):
        if source in identifier.upper() or source in title.upper():
            return SOURCE_LABELS[source]
    return SOURCE_LABELS["Milan"]


def guideline_year(entry: Dict[str, Any]) -> str:
    match = re.search(r"(?:19|20)\d{2}", entry.get("reference", ""))
    return match.group(0) if match else "Not specified"


def openai_settings() -> tuple[str, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL")
    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", "")
            model = model or st.secrets.get("OPENAI_MODEL", "")
        except Exception:
            api_key = ""
    return api_key or "", model or "gpt-4o-mini"


def generate_guideline_chat_response(query: str, history: List[Dict[str, str]]) -> tuple[str, List[Dict[str, Any]]]:
    api_key, model = openai_settings()
    if not api_key:
        raise RuntimeError("OpenAI is not configured. Set OPENAI_API_KEY in your local environment or Streamlit secrets.")

    prior_questions = [item["content"] for item in history[-6:] if item.get("role") == "user"]
    retrieval_query = " ".join(prior_questions + [query])
    retrieved = ClinicalGuidelinesRAG().retrieve_guidelines(retrieval_query, top_k=4)
    context = [
        {
            "id": item["id"],
            "source": guideline_source(item),
            "guideline": item["guideline"],
            "category": item["category"],
            "eligibility": item["eligibility"],
            "recommendation": item["recommendation"],
            "evidence_level": item["evidence_level"],
            "reference": item["reference"],
        }
        for item in retrieved
    ]
    messages = [
        {
            "role": "system",
            "content": (
                "You are a clinical evidence assistant for clinicians. Answer only from the supplied indexed guideline records. "
                "Cite supporting record IDs inline, for example [AASLD-SURG-01]. If the records do not support an answer, "
                "say the indexed library does not contain enough evidence. Do not invent citations, claim to diagnose a patient, "
                "or give patient-specific treatment orders. Distinguish guideline statements from clinical judgment and note "
                "that recommendations depend on the complete clinical context."
            ),
        }
    ]
    messages.extend(
        {"role": item["role"], "content": item["content"]}
        for item in history[-8:]
        if item.get("role") in {"user", "assistant"} and item.get("content")
    )
    messages.append(
        {
            "role": "user",
            "content": f"Retrieved indexed guideline records:\n{json.dumps(context, ensure_ascii=False)}\n\nQuestion: {query}",
        }
    )

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "messages": messages, "temperature": 0.2},
            timeout=45,
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"].strip()
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status == 401:
            raise RuntimeError("OpenAI rejected the configured API key. Check your local OPENAI_API_KEY setting.") from exc
        if status == 429:
            raise RuntimeError("OpenAI rate limit or quota reached. Check your account usage and try again later.") from exc
        raise RuntimeError("OpenAI could not complete the request. Check the model and API configuration.") from exc
    except (requests.RequestException, KeyError, ValueError, TypeError) as exc:
        raise RuntimeError("Could not get a valid response from OpenAI. Check network access and try again.") from exc

    if not answer:
        raise RuntimeError("OpenAI returned an empty answer. Please try again.")
    return answer, retrieved


def render_dashboard_style():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #edf7ff 0%, #eaf1ff 100%);
            color: #0a1f44;
        }
        .main .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a1a3a 0%, #0d2a5c 100%);
            color: white;
        }
        [data-testid="stSidebar"] button {
            text-align: left;
            border-radius: 10px;
            min-height: 46px;
        }
        .brand-box {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 0.8rem 0.5rem 1rem 0.5rem;
            margin-bottom: 1rem;
            border-bottom: 1px solid rgba(255,255,255,0.15);
        }
        .brand-icon {
            width: 42px;
            height: 42px;
            border-radius: 12px;
            background: rgba(120, 203, 255, 0.2);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            font-weight: 700;
        }
        .brand-title {
            font-weight: 700;
            font-size: 1.1rem;
            color: white;
        }
        .brand-subtitle {
            font-size: 0.72rem;
            color: rgba(255,255,255,0.75);
        }
        .nav-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 0.7rem 0.85rem;
            margin: 0.2rem 0.35rem;
            border-radius: 10px;
            color: rgba(255,255,255,0.82);
            font-weight: 500;
        }
        .nav-item.active {
            background: rgba(106, 176, 255, 0.18);
            color: white;
            border: 1px solid rgba(155, 206, 255, 0.35);
        }
        .nav-item.inactive {
            opacity: 0.8;
        }
        .top-panel {
            background: linear-gradient(135deg, #0b2562, #0e3d8e 70%, #0b3f8f);
            border-radius: 20px;
            padding: 1.1rem 1.2rem;
            color: white;
            box-shadow: 0 12px 24px rgba(13, 41, 92, 0.08);
        }
        .metric-card {
            background: rgba(255,255,255,0.88);
            border: 1px solid rgba(19, 52, 110, 0.08);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 8px 16px rgba(22, 52, 112, 0.05);
            min-height: 110px;
        }
        .metric-value {
            font-size: 1.45rem;
            font-weight: 700;
            margin-top: 0.35rem;
            color: #0a1f44;
        }
        .metric-label {
            color: #4d6a96;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .section-card {
            background: rgba(255,255,255,0.88);
            border: 1px solid rgba(17, 46, 94, 0.08);
            border-radius: 20px;
            padding: 1.1rem 1.25rem;
            box-shadow: 0 8px 20px rgba(17, 41, 86, 0.04);
        }
        .upload-box {
            background: rgba(235, 245, 255, 0.8);
            border: 2px dashed rgba(60, 114, 210, 0.45);
            border-radius: 18px;
            padding: 2rem 1rem;
            text-align: center;
            min-height: 220px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .report-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #0a1f44;
        }
        .result-box {
            background: rgba(15, 39, 87, 0.03);
            border-radius: 16px;
            padding: 1rem;
            border: 1px solid rgba(16, 44, 98, 0.08);
        }
        .small-muted {
            color: #5c6d8a;
            font-size: 0.8rem;
        }
        .btn-highlight {
            background: linear-gradient(135deg, #2490ff, #0b5bd3);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.7rem 1.25rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def draw_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-box">
                <div class="brand-icon">L</div>
                <div>
                    <div class="brand-title">LiverAI</div>
                    <div class="brand-subtitle">Clinical Decision Support</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if normalized_role() == "patient":
            nav_items = [
                ("Dashboard", "📊"),
                ("Upload report", "⇧"),
                ("My reports", "📄"),
                ("Settings", "⚙️"),
            ]
        else:
            nav_items = [
                ("Dashboard", "📊"),
                ("Patients", "👥"),
                ("Studies", "🧾"),
                ("AI Analysis", "🧠"),
                ("Reports", "📄"),
                ("Evidence Library", "📚"),
                ("Settings", "⚙️"),
            ]

        allowed_pages = {label for label, _ in nav_items}
        if st.session_state.get("nav") not in allowed_pages:
            st.session_state["nav"] = "Dashboard"

        selected = st.session_state.get("nav", "Dashboard")
        for label, icon in nav_items:
            if st.button(
                f"{icon}  {label}",
                key=f"nav_{label.lower().replace(' ', '_')}",
                type="primary" if label == selected else "secondary",
                use_container_width=True,
            ):
                st.session_state["nav"] = label
                st.rerun()

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        if st.button("Sign out", key="sign_out", use_container_width=True):
            for key in (
                "authenticated",
                "auth_mode",
                "user_role",
                "user_email",
                "firebase_uid",
                "firebase_id_token",
                "firebase_refresh_token",
                "firebase_token_expires_at",
            ):
                st.session_state.pop(key, None)
            st.rerun()


def render_login_screen():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(110deg, #061a35 0%, #0b2d55 49.8%, #fff5f7 50.2%, #fffafa 100%);
        }
        [data-testid="stMainBlockContainer"] {
            max-width: 1440px;
            padding-top: 2rem;
        }
        div[class*="st-key-doctor-login-panel"],
        div[class*="st-key-patient-login-panel"] {
            min-height: 640px;
            padding: 1.5rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.97);
            box-shadow: 0 18px 48px rgba(4, 17, 40, 0.18);
            border: 1px solid rgba(255, 255, 255, 0.72);
        }
        div[class*="st-key-doctor-login-panel"] { border-top: 4px solid #1684ff; }
        div[class*="st-key-patient-login-panel"] { border-top: 4px solid #e32b74; }
        .login-brand {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            margin-bottom: 1.5rem;
            color: #10264a;
        }
        .login-brand-mark {
            display: grid;
            width: 42px;
            height: 42px;
            place-items: center;
            border: 2px solid #1684ff;
            border-radius: 12px;
            color: #1684ff;
            font-size: 1.2rem;
            font-weight: 800;
        }
        .login-brand-name { font-size: 1.35rem; font-weight: 800; }
        .login-brand-tagline { color: #60728f; font-size: 0.75rem; }
        .login-intro { padding: 0.5rem 0 1.1rem; }
        .login-intro h1 { margin: 0; color: #10264a; font-size: 2.1rem; }
        .login-intro p { color: #60728f; line-height: 1.5; }
        .login-features { display: grid; gap: 0.65rem; margin: 0.8rem 0 1.5rem; }
        .login-feature {
            display: flex;
            gap: 0.75rem;
            align-items: center;
            border-radius: 10px;
            padding: 0.65rem;
            background: #f3f7fc;
            color: #193557;
            font-size: 0.85rem;
        }
        .login-feature span { color: #1684ff; font-size: 1.1rem; }
        .login-note { color: #60728f; font-size: 0.78rem; }
        @media (max-width: 760px) {
            .stApp { background: linear-gradient(180deg, #0b2d55 0%, #f6f9fd 50%, #fff5f7 100%); }
            div[class*="st-key-doctor-login-panel"],
            div[class*="st-key-patient-login-panel"] { min-height: auto; padding: 1rem; }
            .login-intro h1 { font-size: 1.8rem; }
        }
        </style>
        <div class="login-brand">
            <div class="login-brand-mark">L</div>
            <div><div class="login-brand-name">LiverAI</div><div class="login-brand-tagline">Clinical Decision Support</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, firebase_configured = get_firebase_status()
    if firebase_configured:
        st.info("Sign in with your provisioned Firebase account. Your account role determines which workspace opens.")
    else:
        st.info("Demo mode is active. Use any non-empty email and password; credentials are not verified or saved.")
    doctor_col, patient_col = st.columns(2, gap="large")

    with doctor_col:
        with st.container(key="doctor-login-panel", border=True):
            st.markdown(
                """
                <div class="login-intro">
                    <h1>Doctor <span style="color:#1684ff">Login</span></h1>
                    <p>Access AI-powered liver CT analysis and clinical decision support.</p>
                </div>
                <div class="login-features">
                    <div class="login-feature"><span>◉</span><div><b>Analyze CT scans</b><br>AI-based liver and lesion detection</div></div>
                    <div class="login-feature"><span>▤</span><div><b>Evidence-based insights</b><br>Guideline-supported recommendations</div></div>
                    <div class="login-feature"><span>▣</span><div><b>Patient management</b><br>Track studies and reports</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("### Welcome, Doctor")
            st.caption("Firebase sign-in" if firebase_configured else "Demo sign-in")
            with st.form("doctor_sign_in", clear_on_submit=True):
                doctor_email = st.text_input("Email address", placeholder="doctor@hospital.com", key="doctor_email")
                doctor_password = st.text_input("Password", placeholder="Enter your password", type="password", key="doctor_password")
                doctor_submitted = st.form_submit_button("Sign in as Doctor", use_container_width=True)

            if doctor_submitted:
                try:
                    authenticate_user(doctor_email, doctor_password, "doctor")
                    st.rerun()
                except RuntimeError as exc:
                    st.error(str(exc))

    with patient_col:
        with st.container(key="patient-login-panel", border=True):
            st.markdown(
                """
                <div class="login-intro">
                    <h1>Patient <span style="color:#e32b74">Login</span></h1>
                    <p>Access your liver CT reports, health insights and follow-up information securely.</p>
                </div>
                <div class="login-features">
                    <div class="login-feature"><span style="color:#e32b74">▣</span><div><b>View your reports</b><br>Access your CT analysis results</div></div>
                    <div class="login-feature"><span style="color:#e32b74">♡</span><div><b>Understand your health</b><br>Clear explanations of your results</div></div>
                    <div class="login-feature"><span style="color:#e32b74">◷</span><div><b>Track follow-ups</b><br>Keep your medical history updated</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("### Welcome, Patient")
            st.caption("Firebase sign-in" if firebase_configured else "Demo sign-in")
            with st.form("patient_sign_in", clear_on_submit=True):
                patient_email = st.text_input("Email address", placeholder="patient@example.com", key="patient_email")
                patient_password = st.text_input("Password", placeholder="Enter your password", type="password", key="patient_password")
                patient_submitted = st.form_submit_button("Sign in as Patient", use_container_width=True)

            if patient_submitted:
                try:
                    authenticate_user(patient_email, patient_password, "patient")
                    st.rerun()
                except RuntimeError as exc:
                    st.error(str(exc))

    if firebase_configured:
        st.caption("Patient report sharing uses the authenticated Firestore account and configured security rules.")
    else:
        st.caption("Demo sign-in accepts any credentials and does not verify identity.")


def render_dashboard_screen():
    st.markdown(
        """
        <div class="top-panel">
            <div style="display:flex; justify-content:space-between; align-items:center; gap: 16px; flex-wrap:wrap;">
                <div>
                    <div style="font-size: 0.8rem; color: rgba(255,255,255,0.72); letter-spacing: 0.06em; text-transform: uppercase;">Welcome back</div>
                    <div style="font-size: 2.2rem; font-weight: 700; margin-top: 0.2rem;">LiverAI Dashboard</div>
                </div>
                <div style="display:flex; align-items:center; gap: 12px; background: rgba(255,255,255,0.08); padding: 0.55rem 0.8rem; border-radius: 14px;">
                    <div style="width: 38px; height: 38px; border-radius: 50%; background: rgba(255,255,255,0.2); display: flex; align-items: center; justify-content: center; font-weight: 700;">DS</div>
                    <div>
                        <div style="font-weight: 600;">Dr. Sarah Mitchell</div>
                        <div style="font-size: 0.72rem; color: rgba(255,255,255,0.7);">Radiologist</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    metric_cols = st.columns(4)
    metric_stats = [
        ("Total Patients", "124", "+12% this month"),
        ("CT Studies", "286", "+10% this month"),
        ("Analyses Completed", "240", "95% success rate"),
        ("Avg. Processing Time", "4.2 min", "-35% improvement"),
    ]

    for col, (label, value, note) in zip(metric_cols, metric_stats):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="small-muted">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    upload_col, status_col = st.columns([1.4, 1.0])
    uploaded_file = None

    with upload_col:
        st.markdown(
            """
            <div class="section-card">
                <div class="report-title">Upload CT Scan</div>
                <div class="small-muted" style="margin-top:0.3rem; margin-bottom: 1rem;">Upload a liver CT study for automated AI analysis.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Select CT scan files",
            type=["nii", "nii.gz", "dcm", "dicom"],
            label_visibility="collapsed",
            accept_multiple_files=True,
        )

        if uploaded_files:
            names = ", ".join(file.name for file in uploaded_files)
            st.caption(f"Selected files: {names}")
        else:
            st.markdown(
                """
                <div class='upload-box'>
                    <div>
                        <div style='font-size: 2.3rem; color: #0d4fa7;'>⇪</div>
                        <div style='font-weight: 600; color: #123d83; margin-top: 0.5rem;'>Drop your CT files here</div>
                        <div class='small-muted' style='margin-top: 0.3rem;'>Supported: 3D NIfTI or a DICOM series folder</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
        analyze_button = st.button("Analyze CT", use_container_width=True)

    with status_col:
        st.markdown(
            """
            <div class="section-card">
                <div class="report-title">Analysis in Progress</div>
                <div class='small-muted' style='margin-top: 0.35rem;'>Your CT scan is being analyzed by our AI system.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

        status_steps = [
            "File upload",
            "Preprocessing",
            "Liver segmentation",
            "Lesion detection",
            "Multi-agent analysis",
            "Clinical report",
        ]

        for index, step in enumerate(status_steps, start=1):
            done = index <= 2 if not analyze_button else index <= 6
            marker = "✓" if done else "○"
            st.markdown(
                f"<div class='nav-item {'active' if done else 'inactive'}'><span>{marker}</span><span>{step}</span></div>",
                unsafe_allow_html=True,
            )

    if uploaded_files and analyze_button:
        os.makedirs("data/uploads", exist_ok=True)
        study_id = uuid.uuid4().hex
        study_dir = os.path.join("data", "uploads", study_id)
        os.makedirs(study_dir, exist_ok=True)

        if len(uploaded_files) == 1 and uploaded_files[0].name.lower().endswith((".nii", ".nii.gz", ".mha", ".mhd")):
            file_path = os.path.join(study_dir, os.path.basename(uploaded_files[0].name))
            with open(file_path, "wb") as f:
                f.write(uploaded_files[0].getbuffer())
            target_path = file_path
        else:
            for uploaded_file in uploaded_files:
                safe_name = os.path.basename(uploaded_file.name)
                target_path = os.path.join(study_dir, safe_name)
                with open(target_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            target_path = study_dir

        from main import run_pipeline

        try:
            with st.spinner("Analyzing the CT scan..."):
                result = run_pipeline(target_path)
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(f"CT analysis failed: {exc}")
            return

        seg = result["segmentation"]
        ana = result["analysis"]
        metrics = seg["metrics"]
        radiomics = ana["radiomics"]
        location = ana["couinaud_localization"]
        lirads = ana["lirads_staging"]

        report = build_clinical_summary(metrics, radiomics, location, lirads)

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        st.subheader("Clinical Report")
        summary_cols = st.columns([1.25, 1.0])

        with summary_cols[0]:
            st.markdown(
                """
                <div class="section-card">
                    <div class="report-title">Key Findings</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="result-box" style="margin-top: 0.6rem; white-space: pre-line; line-height: 1.8;">
                {report['findings']}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="section-card" style="margin-top: 1rem;">
                    <div class="report-title">Recommendation</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.info(report["recommendation"])

        with summary_cols[1]:
            overlay = seg.get("overlay_plot")
            if overlay and os.path.exists(overlay):
                st.image(overlay, caption="Segmentation Overlay", use_container_width=True)

            dashboard = ana.get("dashboard_image")
            if dashboard and os.path.exists(dashboard):
                st.image(dashboard, caption="Lesion Analysis Dashboard", use_container_width=True)

        st.markdown("<div style='height: 1.3rem;'></div>", unsafe_allow_html=True)

        metric_summary = st.columns(4)
        summary_values = [
            ("Liver Volume", f"{metrics['total_liver_volume_cm3']:.2f} mL"),
            ("Lesion Volume", f"{metrics['tumor_volume_cm3']:.2f} cm³"),
            ("Tumor Burden", f"{metrics['tumor_burden_percentage']:.2f}%"),
            ("Longest Axis", f"{metrics['recist_max_diameter_mm']:.2f} mm"),
        ]

        for col, (label, value) in zip(metric_summary, summary_values):
            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value" style="font-size: 1.1rem;">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height: 1.3rem;'></div>", unsafe_allow_html=True)
        st.caption("This output is for research and clinical decision support only and is not a diagnostic substitute.")


def render_report_detail(report: Dict[str, Any], patient_view: bool = False) -> None:
    st.subheader(report.get("filename", "Uploaded report"))
    identity = "Your report" if patient_view else f"Patient: {report.get('patient_email') or 'Demo patient'}"
    st.caption(f"{identity} · Uploaded {report.get('created_at', 'date unavailable')}")

    if report.get("key_points"):
        st.markdown("**Key points from the report**")
        for point in report["key_points"]:
            st.markdown(f"- {point}")
    if report.get("plain_language"):
        with st.expander("What these report terms mean"):
            for explanation in report["plain_language"]:
                st.markdown(f"- {explanation}")
    if report.get("source_text"):
        with st.expander("View extracted report text"):
            st.text(report["source_text"])
    st.caption("This explanation summarizes report wording. It is not a diagnosis or a substitute for your clinician.")


def render_patient_dashboard_screen() -> None:
    email = st.session_state.get("user_email", "")
    patient_name = email.split("@", 1)[0].replace(".", " ").replace("_", " ").title() or "Patient"
    st.markdown(
        f"""
        <div class="top-panel">
            <div style="font-size:0.8rem;color:rgba(255,255,255,0.72);text-transform:uppercase">Patient workspace</div>
            <div style="font-size:2rem;font-weight:700;margin-top:0.2rem">Welcome, {html.escape(patient_name)}</div>
            <div style="margin-top:0.35rem">Your reports and follow-up information, in one place.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        reports = load_report_records()
        report_error = None
    except RuntimeError as exc:
        reports = []
        report_error = str(exc)
    if report_error:
        st.warning(report_error)

    metric_cols = st.columns(3)
    metric_cols[0].metric("Reports", len(reports))
    metric_cols[1].metric("Most recent upload", reports[0].get("created_at", "None")[:10] if reports else "None")
    metric_cols[2].metric("Account", "Firebase" if st.session_state.get("auth_mode") == "firebase" else "Demo")

    action_cols = st.columns([1, 1, 2])
    with action_cols[0]:
        if st.button("Upload a report", type="primary", width="stretch"):
            st.session_state["nav"] = "Upload report"
            st.rerun()
    with action_cols[1]:
        if st.button("View my reports", width="stretch"):
            st.session_state["nav"] = "My reports"
            st.rerun()

    st.header("Recent reports")
    if not reports:
        st.info("No reports yet. Upload a PDF, DOCX, or TXT file to get a clear, source-grounded summary.")
    else:
        for report in reports[:3]:
            with st.container(border=True):
                st.markdown(f"**{report.get('filename', 'Uploaded report')}**")
                st.caption(report.get("created_at", ""))
                if report.get("key_points"):
                    st.write(report["key_points"][0])


def render_patient_upload_screen() -> None:
    st.title("Upload a medical report")
    st.write("Upload a text-based radiology or clinical report. The summary will use the report's own wording.")
    if st.session_state.get("auth_mode") != "firebase":
        st.warning("Demo mode is not private: report summaries are shared across active app sessions and cleared when the server stops. Do not upload real patient records.")
    uploaded_file = st.file_uploader(
        "Choose a PDF, DOCX, or TXT report",
        type=["pdf", "docx", "txt"],
        key="patient_report_upload",
    )
    if uploaded_file is not None:
        st.caption(f"Selected: {uploaded_file.name} · {uploaded_file.size / 1024:.0f} KB")
    consent = st.checkbox(
        "I consent to process this report. In Firebase mode, its extracted text is stored for my care team's review.",
        key="report_upload_consent",
    )

    if st.button("Analyze report", type="primary", disabled=uploaded_file is None or not consent):
        if uploaded_file.size > 20 * 1024 * 1024:
            st.error("The report is larger than 20 MB. Choose a smaller file.")
        else:
            try:
                text = extract_report_text(uploaded_file.name, uploaded_file.getvalue())
                summary = analyze_report_text(text)
                report = {
                    "id": uuid.uuid4().hex,
                    "patient_uid": st.session_state.get("firebase_uid", "demo"),
                    "patient_email": st.session_state.get("user_email", ""),
                    "filename": uploaded_file.name,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "key_points": summary["key_points"],
                    "plain_language": summary["plain_language"],
                    "source_text": summary["source_text"],
                }
                save_report_record(report)
                st.session_state["latest_patient_report"] = report
                st.success("Report analyzed and added to your report history.")
            except (ValueError, RuntimeError, requests.RequestException) as exc:
                st.error(str(exc))

    report = st.session_state.get("latest_patient_report")
    if report:
        with st.container(border=True):
            st.header("Report summary")
            render_report_detail(report, patient_view=True)


def render_patient_reports_screen() -> None:
    st.title("My reports")
    try:
        reports = load_report_records()
    except RuntimeError as exc:
        st.error(str(exc))
        return
    if not reports:
        st.info("Your uploaded reports will appear here after analysis.")
        return
    for report in reports:
        with st.container(border=True):
            render_report_detail(report, patient_view=True)


def render_patients_screen():
    patients = init_demo_data()
    st.title("Patients")
    selected_patient_id = st.selectbox("Select patient", [p["id"] for p in patients])
    patient = next(p for p in patients if p["id"] == selected_patient_id)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patient ID", patient["id"])
    c2.metric("Age", f"{patient['age']} years")
    c3.metric("MRN", patient["mrn"])
    c4.metric("Risk", patient["risk"])

    st.subheader("Patient Overview")
    overview = st.container()
    with overview:
        st.write(f"Name: {patient['name']}")
        st.write(f"Gender: {patient['gender']}")
        st.write(f"Condition: {patient['condition']}")
        st.write(f"Study Date: {patient['study_date']}")
        st.write(f"Assigned Doctor: {patient['doctor']}")

    st.subheader("Clinical Summary")
    st.info(
        "The patient summary is generated using the clinical RAG knowledge base to combine lesion metrics, diagnosis context and evidence-based treatment guidance."
    )

    st.subheader("Patient-submitted reports")
    st.caption("Firebase patient uploads are linked to their account UID in the report review queue.")
    if st.button("Open report review", key="patient_report_review"):
        st.session_state["nav"] = "Reports"
        st.rerun()


def render_studies_screen():
    st.title("Studies")
    studies = [
        {"title": "Liver Segmentation", "date": "2026-09-21", "status": "Completed", "patient": "PT-1024"},
        {"title": "Lesion Detection", "date": "2026-09-14", "status": "Completed", "patient": "PT-2087"},
        {"title": "AI Risk Assessment", "date": "2026-09-05", "status": "Pending", "patient": "PT-3076"},
    ]
    st.dataframe(studies, use_container_width=True)


def render_ai_analysis_screen():
    st.title("AI Analysis")
    st.caption("Ask about liver guidelines. Responses use retrieved records from the indexed clinical RAG library.")
    st.warning("Do not include patient names, dates of birth, medical record numbers, or other identifying information. Questions are sent to OpenAI when configured.")
    if not openai_settings()[0]:
        st.info("OpenAI is not configured yet. Add OPENAI_API_KEY to your local environment or Streamlit secrets to enable chat responses.")

    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("Clear conversation", icon=":material/delete_sweep:"):
            st.session_state["clinical_chat"] = []
            st.rerun()

    messages = st.session_state.setdefault("clinical_chat", [])
    indexed_by_id = {entry["id"]: entry for entry in CLINICAL_KNOWLEDGE_BASE}
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            source_ids = message.get("source_ids", [])
            if source_ids:
                with st.expander("Retrieved guideline sources"):
                    for source_id in source_ids:
                        entry = indexed_by_id.get(source_id)
                        if entry:
                            st.markdown(f"**[{source_id}] {entry['guideline']}**")
                            st.caption(entry["reference"])

    if not messages:
        st.markdown("**Try a question**")
        st.caption("For example: What does LI-RADS say about an LR-4 observation?")

    prompt = st.chat_input("Ask about a guideline, category, or clinical evidence")
    if prompt:
        previous_history = list(messages)
        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching indexed evidence and preparing an answer..."):
                try:
                    answer, sources = generate_guideline_chat_response(prompt, previous_history)
                    source_ids = [entry["id"] for entry in sources]
                    st.markdown(answer)
                    if sources:
                        with st.expander("Retrieved guideline sources"):
                            for entry in sources:
                                st.markdown(f"**[{entry['id']}] {entry['guideline']}**")
                                st.caption(entry["reference"])
                    else:
                        st.info("No matching records were found in the current indexed library. The assistant was asked not to infer beyond it.")
                except RuntimeError as exc:
                    answer = str(exc)
                    source_ids = []
                    st.error(answer)
        messages.append({"role": "assistant", "content": answer, "source_ids": source_ids})

    st.caption("Clinical evidence support only. Verify recommendations in the cited source and apply clinician judgment to the full patient context.")


def render_reports_screen():
    if normalized_role() == "patient":
        render_patient_reports_screen()
        return

    st.title("Patient report review")
    st.write("Reports uploaded by patient accounts appear here for clinician review.")
    try:
        reports = load_report_records()
    except RuntimeError as exc:
        st.error(str(exc))
        return
    if not reports:
        st.info("No patient reports are available yet.")
        return

    st.metric("Reports awaiting review", len(reports))
    options = {
        f"{report.get('filename', 'Report')} · {report.get('patient_email') or 'Demo patient'} · {report.get('created_at', '')[:10]}": report
        for report in reports
    }
    selected_label = st.selectbox("Select a patient report", list(options))
    with st.container(border=True):
        render_report_detail(options[selected_label])


def render_evidence_library_screen():
    entries = CLINICAL_KNOWLEDGE_BASE
    sources = list(dict.fromkeys(guideline_source(entry) for entry in entries))
    source_counts = {
        source: sum(guideline_source(entry) == source for entry in entries)
        for source in sources
    }

    st.markdown(
        """
        <div class="top-panel">
            <div style="font-size:0.8rem;color:rgba(255,255,255,0.72);text-transform:uppercase">Doctor workspace</div>
            <div style="font-size:2rem;font-weight:700;margin-top:0.2rem">Evidence Library</div>
            <div style="margin-top:0.35rem">Search indexed liver-care guidance and inspect the evidence behind each RAG result.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"{len(entries)} indexed RAG records · Counts below reflect this local knowledge base, not the total publications from each organization.")

    for start in range(0, len(sources), 3):
        source_row = st.columns(3)
        for column, source in zip(source_row, sources[start:start + 3]):
            with column:
                with st.container(border=True):
                    st.subheader(source)
                    st.metric("Indexed records", source_counts[source])

    st.space("small")
    filters_column, results_column, detail_column = st.columns([0.8, 1.2, 1.6])
    with filters_column:
        st.subheader("Filters")
        selected_sources = st.multiselect("Source", sources, default=sources, key="evidence_sources")
        categories = sorted({entry["category"] for entry in entries})
        selected_categories = st.multiselect("Topic", categories, key="evidence_categories")
        years = sorted({guideline_year(entry) for entry in entries}, reverse=True)
        selected_years = st.multiselect("Year", years, key="evidence_years")
        search_text = st.text_input("Search indexed guidance", placeholder="Try lesion, ablation, LR-5...")

    filtered_entries = []
    for entry in entries:
        searchable = " ".join(str(value) for value in entry.values()).lower()
        if selected_sources and guideline_source(entry) not in selected_sources:
            continue
        if selected_categories and entry["category"] not in selected_categories:
            continue
        if selected_years and guideline_year(entry) not in selected_years:
            continue
        if search_text and search_text.lower() not in searchable:
            continue
        filtered_entries.append(entry)

    selected_id = st.session_state.get("selected_guideline_id")
    if not any(entry["id"] == selected_id for entry in filtered_entries):
        selected_id = filtered_entries[0]["id"] if filtered_entries else None
        st.session_state["selected_guideline_id"] = selected_id

    with results_column:
        st.subheader(f"Indexed guidance ({len(filtered_entries)})")
        if not filtered_entries:
            st.info("No indexed records match these filters.")
        for entry in filtered_entries:
            with st.container(border=True):
                st.badge(guideline_source(entry), color="blue")
                st.markdown(f"**{entry['guideline']}**")
                st.caption(f"{entry['category']} · {guideline_year(entry)} · {entry['evidence_level']}")
                if st.button("View record", key=f"evidence_{entry['id']}", icon=":material/arrow_forward:"):
                    st.session_state["selected_guideline_id"] = entry["id"]
                    st.rerun()

    with detail_column:
        selected_entry = next((entry for entry in filtered_entries if entry["id"] == selected_id), None)
        if selected_entry:
            with st.container(border=True):
                st.badge(guideline_source(selected_entry), color="green")
                st.subheader(selected_entry["guideline"])
                st.caption(f"Record ID: {selected_entry['id']} · {selected_entry['category']} · {guideline_year(selected_entry)}")
                st.markdown("**Recommendation in indexed record**")
                st.write(selected_entry["recommendation"])
                st.markdown("**Eligibility context**")
                st.write(selected_entry["eligibility"])
                st.markdown("**Evidence level**")
                st.write(selected_entry["evidence_level"])
                with st.expander("Citation"):
                    st.write(selected_entry["reference"])
                st.caption("This is the indexed RAG summary, not a full-text publication or downloadable PDF.")


def render_settings_screen():
    st.title("Settings")
    st.write("Authentication mode: Demo" if st.session_state.get("auth_mode") == "demo" else "Authentication mode: Firebase Authentication")
    st.caption("Demo sign-in does not verify or store email addresses or passwords.")


def main():
    st.set_page_config(page_title="LiverAI Analysis", page_icon="🩺", layout="wide")
    render_dashboard_style()

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    refresh_firebase_session()

    if not st.session_state["authenticated"]:
        render_login_screen()
        return

    draw_sidebar()
    page = st.session_state.get("nav", "Dashboard")

    if normalized_role() == "patient":
        if page == "Dashboard":
            render_patient_dashboard_screen()
        elif page == "Upload report":
            render_patient_upload_screen()
        elif page == "My reports":
            render_patient_reports_screen()
        else:
            render_settings_screen()
        return

    if page == "Dashboard":
        render_dashboard_screen()
    elif page == "Patients":
        render_patients_screen()
    elif page == "Studies":
        render_studies_screen()
    elif page == "AI Analysis":
        render_ai_analysis_screen()
    elif page == "Reports":
        render_reports_screen()
    elif page == "Evidence Library":
        render_evidence_library_screen()
    else:
        render_settings_screen()


if __name__ == "__main__":
    main()