import io
import logging
import json
import os
import pickle
import time
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from rapidfuzz import fuzz
from sklearn.cluster import AgglomerativeClustering
from semhash import SemHash
import spacy

# ✅ DODANE: normalizacja
import re
import unidecode

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="🔍 Groupowanie fraz → Excel Brief Pipeline",
    initial_sidebar_state="expanded"
)

st.sidebar.header("⚙️ Configuration")

# API Key
OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key", type="password")

# Models - wybór z listy
OPENAI_EMBEDDING_MODEL = st.sidebar.selectbox(
    "Model embeddingów",
    ["text-embedding-3-large", "text-embedding-3-small"],
    index=0
)

OPENAI_CHAT_MODEL = st.sidebar.selectbox(
    "Model czatu (dla briefów)",
    ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
    index=0
)

# Parameters with explanations
DEDUP_THRESHOLD = st.sidebar.slider(
    "Deduplication Threshold (RapidFuzz)", 0, 100, 85, 1
)
CLUSTER_SIM = st.sidebar.slider(
    "Initial Clustering Similarity Threshold", 0.0, 1.0, 0.80, 0.01
)
MERGE_SIM = st.sidebar.slider(
    "Cluster Merge Similarity Threshold", 0.0, 1.0, 0.85, 0.01
)
SEMHASH_SIM = st.sidebar.slider(
    "SemHash Similarity Threshold", 0.80, 0.99, 0.95, 0.01
)
USE_SEMHASH = st.sidebar.checkbox("Użyj SemHash do deduplikacji", value=False)

# -----------------------------
# NOWA FUNKCJA: WYCZYSZCZENIE CHECKPOINTA
# -----------------------------
if st.sidebar.button("🗑️ Wyczyść checkpoint"):
    if os.path.exists("briefs.pkl"):
        os.remove("briefs.pkl")
        st.sidebar.success("Checkpoint został usunięty! 🔥")
    else:
        st.sidebar.info("Brak pliku checkpointa do usunięcia.")

# -----------------------------
# Parametry – objaśnienia
# -----------------------------
st.sidebar.markdown("### ℹ️ Objaśnienia parametrów")
st.sidebar.info("""
**Deduplication Threshold (RapidFuzz)** – próg podobieństwa (0–100), powyżej którego frazy są traktowane jako duplikaty.  
**Initial Clustering Similarity Threshold** – minimalne podobieństwo (0–1), żeby frazy trafiły do tego samego klastra.  
**Cluster Merge Similarity Threshold** – próg podobieństwa (0–1), przy którym łączymy klastry.  
**SemHash Similarity Threshold** – jeśli użyjesz SemHash, określa jak bliskie semantycznie muszą być frazy, żeby uznać je za duplikaty.
""")

# -----------------------------
# NLP – Lematyzacja (spaCy)
# -----------------------------
@st.cache_resource
def load_spacy():
    try:
        return spacy.load("pl_core_news_sm")
    except:
        st.warning("⚠️ Musisz zainstalować model spaCy: python -m spacy download pl_core_news_sm")
        return None

nlp = load_spacy()

def lemmatize_texts(texts: List[str]) -> List[str]:
    """Zwraca lematy tekstów (używane tylko do embeddingów)."""
    if not nlp:
        return texts
    return [" ".join([token.lemma_.lower() for token in nlp(t)]) for t in texts]

# -----------------------------
# ✅ DODANE: normalizacja fraz
# -----------------------------
def normalize_phrase(s: str) -> str:
    s = str(s).strip().lower()
    s = unidecode.unidecode(s)
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# -----------------------------
# Helpers
# -----------------------------
# ✅ ZMIENIONE: dedup na token_set_ratio + normalizacja
def deduplicate(questions: List[str], threshold: int = 85) -> List[str]:
    unique = []
    unique_norm = []
    for q in questions:
        nq = normalize_phrase(q)
        if not any(fuzz.token_set_ratio(nq, u) >= threshold for u in unique_norm):
            unique.append(q)
            unique_norm.append(nq)
    return unique

def semhash_deduplicate(questions: List[str], threshold: float = 0.95) -> List[str]:
    try:
        sh = SemHash.from_records(records=questions)
        result = sh.self_deduplicate(threshold=threshold)
        if hasattr(result, "selected"):
            return result.selected
        elif hasattr(result, "deduplicated"):
            return result.deduplicated
        elif isinstance(result, list):
            return result
        else:
            return deduplicate(questions, threshold=90)
    except Exception as e:
        logging.warning(f"⚠️ SemHash failed ({e}) → fallback RapidFuzz")
        return deduplicate(questions, threshold=90)

def embed_texts(client: OpenAI, texts: List[str], model=OPENAI_EMBEDDING_MODEL) -> np.ndarray:
    response = client.embeddings.create(model=model, input=texts)
    return np.array([d.embedding for d in response.data])

def cluster_questions(questions: List[str], embeddings: np.ndarray, sim_threshold=0.8) -> Dict[int, List[str]]:
    if not questions:
        return {}
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=1 - sim_threshold,
    )
    labels = clustering.fit_predict(embeddings)
    clustered: Dict[int, List[str]] = {}
    for label, q in zip(labels, questions):
        clustered.setdefault(int(label), []).append(q)
    return clustered

def merge_similar_clusters(clusters: Dict[int, List[str]], embeddings: np.ndarray, sim_threshold=0.85, q2i: Dict[str, int] = None) -> Dict[int, List[str]]:
    if not clusters:
        return {}
    centroids = {}
    for cid, qs in clusters.items():
        idxs = [q2i[q] for q in qs if q in q2i]
        if not idxs:
            continue
        centroid = np.mean(embeddings[idxs], axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        centroids[cid] = centroid

    merged: Dict[int, List[str]] = {}
    used = set()
    new_id = 0
    cluster_ids = list(clusters.keys())

    for cid in cluster_ids:
        if cid in used or cid not in centroids:
            continue
        merged[new_id] = list(clusters[cid])
        used.add(cid)
        for cid2 in cluster_ids:
            if cid2 in used or cid2 not in centroids:
                continue
            sim = float(np.dot(centroids[cid], centroids[cid2]))
            if sim >= sim_threshold:
                merged[new_id].extend(clusters[cid2])
                used.add(cid2)
        new_id += 1
    return merged

# ✅ DODANE: dedup wewnątrz klastrów po merge
def deduplicate_within_clusters(clusters: Dict[int, List[str]], threshold: int = 92) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    for cid, qs in clusters.items():
        qs = list(dict.fromkeys(qs))  # usuń identyczne
        out[cid] = deduplicate(qs, threshold=threshold)  # usuń semantyczne
    return out

# ✅ ZMIENIONE: global dedup na token_set_ratio + normalizacja
def global_deduplicate_clusters(clusters: Dict[int, List[str]], threshold: int = 90) -> Dict[int, List[str]]:
    seen_norm = []
    new_clusters: Dict[int, List[str]] = {}
    for cid, qs in clusters.items():
        unique_qs = []
        for q in qs:
            nq = normalize_phrase(q)
            if not any(fuzz.token_set_ratio(nq, s) >= threshold for s in seen_norm):
                unique_qs.append(q)
                seen_norm.append(nq)
        if unique_qs:
            new_clusters[cid] = unique_qs
    return new_clusters

# ✅ DODANE: stabilny wybór main_phrase
def pick_main_phrase(qs: List[str]) -> str:
    return sorted(qs, key=lambda x: (len(normalize_phrase(x)), len(str(x))))[0] if qs else ""

def generate_article_brief(questions: List[str], client: OpenAI | None, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    if client is None:
        return {"intencja": "", "frazy": ", ".join(questions), "tytul": "", "wytyczne": ""}
    prompt = f"""
Dla poniższej listy fraz przygotuj dane do planu artykułu.

Frazy: {questions}

Odpowiedz w formacie:

Intencja: [typ intencji wyszukiwania]
Frazy: [lista fraz long-tail, rozdzielona przecinkami]
Tytuł: [SEO-friendly, max 70 znaków, naturalny, z głównym keywordem]
Wytyczne: [2–3 zdania opisu oczekiwań użytkownika]
"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Jesteś asystentem SEO. Zawsze trzymaj się formatu."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        content = resp.choices[0].message.content.strip()
        result = {"intencja": "", "frazy": "", "tytul": "", "wytyczne": ""}
        for line in content.splitlines():
            low = line.lower()
            if low.startswith("intencja:"):
                result["intencja"] = line.split(":", 1)[1].strip()
            elif low.startswith("frazy:"):
                result["frazy"] = line.split(":", 1)[1].strip()
            elif low.startswith("tytuł:") or low.startswith("tytul:"):
                result["tytul"] = line.split(":", 1)[1].strip()
            elif low.startswith("wytyczne:"):
                result["wytyczne"] = line.split(":", 1)[1].strip()
        result["frazy"] = result["frazy"] or ", ".join(questions)
        return result
    except Exception as e:
        logging.warning(f"⚠️ Brief parse failed: {e}")
        return {"intencja": "", "frazy": ", ".join(questions), "tytul": "", "wytyczne": ""}

# -----------------------------
# Main App
# -----------------------------
st.title("🔍 Groupowanie fraz → Excel Brief Pipeline")

status = st.empty()
progress_bar = st.progress(0)
log_box = st.container()

def update_status(message: str, progress: int):
    status.text(message)
    progress_bar.progress(progress)
    log_box.write(message)

phrases_input = st.sidebar.text_area("Wklej frazy, jedna na linię:")

# -----------------------------
# Checkpoint helpers
# -----------------------------
def save_checkpoint(data, filename="briefs.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_checkpoint(filename="briefs.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return []

# -----------------------------
# Główna logika aplikacji
# -----------------------------
if st.sidebar.button("Uruchom grupowanie"):
    if not phrases_input.strip():
        st.warning("⚠️ Wklej najpierw listę fraz.")
        st.stop()

    if not OPENAI_API_KEY:
        st.error("⚠️ Podaj OpenAI API Key w panelu bocznym.")
        st.stop()

    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    questions = [line.strip() for line in phrases_input.splitlines() if line.strip()]
    update_status(f"📥 Wczytano frazy: {len(questions)}", 5)

    if USE_SEMHASH:
        filtered = semhash_deduplicate(questions, threshold=SEMHASH_SIM)
        update_status(f"🧹 Deduplication (SemHash {SEMHASH_SIM}): {len(questions)} → {len(filtered)}", 15)
    else:
        filtered = deduplicate(questions, threshold=DEDUP_THRESHOLD)
        update_status(f"🧹 Deduplication (RapidFuzz {DEDUP_THRESHOLD}): {len(questions)} → {len(filtered)}", 15)

    update_status("🧠 Generowanie embeddingów...", 35)
    lemmatized = lemmatize_texts(filtered)
    embeddings = embed_texts(openai_client, lemmatized, model=OPENAI_EMBEDDING_MODEL)
    q2i = {q: i for i, q in enumerate(filtered)}

    clusters = cluster_questions(filtered, embeddings, sim_threshold=CLUSTER_SIM)
    update_status(f"🧩 Klastrowanie fraz: powstało {len(clusters)} klastrów", 55)

    clusters = merge_similar_clusters(clusters, embeddings, sim_threshold=MERGE_SIM, q2i=q2i)
    update_status(f"🔗 Scalanie podobnych klastrów (próg {MERGE_SIM}): teraz {len(clusters)} klastrów", 70)

    # ✅ DODANE: dedup wewnątrz klastrów po merge (żeby nie wracały powtórki)
    clusters = deduplicate_within_clusters(clusters, threshold=92)

    clusters = global_deduplicate_clusters(clusters, threshold=90)
    update_status(f"🧽 Usuwanie duplikatów między klastrami: {len(clusters)} końcowych klastrów", 85)

    update_status(f"✅ Pominięto walidację LLM – pozostawiono {len(clusters)} klastrów po klasycznym scaleniu", 90)

    # -----------------------------
    # Wczytaj checkpoint i generuj briefy
    # -----------------------------
    rows = load_checkpoint()
    done = len(rows)
    total = len(clusters)

    if done > 0:
        update_status(f"🔁 Wczytano {done} gotowych briefów z checkpointa", 90)
    else:
        update_status("📝 Rozpoczynam generowanie briefów od początku", 90)

    for i, (label, qs) in enumerate(clusters.items(), 1):
        if i <= done:
            continue
        try:
            update_status(f"📝 Generuję brief {i}/{total} ({len(qs)} fraz)", int(95 * i / total))
            brief = generate_article_brief(qs, openai_client, model=OPENAI_CHAT_MODEL)
            rows.append({
                "cluster_id": label,
                "main_phrase": pick_main_phrase(qs),  # ✅ ZMIENIONE: stabilny wybór
                "intencja": brief.get("intencja", ""),
                "frazy": ", ".join(qs),
                "tytul": brief.get("tytul", ""),
                "wytyczne": brief.get("wytyczne", ""),
            })
            save_checkpoint(rows)
            time.sleep(1.5)
        except Exception as e:
            logging.warning(f"⚠️ Błąd przy klastrze {i}/{total}: {e}")
            time.sleep(3)
            continue

    update_status("✅ Wszystkie briefy wygenerowane lub wczytane z checkpointa", 98)

    # -----------------------------
    # Zapis do Excela
    # -----------------------------
    df = pd.DataFrame(rows)
    xlsx_buffer = io.BytesIO()
    with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Briefs", index=False)
    xlsx_buffer.seek(0)

    st.session_state["excel_buffer"] = xlsx_buffer.getvalue()
    st.session_state["results"] = rows

    update_status("✅ Gotowe!", 100)

if "excel_buffer" in st.session_state:
    st.download_button(
        label="📥 Pobierz Excel",
        data=st.session_state["excel_buffer"],
        file_name="frazy_briefy.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.success("✅ Zakończono przetwarzanie.")
    st.subheader("📊 Podgląd wyników")
    st.dataframe(pd.DataFrame(st.session_state["results"]))




