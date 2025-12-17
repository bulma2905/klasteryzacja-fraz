import io
import logging
import os
import pickle
import time
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from rapidfuzz import fuzz
from sklearn.cluster import AgglomerativeClustering
from semhash import SemHash
import spacy

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

OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key", type="password")

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
# Checkpoint cleanup
# -----------------------------
if st.sidebar.button("🗑️ Wyczyść checkpoint"):
    if os.path.exists("briefs.pkl"):
        os.remove("briefs.pkl")
        st.sidebar.success("Checkpoint został usunięty! 🔥")
    else:
        st.sidebar.info("Brak pliku checkpointa do usunięcia.")

st.sidebar.markdown("### ℹ️ Objaśnienia parametrów")
st.sidebar.info("""
**Deduplication Threshold (RapidFuzz)** – próg podobieństwa (0–100), powyżej którego frazy są traktowane jako duplikaty.  
**Initial Clustering Similarity Threshold** – minimalne podobieństwo (0–1), żeby frazy trafiły do tego samego klastra.  
**Cluster Merge Similarity Threshold** – próg podobieństwa (0–1), przy którym łączymy klastry.  
**SemHash Similarity Threshold** – jeśli użyjesz SemHash, określa jak bliskie semantycznie muszą być frazy, żeby uznać je za duplikaty.
""")

# -----------------------------
# spaCy
# -----------------------------
@st.cache_resource
def load_spacy():
    try:
        return spacy.load("pl_core_news_sm")
    except Exception:
        st.warning("⚠️ Musisz zainstalować model spaCy: python -m spacy download pl_core_news_sm")
        return None

nlp = load_spacy()

def lemmatize_texts(texts: List[str]) -> List[str]:
    if not nlp:
        return texts
    return [" ".join([token.lemma_.lower() for token in nlp(t)]) for t in texts]

# -----------------------------
# Normalizacja
# -----------------------------
def normalize_phrase(s: str) -> str:
    s = str(s).strip().lower()
    s = unidecode.unidecode(s)
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# -----------------------------
# ✅ DEDUP, ALE NIC NIE GINIE: reps + mapa rep -> wszystkie frazy
# -----------------------------
def deduplicate_with_map(questions: List[str], threshold: int = 85) -> Tuple[List[str], Dict[str, List[str]]]:
    reps: List[str] = []
    reps_norm: List[str] = []
    rep2all: Dict[str, List[str]] = {}

    for q in questions:
        nq = normalize_phrase(q)

        best_idx = -1
        best_score = -1
        for i, rn in enumerate(reps_norm):
            score = fuzz.token_set_ratio(nq, rn)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_score >= threshold and best_idx >= 0:
            rep = reps[best_idx]
            rep2all[rep].append(q)
        else:
            reps.append(q)
            reps_norm.append(nq)
            rep2all[q] = [q]

    return reps, rep2all

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
        return questions
    except Exception as e:
        logging.warning(f"⚠️ SemHash failed ({e})")
        return questions

def embed_texts(client: OpenAI, texts: List[str], model: str) -> np.ndarray:
    response = client.embeddings.create(model=model, input=texts)
    return np.array([d.embedding for d in response.data])

def cluster_questions(questions: List[str], embeddings: np.ndarray, sim_threshold: float = 0.8) -> Dict[int, List[str]]:
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

def merge_similar_clusters(
    clusters: Dict[int, List[str]],
    embeddings: np.ndarray,
    sim_threshold: float = 0.85,
    q2i: Dict[str, int] | None = None
) -> Dict[int, List[str]]:
    if not clusters or not q2i:
        return clusters

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

        # usuń identyczne reps w obrębie nowego klastra
        merged[new_id] = list(dict.fromkeys(merged[new_id]))
        new_id += 1

    return merged

def pick_main_phrase(qs: List[str]) -> str:
    return sorted(qs, key=lambda x: (len(normalize_phrase(x)), len(str(x))))[0] if qs else ""

def generate_article_brief(questions: List[str], client: OpenAI | None, model: str) -> Dict[str, Any]:
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
            low = line.lower().strip()
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
# UI
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

def save_checkpoint(data, filename="briefs.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_checkpoint(filename="briefs.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return []

# -----------------------------
# Main logic
# -----------------------------
if st.sidebar.button("Uruchom grupowanie"):
    if not phrases_input.strip():
        st.warning("⚠️ Wklej najpierw listę fraz.")
        st.stop()

    if not OPENAI_API_KEY:
        st.error("⚠️ Podaj OpenAI API Key w panelu bocznym.")
        st.stop()

    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    questions_raw = [line.strip() for line in phrases_input.splitlines() if line.strip()]
    update_status(f"📥 Wczytano frazy: {len(questions_raw)}", 5)

    # ✅ Opcjonalnie: SemHash może “odszumić”, ale MAPA zawsze RapidFuzz (żeby nic nie zginęło)
    if USE_SEMHASH:
        sem = semhash_deduplicate(questions_raw, threshold=SEMHASH_SIM)
        update_status(f"🧹 SemHash (info): {len(questions_raw)} → {len(sem)}", 12)

    reps, rep2all = deduplicate_with_map(questions_raw, threshold=DEDUP_THRESHOLD)
    update_status(f"🧹 Dedup (bez utraty fraz): {len(questions_raw)} → {len(reps)} reprezentantów", 20)

    update_status("🧠 Generowanie embeddingów...", 35)
    lemmatized = lemmatize_texts(reps)
    embeddings = embed_texts(openai_client, lemmatized, model=OPENAI_EMBEDDING_MODEL)
    q2i = {q: i for i, q in enumerate(reps)}

    clusters = cluster_questions(reps, embeddings, sim_threshold=CLUSTER_SIM)
    update_status(f"🧩 Klastrowanie reps: {len(clusters)} klastrów", 55)

    clusters = merge_similar_clusters(clusters, embeddings, sim_threshold=MERGE_SIM, q2i=q2i)
    update_status(f"🔗 Scalanie klastrów: {len(clusters)} klastrów", 70)

    # ✅ KLUCZ: rozszerzamy każdy klaster reps -> wszystkie frazy, nic nie ginie
    clusters_full: Dict[int, List[str]] = {}
    for cid, reps_in_cluster in clusters.items():
        all_phrases = []
        for r in reps_in_cluster:
            all_phrases.extend(rep2all.get(r, [r]))
        clusters_full[cid] = list(dict.fromkeys(all_phrases))  # usuń identyczne

    update_status(f"📦 Przywrócono pełne frazy do klastrów (łącznie: {sum(len(v) for v in clusters_full.values())})", 82)
    update_status(f"✅ Pominięto globalne kasowanie między klastrami – nic nie przepada", 90)

    # -----------------------------
    # Checkpoint + briefy
    # -----------------------------
    rows = load_checkpoint()
    done = len(rows)
    total = len(clusters_full)

    if done > 0:
        update_status(f"🔁 Wczytano {done} gotowych briefów z checkpointa", 90)
    else:
        update_status("📝 Rozpoczynam generowanie briefów od początku", 90)

    # ważne: iterujemy po clusters_full (pełne frazy do użycia)
    items = list(clusters_full.items())

    for i, (label, full_qs) in enumerate(items, 1):
        if i <= done:
            continue
        try:
            update_status(f"📝 Generuję brief {i}/{total} ({len(full_qs)} fraz)", int(95 * i / max(total, 1)))
            brief = generate_article_brief(full_qs, openai_client, model=OPENAI_CHAT_MODEL)

            rows.append({
                "cluster_id": label,
                "main_phrase": pick_main_phrase(full_qs),
                "intencja": brief.get("intencja", ""),
                "frazy_do_uzycia": ", ".join(full_qs),   # ✅ pełna pula do artykułu
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






