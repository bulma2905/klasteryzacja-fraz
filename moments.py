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
from sklearn.cluster import AgglomerativeClustering
from rapidfuzz import fuzz
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

CLUSTER_SIM = st.sidebar.slider(
    "Initial Clustering Similarity Threshold", 0.0, 1.0, 0.80, 0.01
)
MERGE_SIM = st.sidebar.slider(
    "Cluster Merge Similarity Threshold", 0.0, 1.0, 0.85, 0.01
)

USE_LEMMATIZATION = st.sidebar.checkbox("Użyj lematyzacji (spaCy) do embeddingów", value=True)

# ✅ Post-dedup w obrębie klastra (dopiero w kroku 2)
POST_DEDUP_THRESHOLD = st.sidebar.slider(
    "Post-cluster dedup (RapidFuzz) – próg podobieństwa", 80, 100, 94, 1
)

# ✅ Limit fraz wysyłanych do GPT
MAX_PHRASES_FOR_GPT = st.sidebar.slider(
    "Limit fraz wysyłanych do GPT (reszta zostaje w Excelu)", 20, 300, 120, 10
)

# ✅ Batch embeddingów (stabilność)
EMBED_BATCH_SIZE = st.sidebar.slider(
    "Batch embeddingów (stabilność)", 20, 500, 200, 20
)

# ✅ Iteracyjny merge: ile rund maks (zwykle 5–15 starczy)
MERGE_MAX_PASSES = st.sidebar.slider(
    "Maks. rund scalania klastrów (iteracyjnie)", 1, 30, 12, 1
)

# -----------------------------
# Checkpoint cleanup
# -----------------------------
if st.sidebar.button("🗑️ Wyczyść checkpoint briefów"):
    if os.path.exists("briefs.pkl"):
        os.remove("briefs.pkl")
        st.sidebar.success("Checkpoint został usunięty! 🔥")
    else:
        st.sidebar.info("Brak pliku checkpointa do usunięcia.")

st.sidebar.markdown("### ℹ️ Logika (ważne)")
st.sidebar.info("""
1) Przed embeddingami usuwamy tylko IDENTYCZNE duplikaty (po normalizacji).
2) Klastrowanie robi się na embeddingach (bez fuzz/semhash przed embeddingami).
3) Dopiero PO klastrach robimy semantyczne sprzątanie wewnątrz klastra (RapidFuzz),
   a do GPT wysyłamy limitowaną listę reprezentantów, ale pełna pula zostaje w Excelu.
4) Scalanie klastrów jest ITERACYJNE, żeby uniknąć “dwóch artykułów o tym samym”.
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
    out = []
    for t in texts:
        doc = nlp(t)
        lemmas = [token.lemma_.lower() for token in doc if token.lemma_]
        out.append(" ".join(lemmas) if lemmas else t)
    return out


# -----------------------------
# Normalizacja + exact dedup (tylko identyczne)
# -----------------------------
def normalize_phrase(s: str) -> str:
    s = str(s).strip().lower()
    s = unidecode.unidecode(s)
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def exact_dedup_keep_first(phrases: List[str]) -> List[str]:
    seen = set()
    out = []
    for p in phrases:
        np_ = normalize_phrase(p)
        if np_ and np_ not in seen:
            out.append(p)
            seen.add(np_)
    return out


# -----------------------------
# Embeddings (batch)
# -----------------------------
def embed_texts_batched(client: OpenAI, texts: List[str], model: str, batch_size: int = 200) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=float)

    all_vecs: List[List[float]] = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        all_vecs.extend([d.embedding for d in resp.data])

        # minimalny „oddech” żeby nie walić rate-limitów
        if i + batch_size < total:
            time.sleep(0.05)

    return np.array(all_vecs, dtype=float)


# -----------------------------
# Clustering
# -----------------------------
def cluster_questions(questions: List[str], embeddings: np.ndarray, sim_threshold: float) -> Dict[int, List[str]]:
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

def _centroid(vecs: np.ndarray) -> np.ndarray:
    c = np.mean(vecs, axis=0)
    n = np.linalg.norm(c) + 1e-12
    return c / n

def merge_similar_clusters_iterative(
    clusters: Dict[int, List[str]],
    embeddings: np.ndarray,
    sim_threshold: float,
    q2i: Dict[str, int],
    max_passes: int = 12
) -> Dict[int, List[str]]:
    """
    Iteracyjnie scala klastry po podobieństwie centroidów.
    Dzięki temu domyka łańcuchy typu: A~B i B~C => A,B,C w 1 klastrze.
    """
    if not clusters:
        return clusters

    # Start: przepisz do listy klastrów jako listy fraz
    current = [list(dict.fromkeys(qs)) for _, qs in clusters.items()]

    for _pass in range(max_passes):
        # policz centroidy
        centroids = []
        valid_idx = []
        for idx, qs in enumerate(current):
            idxs = [q2i[q] for q in qs if q in q2i]
            if not idxs:
                centroids.append(None)
                continue
            centroids.append(_centroid(embeddings[idxs]))
            valid_idx.append(idx)

        used = set()
        new_clusters = []
        changed = False

        for i in range(len(current)):
            if i in used:
                continue
            if centroids[i] is None:
                used.add(i)
                new_clusters.append(current[i])
                continue

            group = list(current[i])
            used.add(i)

            # Scalaj wszystko co przekracza próg (w tej rundzie)
            for j in range(i + 1, len(current)):
                if j in used or centroids[j] is None:
                    continue
                sim = float(np.dot(centroids[i], centroids[j]))
                if sim >= sim_threshold:
                    group.extend(current[j])
                    used.add(j)
                    changed = True

            # usuń literalne duplikaty, zachowaj kolejność
            group = list(dict.fromkeys(group))
            new_clusters.append(group)

        current = new_clusters
        if not changed:
            break

    # Nadaj stabilne ID: sort po main_phrase (żeby checkpoint nie wariował)
    def main_key(qs: List[str]) -> str:
        return sorted(qs, key=lambda x: (len(normalize_phrase(x)), len(str(x))))[0] if qs else ""

    current_sorted = sorted(current, key=main_key)

    merged: Dict[int, List[str]] = {i: qs for i, qs in enumerate(current_sorted)}
    return merged

def pick_main_phrase(qs: List[str]) -> str:
    return sorted(qs, key=lambda x: (len(normalize_phrase(x)), len(str(x))))[0] if qs else ""


# -----------------------------
# Post-cluster: semantyczne sprzątanie w obrębie klastra (bez utraty w Excelu)
# -----------------------------
def dedup_semantic_keep_all(qs: List[str], threshold: int) -> Tuple[List[str], Dict[str, List[str]]]:
    reps: List[str] = []
    reps_norm: List[str] = []
    rep2all: Dict[str, List[str]] = {}

    for q in qs:
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

def pick_reps_for_gpt(reps: List[str], rep2all: Dict[str, List[str]], limit: int) -> List[str]:
    reps_sorted = sorted(reps, key=lambda r: len(rep2all.get(r, [r])), reverse=True)
    return reps_sorted[:limit]


# -----------------------------
# Brief generation (etap 2)
# -----------------------------
def generate_article_brief(questions: List[str], client: OpenAI | None, model: str) -> Dict[str, Any]:
    if client is None:
        return {"intencja": "", "tytul": "", "wytyczne": ""}

    prompt = f"""
Dla poniższej listy fraz przygotuj dane do planu artykułu.

Frazy: {questions}

Odpowiedz w formacie:

Intencja: [typ intencji wyszukiwania]
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
        result = {"intencja": "", "tytul": "", "wytyczne": ""}

        for line in content.splitlines():
            low = line.lower().strip()
            if low.startswith("intencja:"):
                result["intencja"] = line.split(":", 1)[1].strip()
            elif low.startswith("tytuł:") or low.startswith("tytul:"):
                result["tytul"] = line.split(":", 1)[1].strip()
            elif low.startswith("wytyczne:"):
                result["wytyczne"] = line.split(":", 1)[1].strip()

        return result
    except Exception as e:
        logging.warning(f"⚠️ Brief parse failed: {e}")
        return {"intencja": "", "tytul": "", "wytyczne": ""}


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
# BUTTON 1: ANALIZA (KLASTRY)
# -----------------------------
if st.sidebar.button("1) Uruchom analizę klastrów"):
    if not phrases_input.strip():
        st.warning("⚠️ Wklej najpierw listę fraz.")
        st.stop()

    if not OPENAI_API_KEY:
        st.error("⚠️ Podaj OpenAI API Key w panelu bocznym.")
        st.stop()

    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    raw = [line.strip() for line in phrases_input.splitlines() if line.strip()]
    update_status(f"📥 Wczytano frazy: {len(raw)}", 5)

    phrases = exact_dedup_keep_first(raw)
    update_status(f"🧼 Exact dedup (tylko identyczne): {len(raw)} → {len(phrases)}", 12)

    embed_inputs = lemmatize_texts(phrases) if USE_LEMMATIZATION else phrases

    update_status("🧠 Generowanie embeddingów (batch)...", 35)
    embeddings = embed_texts_batched(
        openai_client,
        embed_inputs,
        model=OPENAI_EMBEDDING_MODEL,
        batch_size=EMBED_BATCH_SIZE
    )
    q2i = {q: i for i, q in enumerate(phrases)}

    clusters = cluster_questions(phrases, embeddings, sim_threshold=CLUSTER_SIM)
    update_status(f"🧩 Klastrowanie: powstało {len(clusters)} klastrów", 60)

    clusters = merge_similar_clusters_iterative(
        clusters,
        embeddings,
        sim_threshold=MERGE_SIM,
        q2i=q2i,
        max_passes=MERGE_MAX_PASSES
    )
    update_status(f"🔗 Scalanie (iteracyjne): teraz {len(clusters)} klastrów", 80)

    st.session_state["clusters"] = clusters
    st.session_state["phrases_count"] = len(phrases)

    update_status("✅ Analiza gotowa. Teraz możesz generować briefy (krok 2).", 100)


# -----------------------------
# BUTTON 2: BRIEFY
# -----------------------------
if st.sidebar.button("2) Generuj briefy do klastrów"):
    if "clusters" not in st.session_state:
        st.warning("⚠️ Najpierw uruchom analizę klastrów (krok 1).")
        st.stop()

    if not OPENAI_API_KEY:
        st.error("⚠️ Podaj OpenAI API Key w panelu bocznym.")
        st.stop()

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    clusters: Dict[int, List[str]] = st.session_state["clusters"]

    rows = load_checkpoint()
    done = len(rows)
    total = len(clusters)

    if done > 0:
        update_status(f"🔁 Wczytano {done} gotowych briefów z checkpointa", 5)
    else:
        update_status("📝 Start generowania briefów", 5)

    items = list(clusters.items())

    for i, (cid, qs_full) in enumerate(items, 1):
        if i <= done:
            continue

        try:
            reps, rep2all = dedup_semantic_keep_all(qs_full, threshold=POST_DEDUP_THRESHOLD)
            reps_for_gpt = pick_reps_for_gpt(reps, rep2all, limit=MAX_PHRASES_FOR_GPT)

            update_status(
                f"📝 Brief {i}/{total} | klaster: {len(qs_full)} fraz | reps do GPT: {len(reps_for_gpt)}",
                int(95 * i / max(total, 1))
            )

            brief = generate_article_brief(reps_for_gpt, openai_client, model=OPENAI_CHAT_MODEL)

            rows.append({
                "cluster_id": cid,
                "main_phrase": pick_main_phrase(qs_full),
                "intencja": brief.get("intencja", ""),
                "frazy_w_klastrze_pelne": ", ".join(qs_full),
                "frazy_reprezentatywne_do_GPT": ", ".join(reps_for_gpt),
                "tytul": brief.get("tytul", ""),
                "wytyczne": brief.get("wytyczne", ""),
            })

            save_checkpoint(rows)
            time.sleep(1.0)

        except Exception as e:
            logging.warning(f"⚠️ Błąd przy klastrze {i}/{total}: {e}")
            time.sleep(2)
            continue

    update_status("✅ Briefy gotowe.", 100)
    st.session_state["results"] = rows


# -----------------------------
# EXPORT EXCEL
# -----------------------------
if "clusters" in st.session_state:
    clusters = st.session_state["clusters"]

    clusters_rows = []
    for cid, qs in clusters.items():
        clusters_rows.append({
            "cluster_id": cid,
            "main_phrase": pick_main_phrase(qs),
            "liczba_fraz": len(qs),
            "frazy_w_klastrze_pelne": ", ".join(qs),
        })
    df_clusters = pd.DataFrame(clusters_rows).sort_values(by="liczba_fraz", ascending=False)

    df_briefs = pd.DataFrame(st.session_state.get("results", []))

    xlsx_buffer = io.BytesIO()
    with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
        df_clusters.to_excel(writer, sheet_name="Klastry", index=False)
        if not df_briefs.empty:
            df_briefs.to_excel(writer, sheet_name="Briefy", index=False)

    xlsx_buffer.seek(0)

    st.download_button(
        label="📥 Pobierz Excel (Klastry + Briefy jeśli są)",
        data=xlsx_buffer.getvalue(),
        file_name="frazy_klastry_briefy.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.subheader("📊 Podgląd klastrów")
    st.dataframe(df_clusters)

    if not df_briefs.empty:
        st.subheader("📝 Podgląd briefów")
        st.dataframe(df_briefs)









