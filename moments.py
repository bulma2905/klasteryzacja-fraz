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
from sklearn.metrics.pairwise import cosine_similarity
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

# -----------------------------
# Sidebar Configuration
# -----------------------------
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

# --- KROK 1.5 (NOWE): kanibalizacja między klastrami, bez briefów
st.sidebar.markdown("---")
st.sidebar.subheader("🧩 Krok 1.5: Kanibalizacja klastrów (przed briefami)")
CANN_METHOD = st.sidebar.radio("Metoda wykrywania podobnych klastrów", ["RapidFuzz", "Embeddingi OpenAI"], index=0)

if CANN_METHOD == "RapidFuzz":
    CANN_THRESHOLD_FUZZ = st.sidebar.slider("Próg podobieństwa (RapidFuzz token_set_ratio)", 70, 100, 88, 1)
else:
    CANN_THRESHOLD_EMB = st.sidebar.slider("Próg podobieństwa (cosine similarity)", 0.70, 0.99, 0.83, 0.01)

CANN_TOP_PHRASES = st.sidebar.slider("Ile fraz z klastra brać do porównania CMP", 5, 80, 30, 5)

# --- KROK 2 (briefy): post-dedup i limit do GPT
st.sidebar.markdown("---")
st.sidebar.subheader("📝 Krok 2: Briefy")
POST_DEDUP_THRESHOLD = st.sidebar.slider(
    "Post-cluster dedup (RapidFuzz) – próg podobieństwa", 80, 100, 94, 1
)
MAX_PHRASES_FOR_GPT = st.sidebar.slider(
    "Limit fraz wysyłanych do GPT (reszta zostaje w Excelu)", 20, 300, 120, 10
)

# -----------------------------
# Checkpoint cleanup
# -----------------------------
colA, colB = st.sidebar.columns(2)

with colA:
    if st.sidebar.button("🗑️ Wyczyść checkpoint briefów"):
        if os.path.exists("briefs.pkl"):
            os.remove("briefs.pkl")
            st.sidebar.success("Checkpoint briefów usunięty! 🔥")
        else:
            st.sidebar.info("Brak pliku briefs.pkl")

with colB:
    if st.sidebar.button("🧹 Reset scalania 1.5"):
        st.session_state.pop("merge_log_df", None)
        st.session_state.pop("merge_groups_cid", None)
        st.sidebar.success("Reset podglądu scalania 1.5 ✅")

st.sidebar.markdown("### ℹ️ Logika (ważne)")
st.sidebar.info("""
1) Krok 1: robimy klastry na embeddingach (bez semantycznego dedupu).
2) Krok 1.5: wykrywamy podobne KLASTRY i je scalamy (bez briefów).
3) Krok 2: dopiero po scaleniu generujemy briefy (z post-dedup i limitem do GPT).
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
        out.append(" ".join([token.lemma_.lower() for token in doc if token.lemma_]))
    return out

# -----------------------------
# Normalizacja + exact dedup
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
# Embeddings + clustering
# -----------------------------
def embed_texts(client: OpenAI, texts: List[str], model: str) -> np.ndarray:
    response = client.embeddings.create(model=model, input=texts)
    return np.array([d.embedding for d in response.data])

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

def merge_similar_clusters(
    clusters: Dict[int, List[str]],
    embeddings: np.ndarray,
    sim_threshold: float,
    q2i: Dict[str, int]
) -> Dict[int, List[str]]:
    if not clusters:
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

        merged[new_id] = list(dict.fromkeys(merged[new_id]))
        new_id += 1

    return merged

def pick_main_phrase(qs: List[str]) -> str:
    return sorted(qs, key=lambda x: (len(normalize_phrase(x)), len(str(x))))[0] if qs else ""

# -----------------------------
# ✅ KROK 1.5: Kanibalizacja między klastrami (bez briefów)
# -----------------------------
def build_cluster_cmp(main_phrase: str, phrases: List[str], top_n: int) -> str:
    phrases_sorted = sorted(phrases, key=lambda x: (len(normalize_phrase(x)), len(str(x))))
    top = phrases_sorted[:top_n]
    return normalize_phrase(main_phrase) + " | " + " | ".join([normalize_phrase(x) for x in top])

def merge_clusters_by_groups(groups: List[List[int]], clusters: Dict[int, List[str]]) -> Dict[int, List[str]]:
    merged: Dict[int, List[str]] = {}
    used = set()
    new_id = 0

    for group in groups:
        all_phrases = []
        for cid in group:
            all_phrases.extend(clusters.get(cid, []))
        merged[new_id] = list(dict.fromkeys(all_phrases))
        used.update(group)
        new_id += 1

    for cid, qs in clusters.items():
        if cid not in used:
            merged[new_id] = qs
            new_id += 1

    return merged

def find_groups_graph(edges: List[Tuple[int,int]], n: int) -> List[List[int]]:
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    visited = [False]*n
    groups = []

    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        comp = []
        visited[i] = True
        while stack:
            v = stack.pop()
            comp.append(v)
            for nb in adj[v]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        if len(comp) > 1:
            groups.append(comp)

    return groups

def cannibalize_clusters_fuzz(clusters: Dict[int, List[str]], top_n: int, threshold: int) -> Tuple[Dict[int, List[str]], int, List[List[int]]]:
    cids = list(clusters.keys())
    cmps = []
    for cid in cids:
        qs = clusters[cid]
        main = pick_main_phrase(qs)
        cmps.append(build_cluster_cmp(main, qs, top_n))

    edges = []
    for i in range(len(cids)):
        for j in range(i+1, len(cids)):
            sim = fuzz.token_set_ratio(cmps[i], cmps[j])
            if sim >= threshold:
                edges.append((cids[i], cids[j]))

    cid_to_idx = {cid:i for i, cid in enumerate(cids)}
    idx_edges = [(cid_to_idx[a], cid_to_idx[b]) for a,b in edges]

    groups_idx = find_groups_graph(idx_edges, len(cids))
    groups_cid = [[cids[i] for i in g] for g in groups_idx]

    merged = merge_clusters_by_groups(groups_cid, clusters)
    return merged, len(groups_cid), groups_cid

def cannibalize_clusters_emb(client: OpenAI, clusters: Dict[int, List[str]], top_n: int, threshold: float, model: str) -> Tuple[Dict[int, List[str]], int, List[List[int]]]:
    cids = list(clusters.keys())
    cmps = []
    for cid in cids:
        qs = clusters[cid]
        main = pick_main_phrase(qs)
        cmps.append(build_cluster_cmp(main, qs, top_n))

    emb = embed_texts(client, cmps, model=model)
    sim = cosine_similarity(emb)

    edges_idx = []
    for i in range(len(cids)):
        for j in range(i+1, len(cids)):
            if float(sim[i, j]) >= threshold:
                edges_idx.append((i, j))

    groups_idx = find_groups_graph(edges_idx, len(cids))
    groups_cid = [[cids[i] for i in g] for g in groups_idx]

    merged = merge_clusters_by_groups(groups_cid, clusters)
    return merged, len(groups_cid), groups_cid

def build_merge_log(
    groups_cid: List[List[int]],
    clusters: Dict[int, List[str]],
    preview_per_cluster: int = 5,
    max_preview_total: int = 800
) -> pd.DataFrame:
    rows = []
    for gid, group in enumerate(groups_cid, 1):
        sizes = [len(clusters.get(cid, [])) for cid in group]

        preview_parts = []
        for cid in group:
            qs = clusters.get(cid, [])
            qs_sorted = sorted(qs, key=lambda x: (len(normalize_phrase(x)), len(str(x))))
            sample = qs_sorted[:preview_per_cluster]
            preview_parts.append(f"[{cid}] " + " | ".join(sample))

        preview_joined = "  ||  ".join(preview_parts)
        if len(preview_joined) > max_preview_total:
            preview_joined = preview_joined[:max_preview_total].rstrip() + "…"

        rows.append({
            "merge_group_id": gid,
            "merged_cluster_ids": ", ".join(map(str, group)),
            "merged_clusters_count": len(group),
            "cluster_sizes": ", ".join(map(str, sizes)),
            "total_phrases_after_merge": int(sum(sizes)),
            "sample_main_phrases": " | ".join([pick_main_phrase(clusters.get(cid, [])) for cid in group][:6]),
            "preview_fraz": preview_joined,
        })

    return pd.DataFrame(rows)

# -----------------------------
# ✅ KROK 2: Post-cluster dedup do promptu (bez utraty w Excelu)
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

    update_status("🧠 Generowanie embeddingów (bez semantycznego dedupu)...", 35)
    embeddings = embed_texts(openai_client, embed_inputs, model=OPENAI_EMBEDDING_MODEL)
    q2i = {q: i for i, q in enumerate(phrases)}

    clusters = cluster_questions(phrases, embeddings, sim_threshold=CLUSTER_SIM)
    update_status(f"🧩 Klastrowanie: powstało {len(clusters)} klastrów", 60)

    clusters = merge_similar_clusters(clusters, embeddings, sim_threshold=MERGE_SIM, q2i=q2i)
    update_status(f"🔗 Scalanie podobnych klastrów: teraz {len(clusters)} klastrów", 75)

    st.session_state["clusters"] = clusters
    st.session_state["clusters_stage"] = "po_analizie"
    st.session_state.pop("merge_log_df", None)
    st.session_state.pop("merge_groups_cid", None)

    update_status("✅ Krok 1 gotowy. Teraz możesz zrobić 1.5 (kanibalizacja klastrów) albo od razu briefy.", 100)

# -----------------------------
# BUTTON 1.5: KANIBALIZACJA KLASTRÓW (BEZ BRIEFÓW)
# -----------------------------
if st.sidebar.button("1.5) Wykryj kanibalizację między klastrami (bez briefów)"):
    if "clusters" not in st.session_state:
        st.warning("⚠️ Najpierw uruchom analizę klastrów (krok 1).")
        st.stop()

    clusters: Dict[int, List[str]] = st.session_state["clusters"]
    update_status(f"🔎 Krok 1.5: analizuję kanibalizację na {len(clusters)} klastrach...", 10)

    if CANN_METHOD == "RapidFuzz":
        merged, groups_found, groups_cid = cannibalize_clusters_fuzz(
            clusters=clusters,
            top_n=CANN_TOP_PHRASES,
            threshold=CANN_THRESHOLD_FUZZ
        )
        update_status(f"✅ Krok 1.5: znaleziono {groups_found} grup do scalania | klastry: {len(clusters)} → {len(merged)}", 100)
    else:
        if not OPENAI_API_KEY:
            st.error("⚠️ Podaj OpenAI API Key (embeddingi są potrzebne).")
            st.stop()
        client = OpenAI(api_key=OPENAI_API_KEY)
        merged, groups_found, groups_cid = cannibalize_clusters_emb(
            client=client,
            clusters=clusters,
            top_n=CANN_TOP_PHRASES,
            threshold=CANN_THRESHOLD_EMB,
            model=OPENAI_EMBEDDING_MODEL
        )
        update_status(f"✅ Krok 1.5: znaleziono {groups_found} grup do scalania | klastry: {len(clusters)} → {len(merged)}", 100)

    # log scalania (żeby było widać CO i DLACZEGO)
    merge_log_df = build_merge_log(groups_cid, clusters, preview_per_cluster=5, max_preview_total=1200)

    st.session_state["clusters"] = merged
    st.session_state["clusters_stage"] = "po_kanibalizacji"
    st.session_state["merge_log_df"] = merge_log_df
    st.session_state["merge_groups_cid"] = groups_cid

    st.warning("ℹ️ Po scaleniu klastrów w kroku 1.5 warto wyczyścić checkpoint briefów, jeśli był już robiony wcześniej.")

# -----------------------------
# BUTTON 2: BRIEFY (po kanibalizacji klastrów)
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
# PODGLĄD: co było scalone w kroku 1.5
# -----------------------------
st.markdown("## Pipeline")
if "clusters_stage" in st.session_state:
    st.info(f"Etap: **{st.session_state['clusters_stage']}**")

if "merge_log_df" in st.session_state and isinstance(st.session_state["merge_log_df"], pd.DataFrame):
    st.subheader("🔗 Podgląd scalenia 1.5 (co zostało połączone)")
    st.dataframe(st.session_state["merge_log_df"], use_container_width=True)

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
    df_merge = st.session_state.get("merge_log_df", pd.DataFrame())

    xlsx_buffer = io.BytesIO()
    with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
        df_clusters.to_excel(writer, sheet_name="Klastry", index=False)
        if isinstance(df_merge, pd.DataFrame) and not df_merge.empty:
            df_merge.to_excel(writer, sheet_name="Scalenia_1_5", index=False)
        if not df_briefs.empty:
            df_briefs.to_excel(writer, sheet_name="Briefy", index=False)

    xlsx_buffer.seek(0)

    st.download_button(
        label="📥 Pobierz Excel (Klastry + Scalenia_1_5 + Briefy jeśli są)",
        data=xlsx_buffer.getvalue(),
        file_name="frazy_klastry_briefy.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.subheader("📊 Podgląd klastrów")
    st.dataframe(df_clusters, use_container_width=True)

    if not df_briefs.empty:
        st.subheader("📝 Podgląd briefów")
        st.dataframe(df_briefs, use_container_width=True)












