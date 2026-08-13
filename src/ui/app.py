from __future__ import annotations

import csv
import os
import sys
import time
from typing import Dict, List

import streamlit as st

from src.core.rag import (
    run_rag_query,
    DEFAULT_MAX_PER_SOURCE_RECHERCHE,
    DEFAULT_MAX_PER_SOURCE_RESUME,
)
from src.core.update_pipeline import (
    run_update_pipeline,
    ConversionStepConfig,
    ScanExtractStepConfig,
    ChunkStepConfig,
    IndexStepConfig,
)
from src.utils.paths import REGISTRY_DIR

st.set_page_config(
    page_title="Assistant documentaire",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Assistant documentaire")

DEFAULT_MAX_PER_SOURCE = {
    "Recherche": DEFAULT_MAX_PER_SOURCE_RECHERCHE,
    "Résumé": DEFAULT_MAX_PER_SOURCE_RESUME,
}

if "messages" not in st.session_state:
    st.session_state.messages = []
if "confirm_update" not in st.session_state:
    st.session_state.confirm_update = False
if "shutdown_pending" not in st.session_state:
    st.session_state.shutdown_pending = False


@st.cache_data(show_spinner=False)
def load_filter_values() -> Dict[str, List[str]]:
    registry_path = (REGISTRY_DIR / "sources.csv").resolve()
    values = {"nature": set(), "ext": set(), "filename": set()}
    if not registry_path.exists():
        return {key: [] for key in values}
    with registry_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for key in values:
                raw_value = (row.get(key) or "").strip()
                if raw_value:
                    values[key].add(raw_value)
    return {key: sorted(items) for key, items in values.items()}


def citation_sort_key(citation: Dict[str, object]) -> int:
    citation_label = str(citation.get("label", "")).strip()
    if citation_label.startswith("C"):
        try:
            return int(citation_label[1:])
        except ValueError:
            return 10_000
    return 10_000


def format_unit_type(raw: object) -> str:
    mapping = {
        "page": "Page",
        "paragraph": "Paragraphe",
        "slide": "Slide",
    }
    text = str(raw or "").strip()
    return mapping.get(text.lower(), text)


with st.sidebar:
    st.header("Paramètres")
    debug_mode = "--debug" in sys.argv
    if debug_mode:
        st.caption("Mode debug activé (via --debug).")

    st.subheader("RAG")
    mode = st.segmented_control(
        "Mode de réponse",
        ["Recherche", "Résumé"],
        default="Recherche",
    )
    if not mode:
        mode = "Recherche"

    if "max_per_source_mode" not in st.session_state:
        st.session_state.max_per_source_mode = mode
        st.session_state.max_per_source = DEFAULT_MAX_PER_SOURCE.get(mode, 2)

    if st.session_state.max_per_source_mode != mode:
        st.session_state.max_per_source_mode = mode
        st.session_state.max_per_source = DEFAULT_MAX_PER_SOURCE.get(mode, 2)

    max_per_source = st.number_input(
        "Passages par document",
        min_value=1,
        step=1,
        help="Le système peut ajuster ce nombre en fonction de la pertinence des passages.",
        key="max_per_source",
    )

    st.subheader("Filtres")
    filter_values = load_filter_values()
    nature_options = filter_values["nature"]
    ext_options = filter_values["ext"]
    filename_options = filter_values["filename"]

    st.multiselect("Nature", nature_options, key="filter_nature", placeholder="Toutes")
    st.multiselect("Type de fichier", ext_options, key="filter_ext", placeholder="Tous")
    st.multiselect("Nom du fichier", filename_options, key="filter_filename", placeholder="Tous")

    has_filters = bool(
        st.session_state.get("filter_nature")
        or st.session_state.get("filter_ext")
        or st.session_state.get("filter_filename")
    )
    if has_filters:
        def clear_filters() -> None:
            st.session_state["filter_nature"] = []
            st.session_state["filter_ext"] = []
            st.session_state["filter_filename"] = []

        st.button("Réinitialiser les filtres", on_click=clear_filters)

    st.space()
    if st.session_state.shutdown_pending:
        st.warning("Confirmer l'arrêt de l'application ?")
        col_stop, col_cancel = st.columns([1, 1], gap="small")
        with col_stop:
            if st.button("Arrêter", use_container_width=True, type="primary"):
                st.session_state.shutdown_pending = False
                st.info("Arrêt en cours…")
                time.sleep(0.2)
                os._exit(0)
        with col_cancel:
            if st.button("Annuler", use_container_width=True):
                st.session_state.shutdown_pending = False
                st.rerun()
    else:
        if st.button("Quitter l'application", use_container_width=True):
            st.session_state.shutdown_pending = True
            st.rerun()

    top_k = 0
    context_max_tokens = 0
    neighbor_expansion = 1
    mmr = False
    include_parent = False
    no_parent = False
    debug = False
    run_conversion_step = True
    run_scan_extract_step = True
    run_chunk_step = True
    run_index_step = True

    if debug_mode:
        with st.expander("RAG (avancé)", expanded=True):
            top_k = st.number_input(
                "Top‑k",
                min_value=0,
                value=0,
                step=1,
                help="0 = valeur par défaut du système.",
            )
            context_max_tokens = st.number_input(
                "Budget contexte (tokens)",
                min_value=0,
                value=0,
                step=100,
                help="0 = valeur par défaut du système.",
            )
            neighbor_expansion = st.number_input(
                "Voisins (expansion)",
                min_value=0,
                value=1,
                step=1,
                help="Nombre de voisins ajoutés autour des passages pertinents.",
            )
            mmr = st.checkbox("MMR", value=False)
            include_parent = st.checkbox("Inclure parents (résumé)", value=False)
            no_parent = st.checkbox("Désactiver parents", value=False)
            debug = st.checkbox("Debug", value=False)

        with st.expander("Mise à jour (avancé)", expanded=False):
            run_conversion_step = st.checkbox("Conversion legacy", value=True)
            run_scan_extract_step = st.checkbox("Scan + extraction", value=True)
            run_chunk_step = st.checkbox("Chunking", value=True)
            run_index_step = st.checkbox("Index", value=True)


tab_chat, tab_update = st.tabs(["Conversation", "Mise à jour"])

with tab_chat:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            citations = message.get("citations", []) or []
            if citations:
                with st.expander("Sources", expanded=False):
                    for item in sorted(citations, key=citation_sort_key):
                        label = item.get("label", "")
                        filename = item.get("filename", "")
                        file_path = item.get("path", "")
                        unit_type = format_unit_type(item.get("unit_type", ""))
                        unit_index = item.get("unit_index", "")

                        subtitle = f"{filename} — {unit_type} {unit_index}"

                        st.markdown(f"**[{label}]** {subtitle}")
                        if file_path:
                            st.caption(file_path)

    user_input = st.chat_input("Posez une question")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        filters: List[str] = []
        for value in st.session_state.get("filter_nature", []):
            filters.append(f"nature={value}")
        for value in st.session_state.get("filter_ext", []):
            filters.append(f"ext={value}")
        for value in st.session_state.get("filter_filename", []):
            filters.append(f"filename={value}")

        with st.spinner("Recherche en cours…"):  # type: ignore
            try:
                output = run_rag_query(
                    query=user_input.strip(),
                    mode="resume" if mode == "Résumé" else "recherche",
                    top_k=int(top_k),
                    max_per_source=int(max_per_source),
                    neighbor_expansion=int(neighbor_expansion),
                    mmr=bool(mmr),
                    include_parent=bool(include_parent),
                    no_parent=bool(no_parent),
                    context_max_tokens=int(context_max_tokens),
                    filters=filters,
                    debug=bool(debug),
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Erreur RAG : {exc}")
                st.stop()

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": output.get("answer", ""),
                "citations": output.get("citations", []),
            }
        )
        st.rerun()

with tab_update:
    st.subheader("Mise à jour de la base")
    st.caption("Cette opération peut consommer des crédits si des documents ont changé.")

    # --- State machine ---
    # states: "idle" -> "confirm" -> "running" -> "done" (or "error")
    if "update_state" not in st.session_state:
        st.session_state.update_state = "idle"
    if "last_update_result" not in st.session_state:
        st.session_state.last_update_result = None
    if "last_update_error" not in st.session_state:
        st.session_state.last_update_error = None

    def request_confirm() -> None:
        st.session_state.update_state = "confirm"

    def cancel_confirm() -> None:
        st.session_state.update_state = "idle"

    def start_update() -> None:
        st.session_state.update_state = "running"
        st.session_state.last_update_result = None
        st.session_state.last_update_error = None

    # --- UI: idle ---
    if st.session_state.update_state == "idle":
        st.button("Mettre à jour", on_click=request_confirm)

    # --- UI: confirm ---
    elif st.session_state.update_state == "confirm":
        st.warning("Confirmer la mise à jour de la base maintenant ?")

        col_confirm, col_cancel, _ = st.columns([1, 1, 3], gap="small")
        with col_confirm:
            if st.button("Confirmer", use_container_width=True):
                start_update()
                st.rerun()
        with col_cancel:
            st.button("Annuler", use_container_width=True, on_click=cancel_confirm)

    # --- UI: running ---
    elif st.session_state.update_state == "running":
        status = st.status("Mise à jour en cours…", state="running", expanded=True)

        try:
            status.write("Exécution du pipeline…")

            result = run_update_pipeline(
                conversion=ConversionStepConfig(enabled=run_conversion_step),
                scan_extract=ScanExtractStepConfig(enabled=run_scan_extract_step),
                chunk=ChunkStepConfig(enabled=run_chunk_step),
                index=IndexStepConfig(enabled=run_index_step),
            )

            st.session_state.last_update_result = result
            st.session_state.update_state = "done"
            status.update(label="Mise à jour terminée.", state="complete", expanded=False)

        except Exception as exc:  # noqa: BLE001
            st.session_state.last_update_error = str(exc)
            st.session_state.update_state = "error"
            status.update(label="Échec de la mise à jour.", state="error", expanded=True)
            status.exception(exc)

        st.rerun()

    # --- UI: done ---
    elif st.session_state.update_state == "done":
        st.success("Mise à jour terminée.")
        if debug_mode and st.session_state.last_update_result is not None:
            st.json(st.session_state.last_update_result)

        col_a, col_b, _ = st.columns([1, 1, 3], gap="small")
        with col_a:
            if st.button("OK", use_container_width=True):
                st.session_state.update_state = "idle"
                st.rerun()
        with col_b:
            st.button("Relancer une mise à jour", use_container_width=True, on_click=request_confirm)

    # --- UI: error ---
    else:
        st.error("Erreur mise à jour.")
        if st.session_state.last_update_error:
            st.caption(st.session_state.last_update_error)

        col_a, col_b, _ = st.columns([1, 1, 3], gap="small")
        with col_a:
            if st.button("Retour", use_container_width=True):
                st.session_state.update_state = "idle"
                st.rerun()
        with col_b:
            st.button("Réessayer", use_container_width=True, on_click=request_confirm)
