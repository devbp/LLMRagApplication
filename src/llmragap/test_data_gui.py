"""Streamlit page: generate RAG test cases from chunks already stored in Chroma, then run them.

Uses the functions of generate_test_data_chroma.py (generate) and eval_rag.py (evaluate).
Opened from main_app.py after login, or on its own:
    uv run streamlit run src/llmragap/test_data_gui.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from deepeval.dataset import Golden

from llmragap.eval_rag import RagEvaluator, dated_results_path
from llmragap.generate_test_data_chroma import (
    COLLECTIONS,
    GENERATOR_MODEL,
    PROJECT_ROOT,
    contexts_from_chroma,
    default_output,
    generate_test_cases,
    save_test_cases,
)
from llmragap.app_resources import get_files_embeddings
from llmragap.files_embeddings import DEFAULT_SELECTION_MODE, SEARCH_MODES
from llmragap.RAG_Pdf import RagAgent
from llmragap.session_cost import (
    add_run_cost,
    init_session_costs,
    money,
    session_total_usd,
    show_cost_save_error,
)

EVAL_DATA = PROJECT_ROOT / "eval_data"
GENERATED = "generated"  # selectbox value for the test cases generated on this page

COLLECTION_LABELS = {
    "insurance_docs": "Insurance (VVG)",
    "civil_code_docs": "Civil code (BGB)",
    "medicine_docs": "Medicine",
}

st.set_page_config(page_title="Test case generator", page_icon=":material/fact_check:")
init_session_costs()  # same session totals as the chat page
st.title("Test case generator")
st.caption(
    "Builds contexts from chunks already stored in Chroma (no new embeddings), then asks "
    f"`{GENERATOR_MODEL}` for question/answer pairs grounded in each context."
)

with st.form("generate_form"):
    collection = st.selectbox(
        "Collection", COLLECTIONS, format_func=lambda c: COLLECTION_LABELS.get(c, c)
    )
    col1, col2, col3 = st.columns(3)
    n_contexts = col1.number_input("Contexts", min_value=1, max_value=50, value=5)
    chunks_per_context = col2.number_input("Chunks per context", min_value=1, max_value=10, value=3)
    per_context = col3.number_input("Test cases per context", min_value=1, max_value=5, value=2)
    seed = st.number_input(
        "Random seed (optional)", min_value=0, value=None, step=1, help="Same seed gives the same contexts."
    )
    submitted = st.form_submit_button(
        "Generate", icon=":material/auto_awesome:", type="primary", width="stretch"
    )

if submitted:
    try:
        with st.status("Generating test cases...", expanded=True) as status:
            st.write(f"Building {n_contexts} contexts from **{COLLECTION_LABELS.get(collection, collection)}**...")
            contexts = contexts_from_chroma(collection, int(n_contexts), int(chunks_per_context), seed)
            st.write(f"Asking the model for up to {per_context} test cases per context...")
            cases, usage = generate_test_cases(contexts, int(per_context))
            status.update(
                label=f"Generated {len(cases)} test cases · {money(usage['cost_usd'])}",
                state="complete",
                expanded=False,
            )
        st.session_state.test_cases = {"collection": collection, "cases": cases, "usage": usage}
        add_run_cost("testgen", len(cases), usage["tokens"], usage["cost_usd"])
    except ValueError as e:
        st.error(str(e))
    except Exception as e:  # model / network errors
        st.error(f"Generation failed: {e}")

result = st.session_state.get("test_cases")
if result:
    cases, collection_name = result["cases"], result["collection"]
    st.subheader(f"{len(cases)} test cases · {COLLECTION_LABELS.get(collection_name, collection_name)}")
    if usage := result.get("usage"):
        st.caption(
            f":material/payments: Generation cost {money(usage['cost_usd'])} · "
            f"{usage['calls']} model calls · {usage['tokens']:,} tokens"
        )

    st.dataframe(
        pd.DataFrame(
            {
                "Question": [c["input"] for c in cases],
                "Expected answer": [c["expected_output"] for c in cases],
                "Context chunks": [len(c["context"]) for c in cases],
            }
        ),
        hide_index=True,
        width="stretch",
    )

    for i, case in enumerate(cases, start=1):
        with st.expander(f"{i}. {case['input']}"):
            st.markdown(f"**Expected answer:** {case['expected_output']}")
            for j, chunk in enumerate(case["context"], start=1):
                st.caption(f"Context chunk {j}")
                st.text(chunk)

    output = default_output(collection_name)
    save_col, download_col = st.columns(2)
    if save_col.button(f"Save to eval_data/{output.name}", icon=":material/save:", width="stretch"):
        save_test_cases(cases, output)
        st.toast(f"Saved {len(cases)} test cases to {output.name}", icon=":material/check_circle:")
    download_col.download_button(
        "Download JSON",
        data=json.dumps(cases, indent=2, ensure_ascii=False),
        file_name=output.name,
        mime="application/json",
        icon=":material/download:",
        width="stretch",
    )


# ---------- run test cases with eval_rag.py ----------
def load_goldens(source: str) -> tuple[list[Golden], str]:
    """Goldens to run and the name used in the results file."""
    if source == GENERATED:
        generated = st.session_state.test_cases
        goldens = [
            Golden(input=c["input"], expected_output=c["expected_output"])
            for c in generated["cases"]
            if c["input"].strip()
        ]
        return goldens, generated["collection"]
    return RagEvaluator.load_goldens(source), Path(source).stem.removesuffix("_test_cases")


st.divider()
st.header("Run test cases")
st.caption(
    "Asks the RAG agent every question, then an LLM judge scores the answers "
    "(answer relevancy, faithfulness, contextual precision and recall). This makes many model calls."
)

sources = ([GENERATED] if result else []) + [str(p) for p in sorted(EVAL_DATA.glob("*_test_cases.json"))]
if not sources:
    st.info("Generate test cases above, or put a *_test_cases.json file in eval_data/.")
else:
    with st.form("eval_form"):
        source = st.selectbox(
            "Test cases",
            sources,
            format_func=lambda s: "Generated above" if s == GENERATED else f"eval_data/{Path(s).name}",
        )
        mode = st.selectbox("Search mode", SEARCH_MODES, index=SEARCH_MODES.index(DEFAULT_SELECTION_MODE))
        run_clicked = st.form_submit_button(
            "Run test cases", icon=":material/play_arrow:", type="primary", width="stretch"
        )

    if run_clicked:
        try:
            with st.status("Running test cases...", expanded=True) as status:
                goldens, name = load_goldens(source)
                output_path = dated_results_path(name)
                st.write(f"Asking the agent {len(goldens)} questions with {mode} search, then scoring the answers...")
                evaluator = RagEvaluator(RagAgent(get_files_embeddings(), mode, require_search=True))
                report = evaluator.run_goldens(goldens, output_path)
                summary, cost = report["summary"], report["cost"]
                status.update(
                    label=f"Done: {summary['passed']}/{summary['total']} passed · {money(cost['total_usd'])}",
                    state="complete",
                    expanded=False,
                )
            st.session_state.eval_result = {"path": output_path, "report": report}
            add_run_cost("eval", summary["total"], cost["tokens"], cost["total_usd"])
        except ValueError as e:
            st.error(str(e))
        except Exception as e:  # agent / judge / network errors
            st.error(f"Evaluation failed: {e}")

eval_result = st.session_state.get("eval_result")
if eval_result:
    report, path = eval_result["report"], Path(eval_result["path"])
    summary = report["summary"]
    st.success(f"Saved results to `eval_data/{path.name}`", icon=":material/check_circle:")
    if cost := report.get("cost"):
        st.caption(
            f":material/payments: Evaluation cost {money(cost['total_usd'])} — "
            f"agent {money(cost['agent_usd'])}, search {money(cost['search_usd'])}, "
            f"judge {money(cost['judge_usd'])} · {cost['model_calls']} model calls · {cost['tokens']:,} tokens"
        )
    no_retrieval = summary.get("no_retrieval", 0)
    if no_retrieval:
        st.warning(
            f"{no_retrieval} test case(s) retrieved no chunks: their contextual scores are 0 and "
            "their Faithfulness score means nothing (there was nothing to contradict).",
            icon=":material/warning:",
        )

    averages = summary["average_scores"]
    cols = st.columns(1 + len(averages))
    cols[0].metric("Passed", f"{summary['passed']}/{summary['total']}")
    for col, (metric, score) in zip(cols[1:], averages.items()):
        col.metric(metric.removesuffix(" Metric").removesuffix("Metric"), "–" if score is None else f"{score:.2f}")

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Question": r["input"],
                    "Passed": r["passed"],
                    "Chunks": r.get("retrieved_chunks", len(r.get("retrieval_context", []))),
                    **{name: m["score"] for name, m in r["metrics"].items()},
                }
                for r in report["results"]
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    for i, r in enumerate(report["results"], start=1):
        icon = ":material/check_circle:" if r["passed"] else ":material/cancel:"
        with st.expander(f"{i}. {r['input']}", icon=icon):
            st.markdown(f"**Expected:** {r['expected_output']}")
            st.markdown(f"**Agent answer:** {r['actual_output']}")
            if not r.get("retrieval_context"):
                st.warning("No chunks retrieved for this question.", icon=":material/search_off:")
            for name, m in r["metrics"].items():
                st.markdown(f"- **{name}**: {m['score']} ({'pass' if m['passed'] else 'fail'}) — {m['reason'] or m['error'] or ''}")

    st.download_button(
        "Download results JSON",
        data=json.dumps(report, indent=2, ensure_ascii=False),
        file_name=path.name,
        mime="application/json",
        icon=":material/download:",
        width="stretch",
    )


# ---------- session cost (drawn last, so it includes a run that just finished) ----------
show_cost_save_error()
with st.sidebar, st.container(border=True):
    testgen, evals = st.session_state.testgen_costs, st.session_state.eval_costs
    st.metric(":material/payments: Total cost this session", money(session_total_usd()), border=False)
    st.table(
        {
            "Part": ["Test generation", "Evaluation"],
            "Cost": [money(testgen["usd"]), money(evals["usd"])],
            "Details": [
                f"{testgen['runs']} run(s), {testgen['cases']} test cases, {testgen['tokens']:,} tokens",
                f"{evals['runs']} run(s), {evals['cases']} test cases, {evals['tokens']:,} tokens",
            ],
        },
        hide_index=True,
    )
    if st.session_state.get("username"):
        st.caption("Saved to your session cost in the database.")
    else:
        st.caption("Not logged in: costs are not saved to the database.")
