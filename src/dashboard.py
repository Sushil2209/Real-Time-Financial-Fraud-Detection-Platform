"""Review scored transactions. Run with: streamlit run src/dashboard.py."""

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st


DATABASE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "scored.db"
STATUSES = ("New", "Under Review", "Fraud", "Not Fraud")
RISK_LEVELS = ("high", "medium", "low")
RISK_LABELS = {"high": "🔴 High risk", "medium": "🟠 Medium risk", "low": "🟢 Low risk"}
RISK_HELP = {
    "high": "Fraud chance above 70%. Check these first.",
    "medium": "Fraud chance 30% to 70%. Worth a look.",
    "low": "Fraud chance below 30%. Probably safe.",
}
COLUMN_CONFIG = {
    "TransactionID": "Transaction ID",
    "card1": "Card ID",
    "TransactionAmt": st.column_config.NumberColumn("Amount", format="$%.2f"),
    "fraud_score": st.column_config.ProgressColumn(
        "Fraud chance", min_value=0.0, max_value=1.0, format="percent"
    ),
    "risk_level": "Risk",
}


def with_icons(table: pd.DataFrame) -> pd.DataFrame:
    """Copy of the table with colored risk labels, for display only."""
    return table.assign(risk_level=table["risk_level"].map(RISK_LABELS))


@st.cache_data(ttl=5)
def read_scores(database_path: Path) -> pd.DataFrame:
    """Read scores without creating a missing database."""
    with closing(sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        return pd.read_sql_query(
            "SELECT TransactionID, card1, TransactionAmt, fraud_score, risk_level FROM scores",
            connection,
        )


def create_feedback_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """CREATE TABLE IF NOT EXISTS feedback (
            TransactionID INTEGER PRIMARY KEY,
            status TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            note TEXT
        )"""
    )
    connection.commit()


def read_feedback(connection: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT TransactionID, status, updated_at, note FROM feedback", connection)


def save_feedback(connection: sqlite3.Connection, transaction_id: int, status: str, note: str) -> None:
    if status not in STATUSES:
        raise ValueError(f"Unknown case status: {status}")
    connection.execute(
        """INSERT INTO feedback (TransactionID, status, updated_at, note)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(TransactionID) DO UPDATE SET
            status = excluded.status,
            updated_at = excluded.updated_at,
            note = excluded.note""",
        (transaction_id, status, datetime.now(timezone.utc).isoformat(), note.strip()),
    )
    connection.commit()


def show_dashboard(connection: sqlite3.Connection, scores: pd.DataFrame) -> None:
    feedback = read_feedback(connection)
    counts = scores["risk_level"].value_counts().reindex(RISK_LEVELS, fill_value=0)

    # Feedback belongs to a transaction, even if it has been scored more than once.
    flagged = scores.loc[scores["risk_level"].isin(("medium", "high"))].sort_values(
        "fraud_score", ascending=False
    )
    cases = flagged.dropna(subset=["TransactionID"]).drop_duplicates(subset=["TransactionID"])
    cases = cases.merge(feedback, on="TransactionID", how="left")
    cases["status"] = cases["status"].fillna("New")
    waiting = int((cases["status"] == "New").sum())

    st.title("🛡️ Fraud Monitor")
    st.markdown(
        "A computer model checks every card payment and gives it a **fraud chance** "
        "(0% = surely safe, 100% = surely fraud). This page shows which payments look suspicious "
        "so a person can check them."
    )
    if st.session_state.pop("feedback_saved", False):
        st.success("Feedback saved.")

    # Plain-language summary first, numbers second.
    st.info(
        f"**{len(scores)}** payments were checked. **{int(counts['high'])}** look very risky "
        f"and **{int(counts['medium'])}** look a bit risky. **{waiting}** are still waiting for a person to check them."
    )
    columns = st.columns(4)
    columns[0].metric("Payments checked", len(scores), help="Every payment the model has scored.")
    for column, level in zip(columns[1:], RISK_LEVELS):
        column.metric(RISK_LABELS[level], int(counts[level]), help=RISK_HELP[level])
    st.metric("Waiting for a person to check", waiting, help="Risky payments nobody has reviewed yet.")

    st.header("Step 1. See the riskiest payments")
    threshold = st.sidebar.slider(
        "Show payments with fraud chance above", 0.0, 1.0, 0.8, 0.01,
        help="0.80 means 80%. Move it left to see more payments, right to see fewer.",
    )
    st.sidebar.caption("Example: 0.80 = show payments that are more than 80% likely to be fraud.")
    alerts = scores.loc[scores["fraud_score"] > threshold].sort_values("fraud_score", ascending=False)
    st.caption(f"Payments with a fraud chance above {threshold:.0%}, riskiest first.")
    if alerts.empty:
        st.info("No transactions exceed the alert threshold.")
    else:
        st.dataframe(with_icons(alerts), hide_index=True, column_config=COLUMN_CONFIG)

    with st.expander("Show all payments that need checking (medium and high risk)"):
        st.dataframe(with_icons(flagged), hide_index=True, column_config=COLUMN_CONFIG)
    if flagged["TransactionID"].isna().any():
        st.warning("Flagged rows without a TransactionID cannot be reviewed.")

    st.header("Step 2. Check a payment and record your decision")
    if cases.empty:
        st.info("There are no flagged transactions to review.")
    else:
        transaction_id = st.selectbox(
            "Choose a payment (Transaction ID)", cases["TransactionID"].tolist(),
            help="Only risky payments are listed, riskiest first.",
        )
        case = cases.loc[cases["TransactionID"] == transaction_id].iloc[0]
        st.write(
            f"Payment **{transaction_id}** of **${case['TransactionAmt']:,.2f}** has a "
            f"**{case['fraud_score']:.0%} fraud chance** ({RISK_LABELS[case['risk_level']]}). "
            f"Current status: **{case['status']}**."
        )
        with st.form(key=f"feedback_{transaction_id}"):
            status = st.selectbox(
                "Your decision", STATUSES, index=STATUSES.index(case["status"]),
                help="New = not checked yet. Under Review = being checked. "
                     "Fraud = confirmed fraud. Not Fraud = false alarm.",
            )
            note = st.text_area("Note (optional)", value=case["note"] if pd.notna(case["note"]) else "")
            if st.form_submit_button("Save feedback"):
                save_feedback(connection, int(transaction_id), status, note)
                st.session_state["feedback_saved"] = True
                st.rerun()

    st.header("Overview")
    st.caption("Number of payments in each risk level.")
    st.bar_chart(counts.rename(index=RISK_LABELS).rename("Payments"))


def main() -> None:
    try:
        # mode=rw prevents SQLite from silently creating an empty database.
        with closing(sqlite3.connect(DATABASE_PATH.resolve().as_uri() + "?mode=rw", uri=True)) as connection:
            create_feedback_table(connection)
            show_dashboard(connection, read_scores(DATABASE_PATH))
    except (sqlite3.Error, pd.errors.DatabaseError, ValueError) as exc:
        st.error(f"Could not load or save dashboard data: {exc}")


if __name__ == "__main__":
    main()
