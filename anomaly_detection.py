"""
anomaly_detection.py
---------------------
Unsupervised anomaly detection on bank transaction data using Isolation
Forest — mirrors how an Operations/risk-monitoring team flags unusual
transaction patterns for review when there's no pre-existing fraud label
to train against.

Input : transactions.csv  (real Kaggle file, or transactions_demo.csv for testing)
Output: anomaly_summary.csv          -- headline stats for the dashboard
        anomaly_flagged_transactions.csv -- full detail on flagged rows
        anomaly_by_channel.csv       -- breakdown by channel
        anomaly_by_location.csv      -- breakdown by location

Usage:
    python anomaly_detection.py transactions.csv
    python anomaly_detection.py transactions_demo.csv   # for testing only
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

CONTAMINATION = 0.05  # assume ~5% of transactions are worth a second look


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
    df["PreviousTransactionDate"] = pd.to_datetime(df["PreviousTransactionDate"])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw columns into signals a risk analyst would actually look at."""
    df = df.copy()

    # How long since this account's last transaction (fast repeats look odd)
    df["HoursSincePrevTx"] = (
        (df["TransactionDate"] - df["PreviousTransactionDate"])
        .dt.total_seconds() / 3600
    ).clip(lower=0)

    # Hour of day — late-night activity is a classic risk signal
    df["HourOfDay"] = df["TransactionDate"].dt.hour

    # How large this transaction is relative to the account's own balance
    df["AmountToBalanceRatio"] = df["TransactionAmount"] / df["AccountBalance"].replace(0, np.nan)
    df["AmountToBalanceRatio"] = df["AmountToBalanceRatio"].fillna(df["AmountToBalanceRatio"].median())

    # How many times has this specific device been seen? (new/rare device = higher risk)
    device_counts = df["DeviceID"].value_counts()
    df["DeviceFrequency"] = df["DeviceID"].map(device_counts)

    return df


FEATURES = [
    "TransactionAmount",
    "TransactionDuration",
    "LoginAttempts",
    "HoursSincePrevTx",
    "HourOfDay",
    "AmountToBalanceRatio",
    "DeviceFrequency",
]


def run_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    X = df[FEATURES].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=42,
    )
    df["AnomalyFlag"] = model.fit_predict(X_scaled)  # -1 = anomaly, 1 = normal
    df["AnomalyScore"] = -model.score_samples(X_scaled)  # higher = more unusual
    df["IsAnomalous"] = df["AnomalyFlag"] == -1

    return df


def classify_reason(row) -> str:
    """Give each flagged transaction a plain-English reason, the way an
    analyst would annotate a case before escalating it."""
    reasons = []
    if row["LoginAttempts"] >= 6:
        reasons.append("high login attempts")
    if row["TransactionAmount"] >= 3500:
        reasons.append("unusually large amount")
    if row["HourOfDay"] in (1, 2, 3, 4):
        reasons.append("odd-hour activity")
    if row["DeviceFrequency"] <= 1:
        reasons.append("rarely-seen device")
    if row["AmountToBalanceRatio"] >= 0.8:
        reasons.append("amount close to full balance")
    return "; ".join(reasons) if reasons else "no single dominant factor (multivariate)"


def summarize(df: pd.DataFrame) -> None:
    flagged = df[df["IsAnomalous"]].copy()
    flagged["Reason"] = flagged.apply(classify_reason, axis=1)

    total = len(df)
    n_flagged = len(flagged)
    pct_flagged = round(100 * n_flagged / total, 2)

    print("=" * 60)
    print("ANOMALY DETECTION SUMMARY")
    print("=" * 60)
    print(f"Total transactions analyzed : {total}")
    print(f"Flagged as anomalous        : {n_flagged} ({pct_flagged}%)")
    print(f"Contamination parameter used: {CONTAMINATION}")
    print()
    print("Top contributing factors among flagged transactions:")
    reason_counts = (
        flagged["Reason"].str.split("; ").explode().value_counts()
    )
    print(reason_counts.to_string())
    print()

    # --- Save dashboard-ready CSVs ---
    summary_df = pd.DataFrame({
        "Metric": ["Total Transactions", "Flagged Anomalous", "Flagged %", "Clean %"],
        "Value": [total, n_flagged, pct_flagged, round(100 - pct_flagged, 2)],
    })
    summary_df.to_csv("anomaly_summary.csv", index=False)

    flagged_out = flagged[[
        "TransactionID", "AccountID", "TransactionAmount", "TransactionDate",
        "Channel", "Location", "LoginAttempts", "HourOfDay",
        "AmountToBalanceRatio", "DeviceFrequency", "AnomalyScore", "Reason",
    ]].sort_values("AnomalyScore", ascending=False)
    flagged_out.to_csv("anomaly_flagged_transactions.csv", index=False)

    by_channel = (
        df.groupby("Channel")["IsAnomalous"]
        .agg(TotalTx="count", FlaggedTx="sum")
        .reset_index()
    )
    by_channel["FlaggedPct"] = round(100 * by_channel["FlaggedTx"] / by_channel["TotalTx"], 2)
    by_channel.to_csv("anomaly_by_channel.csv", index=False)

    by_location = (
        df.groupby("Location")["IsAnomalous"]
        .agg(TotalTx="count", FlaggedTx="sum")
        .reset_index()
    )
    by_location["FlaggedPct"] = round(100 * by_location["FlaggedTx"] / by_location["TotalTx"], 2)
    by_location = by_location.sort_values("FlaggedPct", ascending=False)
    by_location.to_csv("anomaly_by_location.csv", index=False)

    print("Saved: anomaly_summary.csv, anomaly_flagged_transactions.csv, "
          "anomaly_by_channel.csv, anomaly_by_location.csv")


def main():
    if len(sys.argv) < 2:
        print("Usage: python anomaly_detection.py <transactions.csv>")
        sys.exit(1)

    path = sys.argv[1]
    df = load_data(path)
    df = engineer_features(df)
    df = run_isolation_forest(df)
    summarize(df)


if __name__ == "__main__":
    main()
