-- anomaly_flags.sql
-- ------------------
-- Rule-based approximation of the Isolation Forest flags, useful for
-- explaining *why* something got flagged in plain SQL (interviewers often
-- ask "could you do this without ML?" -- this is that answer).
--
-- Assumes a table `transactions` loaded from the Kaggle CSV with the same
-- column names: TransactionID, AccountID, TransactionAmount, TransactionDate,
-- TransactionType, Location, DeviceID, MerchantID, Channel, CustomerAge,
-- CustomerOccupation, TransactionDuration, LoginAttempts, AccountBalance,
-- PreviousTransactionDate.

WITH device_freq AS (
    SELECT DeviceID, COUNT(*) AS device_count
    FROM transactions
    GROUP BY DeviceID
),
enriched AS (
    SELECT
        t.*,
        EXTRACT(HOUR FROM t.TransactionDate)                                   AS hour_of_day,
        (JULIANDAY(t.TransactionDate) - JULIANDAY(t.PreviousTransactionDate))
            * 24                                                               AS hours_since_prev_tx,
        t.TransactionAmount * 1.0 / NULLIF(t.AccountBalance, 0)                AS amount_to_balance_ratio,
        d.device_count
    FROM transactions t
    LEFT JOIN device_freq d ON t.DeviceID = d.DeviceID
),
flagged AS (
    SELECT
        *,
        CASE WHEN LoginAttempts >= 6 THEN 1 ELSE 0 END              AS flag_login_attempts,
        CASE WHEN TransactionAmount >= 3500 THEN 1 ELSE 0 END       AS flag_large_amount,
        CASE WHEN hour_of_day BETWEEN 1 AND 4 THEN 1 ELSE 0 END     AS flag_odd_hour,
        CASE WHEN device_count <= 1 THEN 1 ELSE 0 END               AS flag_rare_device,
        CASE WHEN amount_to_balance_ratio >= 0.8 THEN 1 ELSE 0 END  AS flag_near_full_balance
    FROM enriched
)
SELECT
    TransactionID,
    AccountID,
    TransactionAmount,
    TransactionDate,
    Channel,
    Location,
    LoginAttempts,
    hour_of_day,
    ROUND(amount_to_balance_ratio, 2) AS amount_to_balance_ratio,
    device_count,
    (flag_login_attempts + flag_large_amount + flag_odd_hour
        + flag_rare_device + flag_near_full_balance) AS rule_score,
    flag_login_attempts, flag_large_amount, flag_odd_hour,
    flag_rare_device, flag_near_full_balance
FROM flagged
WHERE (flag_login_attempts + flag_large_amount + flag_odd_hour
        + flag_rare_device + flag_near_full_balance) >= 1
ORDER BY rule_score DESC, TransactionAmount DESC;

-- Summary: % flagged by channel
SELECT
    Channel,
    COUNT(*) AS total_tx,
    SUM(CASE WHEN (flag_login_attempts + flag_large_amount + flag_odd_hour
            + flag_rare_device + flag_near_full_balance) >= 1 THEN 1 ELSE 0 END) AS flagged_tx,
    ROUND(100.0 * SUM(CASE WHEN (flag_login_attempts + flag_large_amount + flag_odd_hour
            + flag_rare_device + flag_near_full_balance) >= 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS flagged_pct
FROM flagged
GROUP BY Channel
ORDER BY flagged_pct DESC;
