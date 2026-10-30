"""Synthetic test data. Column names here are test inputs only; the engine must not depend on them."""
import io

import numpy as np
import pandas as pd


def make_delivery(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hubs = [f"H{i}" for i in range(1, 7)]
    cap = {h: c for h, c in zip(hubs, [120, 90, 150, 80, 100, 110])}
    zone_of = {h: z for h, z in zip(hubs, ["North", "North", "South", "East", "West", "South"])}
    hub = rng.choice(hubs, n)
    promised = rng.choice([30, 45, 60], n)
    df = pd.DataFrame({
        "Order_ID": [f"ORD{i:06d}" for i in range(n)],
        "Order_Date": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 30, n), "D"),
        "Hub": hub,
        "Zone": [zone_of[h] for h in hub],
        "Hub_Capacity": [cap[h] for h in hub],
        "Priority": rng.choice(["High", "Medium", "Low"], n),
        "Peak_Hour": rng.choice(["Yes", "No"], n),
        "Distance_km": rng.uniform(0.5, 15, n).round(2),
        "Promised_Time_min": promised,
        "Actual_Time_min": (promised + rng.normal(0, 12, n)).round(0),
        "Delivery_Fee": rng.uniform(20, 80, n).round(2),
        "Delivery_Cost": rng.uniform(15, 70, n).round(2),
        "Customer_Rating": rng.integers(1, 6, n),
        "Order_Status": rng.choice(["Delivered", "Cancelled"], n, p=[0.92, 0.08]),
        "Items": rng.integers(1, 10, n),
        "Customer_Comment": rng.choice(["Great service and fast delivery, thank you", "Late again, food was cold",
                                        None], n),
    })
    for i in range(24):  # pad to 40 columns
        df[f"Extra_Metric_{i}"] = rng.normal(100, 10, n).round(3)
    dup = df.sample(12, random_state=1).copy()
    df = pd.concat([df, dup], ignore_index=True)
    df.loc[df.sample(15, random_state=2).index, "Zone"] = None
    df.loc[df.sample(10, random_state=3).index, "Hub"] = None
    df.loc[5, "Distance_km"] = -1
    df.loc[df.sample(4, random_state=4).index, "Customer_Rating"] = 6
    df.loc[df.sample(3, random_state=5).index, "Actual_Time_min"] = 0
    return df


def to_xlsx_with_title(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame([["Quarterly delivery report"], [None]]).to_excel(w, sheet_name="Orders", header=False, index=False)
        df.to_excel(w, sheet_name="Orders", startrow=2, index=False)
        pd.DataFrame({"note": ["readme"]}).to_excel(w, sheet_name="Notes", index=False)
    return buf.getvalue()
