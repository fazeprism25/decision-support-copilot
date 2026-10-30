"""Generate two synthetic datasets unrelated to delivery, used to prove the app is not BUDS-specific.
Each has one messy column and some duplicate rows. Run: .venv\\Scripts\\python scripts\\make_synthetic.py"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data"


def retail(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 1500
    stores = ["Indiranagar", "Koramangala", "Whitefield", "Jayanagar", "HSR Layout"]
    products = {"Rice 5kg": 320, "Atta 10kg": 450, "Sunflower Oil 1L": 160, "Tea 500g": 240, "Soap 4-pack": 140,
                "Detergent 2kg": 380, "Biscuits": 40, "Shampoo": 210, "Toothpaste": 95, "Coffee 200g": 290}
    prod = rng.choice(list(products), n)
    price = np.array([products[p] for p in prod]) * rng.uniform(0.9, 1.1, n)
    units = rng.integers(1, 12, n)
    store = rng.choice(stores, n, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    df = pd.DataFrame({
        "date": (pd.Timestamp("2025-03-01") + pd.to_timedelta(rng.integers(0, 60, n), "D")).strftime("%d/%m/%Y"),
        "store": store,
        "product": prod,
        "units": units,
        "price": price.round(2),
        "cost": (price * rng.uniform(0.62, 0.8, n)).round(2),
        "returns": rng.binomial(units, np.where(store == "Whitefield", 0.12, 0.03)),
    })
    # messy column: price as text with currency symbols, thousands separators and some junk
    p = df["price"].map(lambda v: f"₹{v:,.2f}")
    junk = rng.choice(n, 25, replace=False)
    p.iloc[junk[:15]] = "n/a"
    p.iloc[junk[15:]] = df["price"].iloc[junk[15:]].map(lambda v: f"Rs. {v:.0f}")
    df["price"] = p
    dup = df.sample(9, random_state=1)
    return pd.concat([df, dup], ignore_index=True).sample(frac=1, random_state=2).reset_index(drop=True)


def hr(seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 600
    depts = ["Engineering", "Sales", "Operations", "Finance", "HR"]
    dept = rng.choice(depts, n, p=[0.35, 0.25, 0.2, 0.12, 0.08])
    base = {"Engineering": 95000, "Sales": 62000, "Operations": 54000, "Finance": 78000, "HR": 58000}
    salary = np.array([base[d] for d in dept]) * rng.lognormal(0, 0.18, n)
    rating = np.clip(np.round(rng.normal(3.4, 0.8, n)), 1, 5).astype(int)
    p_leave = 0.08 + 0.12 * (dept == "Sales") + 0.08 * (rating <= 2)
    df = pd.DataFrame({
        "employee": [f"E{10000 + i}" for i in range(n)],
        "dept": dept,
        "hire date": (pd.Timestamp("2016-01-01") + pd.to_timedelta(rng.integers(0, 3200, n), "D")).strftime("%Y-%m-%d"),
        "salary": salary.round(0),
        "rating": rating,
        "attrition flag": np.where(rng.random(n) < p_leave, "Yes", "No"),
    })
    # messy column: dept with casing/spacing variants and a few blanks
    idx = rng.choice(n, 40, replace=False)
    df.loc[idx[:15], "dept"] = df.loc[idx[:15], "dept"].str.lower()
    df.loc[idx[15:30], "dept"] = df.loc[idx[15:30], "dept"].str.upper() + " "
    df.loc[idx[30:], "dept"] = None
    dup = df.sample(6, random_state=3)
    return pd.concat([df, dup], ignore_index=True)


if __name__ == "__main__":
    retail().to_csv(OUT / "retail_sales.csv", index=False, encoding="utf-8")
    hr().to_csv(OUT / "hr_employees.csv", index=False, encoding="utf-8")
    print("wrote data/retail_sales.csv and data/hr_employees.csv")
