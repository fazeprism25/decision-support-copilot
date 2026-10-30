import pandas as pd
import pytest

from engine.loader import list_sheets, load_table
from engine.profile import profile_frame
from engine.roles import detect_roles, entity_attributes
from tests.fixtures import make_delivery, to_xlsx_with_title


@pytest.fixture(scope="module")
def delivery():
    return make_delivery()


def test_xlsx_title_rows_and_sheets(delivery):
    data = to_xlsx_with_title(delivery)
    assert list_sheets(data, "f.xlsx") == ["Orders", "Notes"]
    df, info = load_table(data, "f.xlsx", "Orders")
    assert info.header_row == 2
    assert list(df.columns) == list(delivery.columns)
    assert len(df) == len(delivery)


def test_csv_semicolon_and_header_offset():
    text = "Export from system\n\nname;amount;when\na;1,5;2024-01-01\nb;2;2024-01-02\n"
    df, info = load_table(text.encode(), "x.csv")
    assert info.header_row == 2
    assert list(df.columns) == ["name", "amount", "when"]
    assert len(df) == 2


def test_duplicate_and_blank_headers_cleaned():
    df, _ = load_table(b"a,a,,b\n1,2,3,4\n", "x.csv")
    assert list(df.columns) == ["a", "a_2", "Column_3", "b"]


def test_rule_roles(delivery):
    roles, _ = detect_roles(delivery, "t", use_ai=False)
    r = dict(zip(roles["column"], roles["role"]))
    assert r["Order_ID"] == "id"
    assert r["Order_Date"] == "date"
    assert r["Hub"] == "location"
    assert r["Priority"] == "category"
    assert r["Peak_Hour"] == "flag"
    assert r["Delivery_Fee"] == "money"
    assert r["Actual_Time_min"] == "duration"
    assert r["Customer_Rating"] == "rating"
    assert r["Customer_Comment"] == "text"
    assert r["Extra_Metric_0"] == "measure"


def test_ai_label_rejected_when_stats_disagree(delivery, monkeypatch):
    fake = {c: {"role": "category"} for c in delivery.columns}
    fake["Delivery_Fee"] = {"role": "date", "meaning": "wrong on purpose"}
    monkeypatch.setattr("engine.roles.chat_json", lambda *a, **k: fake)
    roles, _ = detect_roles(delivery, "t2", use_ai=True)
    row = roles.set_index("column").loc["Delivery_Fee"]
    assert row["role"] == "money" and row["source"] == "rules (AI rejected)"


def test_entity_attributes(delivery):
    roles, _ = detect_roles(delivery, "t", use_ai=False)
    attrs = entity_attributes(delivery, roles).set_index("attribute")
    assert attrs.loc["Hub_Capacity", "entity"] == "Hub"
    assert "Delivery_Fee" not in attrs.index


def test_profile_handles_messy_numbers():
    p = profile_frame(pd.DataFrame({"x": ["₹1,200", "$3", "45%", None]}))["x"]
    assert p["numeric_rate"] == 1.0 and p["max"] == 1200
