import pandas as pd

from riskprism.data.edgar import Fundamentals, store_from_frame, store_to_frame


def _fund(vals):
    return Fundamentals({
        "book_equity": pd.DataFrame({
            "end": pd.to_datetime(["2025-12-31", "2026-03-31"]),
            "filed": pd.to_datetime(["2026-02-15", "2026-05-10"]),
            "val": vals,
        }),
        "shares_out": pd.DataFrame(columns=["end", "filed", "val"]),
    })


def test_store_roundtrip_preserves_pit_lookups():
    store = {"AAPL": _fund([100.0, 110.0]), "MSFT": _fund([200.0, 220.0])}
    frame = store_to_frame(store)
    back = store_from_frame(frame)
    as_of = pd.Timestamp("2026-03-01")  # only the first filing is known
    assert back["AAPL"].asof(as_of)["book_equity"] == 100.0
    assert back["MSFT"].asof(pd.Timestamp("2026-06-01"))["book_equity"] == 220.0
    # missing fields come back as NaN, same as before serialization
    assert pd.isna(back["AAPL"].asof(as_of)["shares_out"])


def test_empty_store_serializes():
    frame = store_to_frame({"X": Fundamentals({})})
    assert frame.empty
    assert store_from_frame(frame) == {}
