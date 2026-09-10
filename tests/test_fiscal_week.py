"""Ground-truth tests for the fiscal-week rule used by
`scripts/measure_finetune_fiscal.py`.

The whole point of that measurement is that the ground truth is computable, so the
measurement is only as trustworthy as this file. Every expectation below was worked out by
hand from a calendar, not by running the function and pasting its answer.
"""

import datetime as dt
import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "measure_finetune_fiscal.py"
_spec = importlib.util.spec_from_file_location("measure_finetune_fiscal", _SCRIPT)
mff = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(mff)


# The fiscal-year epochs, checked against a calendar. 1 Feb 2024 is a Thursday, so the
# first Monday is the 5th; 1 Feb 2025 is a Saturday -> the 3rd; 1 Feb 2026 is a Sunday ->
# the 2nd; 1 Feb 2027 is itself a Monday -> the 1st; 1 Feb 2028 is a Tuesday -> the 7th.
FIRST_MONDAYS = {
    2023: dt.date(2023, 2, 6),
    2024: dt.date(2024, 2, 5),
    2025: dt.date(2025, 2, 3),
    2026: dt.date(2026, 2, 2),
    2027: dt.date(2027, 2, 1),
    2028: dt.date(2028, 2, 7),
}


@pytest.mark.parametrize("year,expected", sorted(FIRST_MONDAYS.items()))
def test_first_monday_of_february(year, expected):
    got = mff.first_monday_of_february(year)
    assert got == expected
    assert got.weekday() == 0
    assert got.month == 2
    assert 1 <= got.day <= 7


def test_epoch_is_week_one():
    for year, monday in FIRST_MONDAYS.items():
        assert mff.fiscal_label(monday) == f"FY{year}-W01"


def test_worked_examples_from_the_spec():
    # These two are quoted verbatim inside SPEC; if they are wrong, every arm was told a
    # lie and the measurement is void.
    assert mff.fiscal_label(dt.date(2026, 3, 3)) == "FY2026-W05"
    assert mff.fiscal_label(dt.date(2026, 1, 20)) == "FY2025-W51"
    assert "FY2026-W05" in mff.SPEC
    assert "FY2025-W51" in mff.SPEC


@pytest.mark.parametrize("year", [2024, 2025, 2026, 2027])
def test_boundary_days_around_each_first_monday(year):
    """The three days that decide the rule: the Sunday before, the Monday, the Tuesday."""
    monday = FIRST_MONDAYS[year]
    sunday = monday - dt.timedelta(days=1)
    tuesday = monday + dt.timedelta(days=1)

    prev_weeks = mff.fiscal_year_weeks(year - 1)
    assert mff.fiscal_label(sunday) == f"FY{year - 1}-W{prev_weeks:02d}"
    assert mff.fiscal_label(monday) == f"FY{year}-W01"
    assert mff.fiscal_label(tuesday) == f"FY{year}-W01"

    # The whole first week is W01, and the eighth day is W02.
    for offset in range(7):
        assert mff.fiscal_label(monday + dt.timedelta(days=offset)) == f"FY{year}-W01"
    assert mff.fiscal_label(monday + dt.timedelta(days=7)) == f"FY{year}-W02"


def test_hand_checked_dates():
    cases = {
        # 2024-02-05 is the FY2024 epoch. 2024-03-04 is 28 days later = 4 whole weeks -> W05.
        dt.date(2024, 3, 4): "FY2024-W05",
        # 2024-12-30 is 329 days after 2024-02-05; 329/7 = 47 exactly -> W48.
        dt.date(2024, 12, 30): "FY2024-W48",
        # 2025-01-01 is in January -> FY2024. 331 days after the epoch; 331//7 = 47 -> W48.
        dt.date(2025, 1, 1): "FY2024-W48",
        # The last day of FY2024 is the Sunday before 2025-02-03, i.e. 2025-02-02, W52.
        dt.date(2025, 2, 2): "FY2024-W52",
        # 2026-02-02 epoch + 364 days = 2027-01-31, the last day of FY2026 -> W52.
        dt.date(2027, 1, 31): "FY2026-W52",
        # 2027-02-01 epoch, so 2027-12-31 is 333 days later; 333//7 = 47 -> W48.
        dt.date(2027, 12, 31): "FY2027-W48",
        # Mid-January of a year whose fiscal predecessor is a normal 52-week year.
        dt.date(2026, 1, 20): "FY2025-W51",
    }
    for date, expected in cases.items():
        assert mff.fiscal_label(date) == expected, date


def test_fifty_three_week_year_exists_and_is_fy2027():
    """FY2027 runs 2027-02-01 -> 2028-02-06, which is 371 days = 53 weeks.

    Its W53 begins at epoch + 52*7 = 364 days, i.e. 2028-01-31, and runs to 2028-02-06 --
    entirely outside the 2024-01-01..2027-12-31 evaluation window, so no evaluation case
    can be labelled W53. The highest week number reachable in the evaluation set is W52.
    """
    assert mff.fiscal_year_weeks(2027) == 53
    for year in (2023, 2024, 2025, 2026, 2028):
        assert mff.fiscal_year_weeks(year) == 52

    assert mff.fiscal_label(dt.date(2028, 1, 30)) == "FY2027-W52"
    assert mff.fiscal_label(dt.date(2028, 1, 31)) == "FY2027-W53"
    assert mff.fiscal_label(dt.date(2028, 2, 6)) == "FY2027-W53"
    assert mff.fiscal_label(dt.date(2028, 2, 7)) == "FY2028-W01"

    labels = {mff.fiscal_label(mff.RANGE_START + dt.timedelta(days=i))
              for i in range((mff.RANGE_END - mff.RANGE_START).days + 1)}
    assert not any(l.endswith("W53") for l in labels)


def test_every_date_in_range_is_covered_and_weeks_are_contiguous():
    d = mff.RANGE_START
    seen = {}
    while d <= mff.RANGE_END:
        label = mff.fiscal_label(d)
        assert mff.STRICT_LABEL_RE.match(label), (d, label)
        fy, week = int(label[2:6]), int(label[8:10])
        assert 1 <= week <= mff.fiscal_year_weeks(fy)
        seen.setdefault(fy, set()).add(week)
        d += dt.timedelta(days=1)
    # FY2024, FY2025 and FY2026 are wholly inside the range, so each must show all 52.
    for fy in (2024, 2025, 2026):
        assert seen[fy] == set(range(1, 53)), fy


def test_render_date_round_trips_all_four_formats():
    date = dt.date(2026, 3, 3)
    assert mff.render_date(date, "iso") == "2026-03-03"
    assert mff.render_date(date, "long") == "March 3, 2026"
    assert mff.render_date(date, "day_first") == "3 Mar 2026"
    assert mff.render_date(date, "weekday_prefixed") == "Tuesday 3 March 2026"
    for fmt in mff.FORMATS:
        # No numeric-only ambiguous form anywhere.
        assert not __import__("re").fullmatch(r"[\d/.\-]+", mff.render_date(date, fmt)) or fmt == "iso"


def test_is_boundary_window():
    monday = FIRST_MONDAYS[2026]
    assert mff.is_boundary(monday)
    assert mff.is_boundary(monday + dt.timedelta(days=10))
    assert mff.is_boundary(monday - dt.timedelta(days=10))
    assert not mff.is_boundary(monday + dt.timedelta(days=11))
    assert not mff.is_boundary(monday - dt.timedelta(days=11))


def test_fixture_is_deterministic_and_disjoint(tmp_path):
    a = mff.build_fixture(tmp_path / "a.json")
    b = mff.build_fixture(tmp_path / "b.json")
    assert [c["input"] for c in a["evaluation"]] == [c["input"] for c in b["evaluation"]]

    assert a["n_evaluation"] == mff.N_EVAL
    assert a["n_folding"] == mff.N_FOLD

    eval_dates = {c["date"] for c in a["evaluation"]}
    fold_dates = {c["date"] for c in a["folding_examples"]}
    assert len(eval_dates) == mff.N_EVAL          # no repeated dates
    assert not (eval_dates & fold_dates)          # disjoint by construction

    # Every label in the fixture agrees with the ground-truth function.
    for c in a["evaluation"] + a["folding_examples"]:
        assert c["expected"] == mff.fiscal_label(dt.date.fromisoformat(c["date"]))

    # All four formats present in both sets; the folding pool covers each twice.
    assert {c["format"] for c in a["folding_examples"]} == set(mff.FORMATS)
    assert sum(1 for c in a["folding_examples"] if c["boundary"]) >= 1
    from collections import Counter
    counts = Counter(c["format"] for c in a["evaluation"])
    assert set(counts) == set(mff.FORMATS)
    assert max(counts.values()) - min(counts.values()) <= 1

    # The +/-10 windows around all four in-range fiscal starts are fully represented.
    assert a["n_boundary"] >= 80
    for c in a["evaluation"]:
        assert c["boundary"] == mff.is_boundary(dt.date.fromisoformat(c["date"]))
