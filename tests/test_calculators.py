"""Tests for calculators/finance.py — pure deterministic math, no LLM or
network involved, so these check exact values."""

import pytest

from calculators.finance import (
    ASSET_ALLOCATION,
    asset_allocation_split,
    lean_fire_number,
    required_monthly_sip,
    sip_future_value,
)


def test_lean_fire_number_is_20x_annual_expenses():
    assert lean_fire_number(600_000) == 12_000_000  # 6 lakh/yr -> 1.2 cr, Sharan's own worked example


def test_asset_allocation_split_sums_to_total():
    total = 1_000_000
    split = asset_allocation_split(total)
    assert sum(split.values()) == pytest.approx(total)


def test_asset_allocation_split_matches_percentages():
    split = asset_allocation_split(100_000)
    assert split["Domestic equity"] == pytest.approx(60_000)
    assert split["US equity"] == pytest.approx(10_000)
    assert split["Debt"] == pytest.approx(15_000)
    assert split["Gold"] == pytest.approx(5_000)
    assert split["Crypto"] == pytest.approx(5_000)
    assert split["Real estate"] == pytest.approx(5_000)


def test_asset_allocation_weights_sum_to_one():
    assert sum(ASSET_ALLOCATION.values()) == pytest.approx(1.0)


def test_sip_future_value_zero_return_is_just_sum_of_contributions():
    assert sip_future_value(10_000, annual_return=0, years=5) == 10_000 * 5 * 12


def test_sip_future_value_grows_with_positive_return():
    zero_return = sip_future_value(10_000, annual_return=0, years=10)
    with_return = sip_future_value(10_000, annual_return=0.12, years=10)
    assert with_return > zero_return


def test_required_monthly_sip_inverts_future_value():
    """Round-trip: the SIP required to reach a target, fed back through
    sip_future_value, should reproduce that same target."""
    target = 5_000_000
    annual_return = 0.12
    years = 15

    monthly = required_monthly_sip(target, annual_return, years)
    result = sip_future_value(monthly, annual_return, years)

    assert result == pytest.approx(target, rel=1e-9)


def test_required_monthly_sip_zero_return():
    assert required_monthly_sip(1_200_000, annual_return=0, years=10) == pytest.approx(10_000)
