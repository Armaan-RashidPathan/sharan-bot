"""
Deterministic financial calculators built on Sharan Hegde's own frameworks —
real Python math, not an LLM guessing at arithmetic. The RAG chain (chain/)
handles explaining *why*; this module handles computing *how much*, which is
exactly the kind of thing an LLM is unreliable at and code isn't.

Every number below is grounded in a specific, verified transcript moment,
not a generic finance-industry rule:

- Lean FIRE = annual expenses x 20 (a 5% withdrawal rate) — stated directly
  in "How To Invest For Early Retirement" (GjfjqfqDzCg @114s): "lean fire
  simply put is your annual expenses multiplied by 20."

- 60% domestic equity / 10% US equity / 15% debt / 5% gold / 5% crypto /
  5% real estate — Sharan's stated asset-allocation split for reaching a
  FIRE number, same video @616s-718s, verbatim.

- 12% expected annual equity return is offered as the default (not a fixed
  rule) — it's the return figure referenced most consistently across the
  corpus (e.g. IgjhqPgwwGI @11793s, @11916s: "12% return"), not something
  Sharan states as a hard guarantee, so it's left adjustable.
"""

LEAN_FIRE_MULTIPLE = 20  # annual expenses x 20 == a 5% withdrawal rate

ASSET_ALLOCATION = {
    "Domestic equity": 0.60,
    "US equity": 0.10,
    "Debt": 0.15,
    "Gold": 0.05,
    "Crypto": 0.05,
    "Real estate": 0.05,
}

DEFAULT_ANNUAL_RETURN = 0.12


def lean_fire_number(annual_expenses: float) -> float:
    """The corpus needed to cover annual_expenses indefinitely at a 5% withdrawal rate."""
    return annual_expenses * LEAN_FIRE_MULTIPLE


def asset_allocation_split(total_amount: float) -> dict[str, float]:
    """Split total_amount across Sharan's 60/10/15/5/5/5 asset-class framework."""
    return {asset_class: total_amount * weight for asset_class, weight in ASSET_ALLOCATION.items()}


def sip_future_value(monthly_investment: float, annual_return: float, years: float) -> float:
    """Future value of a monthly SIP compounding at annual_return over years.

    Standard SIP future-value formula, with each month's contribution
    compounding for the remainder of the period (i.e. the last month's
    contribution earns interest too, not just prior months' — the "annuity
    due" convention most SIP calculators use, matching Sharan's own
    Excel-tool figures rather than the plain "ordinary annuity" formula).
    """
    if annual_return == 0:
        return monthly_investment * years * 12

    monthly_rate = annual_return / 12
    n_months = years * 12
    return monthly_investment * (((1 + monthly_rate) ** n_months - 1) / monthly_rate) * (1 + monthly_rate)


def required_monthly_sip(target_amount: float, annual_return: float, years: float) -> float:
    """The monthly SIP needed to reach target_amount in years, at annual_return — the inverse of sip_future_value."""
    if annual_return == 0:
        return target_amount / (years * 12)

    monthly_rate = annual_return / 12
    n_months = years * 12
    return target_amount / ((((1 + monthly_rate) ** n_months - 1) / monthly_rate) * (1 + monthly_rate))
