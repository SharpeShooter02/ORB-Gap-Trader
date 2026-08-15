"""
tests/test_refresh_sigma_master.py — the sigma refresher is self-contained in
the live repo (yfinance + built-in alias map) and merges without dropping rows.

No network: the yfinance fetch is monkeypatched.
"""

import numpy as np
import pandas as pd

from orb_live.scripts import refresh_sigma_master as rsm
from orb_live.config.live_config import _load_v1_universe


def test_crypto_underlyings_have_yf_aliases():
    """Every non-equity underlying in the live universe must map to a yfinance
    ticker, or a refresh would silently skip it."""
    instruments, _s, universe = _load_v1_universe()
    uls = {instruments[s].underlying for s in universe if s in instruments}
    for crypto in ("BTC", "ETH", "SOL", "XRP"):
        if crypto in uls:
            assert crypto in rsm._YF_ALIASES, f"{crypto} needs a yfinance alias"
            assert rsm._YF_ALIASES[crypto].endswith("-USD")


def test_refresh_merges_and_preserves_untouched_rows(tmp_path, monkeypatch):
    """Recomputed underlyings are updated; every other CSV row is preserved."""
    csv = tmp_path / "sigma_master.csv"
    pd.DataFrame({
        "underlying": ["AAA", "BBB", "CCC"],
        "sigma": [0.010, 0.020, 0.030],
        "n_obs": [100, 200, 300],
        "source": ["cache", "cache", "cache"],
        "computed_at": ["old", "old", "old"],
    }).to_csv(csv, index=False)

    # Deterministic synthetic history only for AAA.
    def _fake_fetch(ul):
        if ul != "AAA":
            return None
        rng = np.random.default_rng(0)
        return pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.02, 500)))
    monkeypatch.setattr(rsm, "_fetch_adjusted_closes", _fake_fetch)

    rsm.refresh(targets=["AAA"], confirm=True, out_path=csv)

    out = pd.read_csv(csv)
    m = dict(zip(out["underlying"], out["sigma"]))
    assert set(out["underlying"]) == {"AAA", "BBB", "CCC"}     # nothing dropped
    assert m["AAA"] != 0.010                                    # AAA recomputed
    assert m["BBB"] == 0.020 and m["CCC"] == 0.030             # others preserved
    src = dict(zip(out["underlying"], out["source"]))
    assert src["AAA"] == "yfinance" and src["BBB"] == "cache"


def test_refresh_preview_does_not_write(tmp_path, monkeypatch):
    """Without --confirm the CSV must be left untouched."""
    csv = tmp_path / "sigma_master.csv"
    pd.DataFrame({
        "underlying": ["AAA"], "sigma": [0.010], "n_obs": [100],
        "source": ["cache"], "computed_at": ["old"],
    }).to_csv(csv, index=False)
    before = csv.read_text()

    monkeypatch.setattr(rsm, "_fetch_adjusted_closes",
                        lambda ul: pd.Series(np.linspace(100, 110, 400)))
    rsm.refresh(targets=["AAA"], confirm=False, out_path=csv)

    assert csv.read_text() == before, "preview must not modify the CSV"
