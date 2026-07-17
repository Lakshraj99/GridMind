"""Centralized Streamlit visual system using only owned CSS classes."""

from __future__ import annotations

from typing import Any

THEME_CSS = """
<style>
:root {
  --gm-bg: #0b1220;
  --gm-surface: #111b2e;
  --gm-surface-2: #162238;
  --gm-border: #283753;
  --gm-text: #e6edf7;
  --gm-muted: #93a4bd;
  --gm-blue: #55a6ff;
  --gm-cyan: #46c2c8;
  --gm-green: #3ecf8e;
  --gm-amber: #f5b942;
  --gm-red: #f06a6a;
}
.stApp { background: var(--gm-bg); color: var(--gm-text); }
.gm-shell-title { font-size: 1.62rem; font-weight: 720; letter-spacing: -0.02em; margin: 0; }
.gm-shell-subtitle, .gm-muted { color: var(--gm-muted); }
.gm-page-head { display: flex; justify-content: space-between; gap: 1rem; align-items: flex-end;
  margin: .25rem 0 1.15rem; padding-bottom: .9rem; border-bottom: 1px solid var(--gm-border); }
.gm-page-title { font-size: 1.75rem; font-weight: 700; letter-spacing: -.025em; margin: 0; }
.gm-page-kicker { color: var(--gm-blue); font-size: .72rem; text-transform: uppercase;
  letter-spacing: .12em; font-weight: 700; margin-bottom: .25rem; }
.gm-page-description { color: var(--gm-muted); margin-top: .28rem; max-width: 54rem; }
.gm-refresh { color: var(--gm-muted); font-size: .76rem; white-space: nowrap; }
.gm-card { background: linear-gradient(180deg, var(--gm-surface-2), var(--gm-surface));
  border: 1px solid var(--gm-border); border-radius: 10px; padding: 1rem 1.05rem;
  min-height: 112px; box-shadow: 0 8px 24px rgba(0,0,0,.14); }
.gm-card-label { color: var(--gm-muted); font-size: .76rem; text-transform: uppercase;
  letter-spacing: .07em; font-weight: 650; }
.gm-card-value { color: var(--gm-text); font-size: 1.52rem; font-weight: 720;
  margin: .32rem 0 .1rem; }
.gm-card-detail { color: var(--gm-muted); font-size: .78rem; line-height: 1.35; }
.gm-section { margin: 1.5rem 0 .65rem; }
.gm-section-title { font-size: 1.02rem; font-weight: 680; margin: 0; }
.gm-section-caption { color: var(--gm-muted); font-size: .8rem; margin-top: .18rem; }
.gm-badge { display: inline-flex; align-items: center; border: 1px solid currentColor;
  border-radius: 999px; padding: .16rem .48rem; font-size: .69rem; line-height: 1.2;
  font-weight: 700; letter-spacing: .035em; text-transform: uppercase; }
.gm-neutral { color: #aebbd0; background: rgba(174,187,208,.08); }
.gm-info { color: #67b4ff; background: rgba(85,166,255,.1); }
.gm-success { color: #55d99e; background: rgba(62,207,142,.1); }
.gm-warning { color: #f5c65e; background: rgba(245,185,66,.1); }
.gm-critical { color: #ff8585; background: rgba(240,106,106,.1); }
.gm-state { border: 1px dashed var(--gm-border); border-radius: 10px; padding: 1.2rem;
  background: rgba(17,27,46,.62); }
.gm-state-title { font-weight: 680; margin-bottom: .25rem; }
.gm-state-copy { color: var(--gm-muted); font-size: .86rem; line-height: 1.45; }
.gm-strip { border: 1px solid var(--gm-border); border-left: 3px solid var(--gm-blue);
  border-radius: 8px; background: rgba(17,27,46,.7); padding: .72rem .85rem;
  color: var(--gm-muted); font-size: .8rem; line-height: 1.45; }
.gm-disclaimer { border-left-color: var(--gm-amber); }
.gm-lineage { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr));
  gap: .55rem; }
.gm-lineage-item { border: 1px solid var(--gm-border); border-radius: 8px; padding: .62rem .7rem;
  background: rgba(17,27,46,.62); min-width: 0; }
.gm-lineage-label { color: var(--gm-muted); font-size: .68rem; text-transform: uppercase;
  letter-spacing: .06em; }
.gm-lineage-value { font-size: .8rem; margin-top: .22rem; overflow-wrap: anywhere; }
.gm-footer { margin-top: 2.6rem; padding: 1rem 0 .3rem; border-top: 1px solid var(--gm-border);
  color: var(--gm-muted); font-size: .75rem; text-align: center; }
@media (max-width: 900px) {
  .gm-page-head { display: block; }
  .gm-refresh { margin-top: .5rem; }
  .gm-card { min-height: 98px; }
}
</style>
"""


def apply_theme(st: Any) -> None:
    """Install the dashboard's owned visual styles once per rerun."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)
