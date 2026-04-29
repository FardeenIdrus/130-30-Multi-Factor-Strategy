"""Reusable Streamlit UI components.

Build pages out of these instead of raw Streamlit primitives so styling
stays consistent across the dashboard.
"""

from __future__ import annotations

import streamlit as st

from .theme import COLORS, CUSTOM_CSS


def page_setup(title: str, icon: str = "📊") -> None:
    """Standard page header. Call once at the top of every page."""
    st.set_page_config(
        page_title=f"{title} | Team Wittgenstein",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def kpi_card(
    label: str,
    value: str,
    delta: str | None = None,
    delta_positive: bool | None = None,
    sub: str | None = None,
) -> None:
    """A polished metric card. Use inside st.columns for grids."""
    parts = [
        '<div class="kpi-card">',
        f'<div class="kpi-label">{label}</div>',
        f'<div class="kpi-value">{value}</div>',
    ]
    if delta is not None:
        if delta_positive is None:
            cls = "kpi-sub"
        else:
            cls = "kpi-delta-positive" if delta_positive else "kpi-delta-negative"
        parts.append(f'<div class="{cls}">{delta}</div>')
    if sub:
        parts.append(f'<div class="kpi-sub">{sub}</div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def section_header(title: str, subtitle: str | None = None) -> None:
    """Consistent section header above each block of content."""
    parts = [f'<div class="section-header">{title}</div>']
    if subtitle:
        parts.append(
            f'<div style="color:{COLORS["text_muted"]};font-size:0.85rem;'
            f'margin-bottom:0.5rem;">{subtitle}</div>'
        )
    st.markdown("".join(parts), unsafe_allow_html=True)


def badge(text: str, kind: str = "info") -> str:
    """Inline pill badge. Returns HTML string - caller wraps in st.markdown."""
    return f'<span class="badge badge-{kind}">{text}</span>'


def status_pill(passed: bool, true_text: str = "OK", false_text: str = "FAIL") -> str:
    """Green/red pill for boolean status checks."""
    if passed:
        return badge(f"✓ {true_text}", "success")
    return badge(f"✗ {false_text}", "danger")


def db_status_badge(connected: bool) -> str:
    """Top-of-page DB connection indicator."""
    if connected:
        return badge("● Database connected", "success")
    return badge("● Database offline", "danger")


def info_panel(title: str, body: str) -> None:
    """Soft-coloured info panel (e.g. caveats, notes)."""
    html = (
        f'<div style="background:{COLORS["surface_alt"]};'
        f"border-left:3px solid {COLORS['primary']};"
        f'padding:0.75rem 1rem;border-radius:4px;margin:1rem 0;">'
        f'<div style="font-weight:600;color:{COLORS["text"]};'
        f'margin-bottom:0.25rem;">{title}</div>'
        f'<div style="color:{COLORS["text_muted"]};font-size:0.9rem;">'
        f"{body}</div></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def hero_header(title: str, subtitle: str = "") -> None:
    """Big page header for landing pages (Home)."""
    html = (
        f'<div style="margin-bottom:1.5rem;">'
        f'<div style="font-size:2rem;font-weight:700;color:{COLORS["text"]};'
        f'line-height:1.2;">{title}</div>'
        f'<div style="color:{COLORS["text_muted"]};font-size:1rem;'
        f'margin-top:0.25rem;">{subtitle}</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)
