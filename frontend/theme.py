"""Shared Streamlit theme tokens for InvenioAI.

Single source of truth for the UI color palette used by Streamlit pages.
"""

from __future__ import annotations

COLORS: dict[str, str] = {
    # Base backgrounds
    "bg_primary": "#0F1117",
    "bg_secondary": "#1A1D27",
    "bg_card": "#1E2130",
    "bg_sidebar": "#13151F",

    # Accents
    "accent": "#6C63FF",
    "accent_hover": "#8B85FF",
    "accent_light": "#6C63FF22",

    # Chat bubbles (some pages reference these explicitly)
    "user_bubble": "#6C63FF",
    "bot_bubble": "#1E2130",

    # Borders / text
    "border": "#2A2D3E",
    "text_primary": "#E8EAF6",
    "text_secondary": "#8F95B2",
    "text_muted": "#555876",

    # Status
    "success": "#00C9A7",
    "error": "#FF6B6B",
    "warning": "#FFB347",

    # Dashboard chart palette
    "chart1": "#6C63FF",
    "chart2": "#00C9A7",
    "chart3": "#FFB347",
    "chart4": "#FF6B6B",
    "chart5": "#38BDF8",
}
