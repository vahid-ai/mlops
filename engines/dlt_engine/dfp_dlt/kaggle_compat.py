"""Compatibility helpers for Kaggle ingestion.

The upstream `kaggle` Python package currently constructs a `KaggleHttpClient`
with `user_agent=None` via `kagglesdk.KaggleClient`, which causes `requests` to
raise `InvalidHeader` when preparing a request.

This module patches `kagglesdk` at runtime to ensure a non-None User-Agent is
used before importing `kaggle` (which authenticates at import time).
"""

from __future__ import annotations

from functools import wraps
from typing import Any


def patch_kagglesdk_user_agent() -> None:
    """Ensure kagglesdk never initializes a session with a None User-Agent."""
    try:
        from kagglesdk.kaggle_http_client import KaggleHttpClient
    except Exception:
        return

    if getattr(KaggleHttpClient, "_dfp_user_agent_patched", False):
        return

    original_init = KaggleHttpClient.__init__
    default_user_agent = "kaggle-api"
    if getattr(original_init, "__defaults__", None):
        default_user_agent = original_init.__defaults__[-1] or default_user_agent

    @wraps(original_init)
    def patched_init(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[no-untyped-def]
        # `user_agent` is the 6th arg after `self` in the current signature:
        # (env, verbose, username, password, api_token, user_agent)
        if len(args) >= 6 and args[5] is None:
            args = list(args)
            args[5] = default_user_agent
            args = tuple(args)
        if kwargs.get("user_agent") is None:
            kwargs["user_agent"] = default_user_agent
        original_init(self, *args, **kwargs)

    KaggleHttpClient.__init__ = patched_init  # type: ignore[assignment]
    KaggleHttpClient._dfp_user_agent_patched = True  # type: ignore[attr-defined]
