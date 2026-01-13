"""Compatibility helpers for Kaggle ingestion.

The upstream `kaggle` Python package has had various issues with the
`KaggleHttpClient` initialization across different versions:

1. Older versions passed `user_agent=None`, causing `requests` to raise
   `InvalidHeader` when preparing a request.
2. Newer versions removed the `user_agent` parameter entirely, but callers
   may still pass it, causing `TypeError: unexpected keyword argument`.

This module patches `kagglesdk` at runtime to handle both cases before
importing `kaggle` (which authenticates at import time).
"""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Any


def patch_kagglesdk_user_agent() -> None:
    """Patch kagglesdk to handle user_agent compatibility issues."""
    try:
        from kagglesdk.kaggle_http_client import KaggleHttpClient
    except Exception:
        return

    if getattr(KaggleHttpClient, "_dfp_user_agent_patched", False):
        return

    original_init = KaggleHttpClient.__init__

    # Check if the original __init__ accepts 'user_agent' parameter
    try:
        sig = inspect.signature(original_init)
        accepts_user_agent = "user_agent" in sig.parameters
    except (ValueError, TypeError):
        # Can't inspect, assume it doesn't accept user_agent (newer behavior)
        accepts_user_agent = False

    default_user_agent = "kaggle-api"

    @wraps(original_init)
    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        if accepts_user_agent:
            # Old API: accepts user_agent, ensure it's not None
            if len(args) >= 6 and args[5] is None:
                args = (*args[:5], default_user_agent, *args[6:])
            if kwargs.get("user_agent") is None:
                kwargs["user_agent"] = default_user_agent
        else:
            # New API: doesn't accept user_agent, remove it if present
            kwargs.pop("user_agent", None)

        original_init(self, *args, **kwargs)

    KaggleHttpClient.__init__ = patched_init  # type: ignore[assignment]
    KaggleHttpClient._dfp_user_agent_patched = True  # type: ignore[attr-defined]


def patch_kaggle_client() -> None:
    """Patch KaggleClient to not pass user_agent to KaggleHttpClient."""
    try:
        from kagglesdk import KaggleClient
        from kagglesdk.kaggle_http_client import KaggleHttpClient
    except Exception:
        return

    if getattr(KaggleClient, "_dfp_patched", False):
        return

    # Check if KaggleHttpClient accepts user_agent
    try:
        sig = inspect.signature(KaggleHttpClient.__init__)
        accepts_user_agent = "user_agent" in sig.parameters
    except (ValueError, TypeError):
        accepts_user_agent = False

    if accepts_user_agent:
        # No need to patch KaggleClient if HttpClient accepts user_agent
        return

    original_init = KaggleClient.__init__

    @wraps(original_init)
    def patched_client_init(self: Any, *args: Any, **kwargs: Any) -> None:
        # Remove user_agent from kwargs before passing to parent
        kwargs.pop("user_agent", None)
        original_init(self, *args, **kwargs)

    KaggleClient.__init__ = patched_client_init  # type: ignore[assignment]
    KaggleClient._dfp_patched = True  # type: ignore[attr-defined]


def patch_all() -> None:
    """Apply all Kaggle compatibility patches."""
    patch_kagglesdk_user_agent()
    patch_kaggle_client()
