"""Compatibility helpers for Kaggle ingestion.

The upstream `kaggle` Python package previously required patching for user_agent
handling. This module patches `kagglesdk` at runtime to handle different SDK
versions - older versions with user_agent parameter and newer versions that
handle User-Agent internally.
"""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Any


def patch_kagglesdk_user_agent() -> None:
    """Ensure kagglesdk works correctly across different SDK versions."""
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
        # Check if user_agent is a valid parameter before modifying kwargs
        sig = inspect.signature(original_init)
        param_names = list(sig.parameters.keys())

        if "user_agent" in param_names:
            # Old SDK version with user_agent parameter - apply fix
            if len(args) >= 6 and args[5] is None:
                args = list(args)
                args[5] = default_user_agent
                args = tuple(args)
            if kwargs.get("user_agent") is None:
                kwargs["user_agent"] = default_user_agent
        # New SDK version handles User-Agent internally, no patch needed

        original_init(self, *args, **kwargs)

    KaggleHttpClient.__init__ = patched_init  # type: ignore[assignment]
    KaggleHttpClient._dfp_user_agent_patched = True  # type: ignore[attr-defined]
