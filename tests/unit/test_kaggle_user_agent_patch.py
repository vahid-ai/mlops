"""Tests for Kaggle SDK compatibility patches."""

import inspect

import pytest


def test_kagglesdk_user_agent_patch():
    """Test that the patch handles both old and new Kaggle SDK versions."""
    # Import inside the test to avoid side effects at collection time.
    from engines.dlt_engine.dfp_dlt.kaggle_compat import patch_all

    patch_all()

    try:
        from kagglesdk.kaggle_http_client import KaggleHttpClient
    except ImportError:
        pytest.skip("kagglesdk not installed")

    # Check if the new API (no user_agent) or old API (with user_agent)
    try:
        sig = inspect.signature(KaggleHttpClient.__init__)
        accepts_user_agent = "user_agent" in sig.parameters
    except (ValueError, TypeError):
        accepts_user_agent = False

    if accepts_user_agent:
        # Old API: test that None user_agent is replaced
        client = KaggleHttpClient(user_agent=None)
        client._init_session()

        user_agent = client._session.headers.get("User-Agent")
        assert isinstance(user_agent, str)
        assert user_agent

        class DummyRequest:
            @staticmethod
            def to_dict(_request):
                return {}

        # This is where requests would raise InvalidHeader if the UA were None.
        client._prepare_request("security.OAuthService", "IntrospectToken", DummyRequest())
    else:
        # New API: test that user_agent kwarg is stripped without error
        # The patch should allow calling with user_agent even though the
        # underlying __init__ doesn't accept it
        try:
            # This should not raise TypeError about unexpected keyword argument
            client = KaggleHttpClient(user_agent=None)
            # If we get here, the patch worked
        except TypeError as e:
            if "user_agent" in str(e):
                pytest.fail(f"Patch failed to strip user_agent kwarg: {e}")
            raise


def test_patch_is_idempotent():
    """Test that calling patch multiple times doesn't cause issues."""
    from engines.dlt_engine.dfp_dlt.kaggle_compat import patch_all

    # Call multiple times
    patch_all()
    patch_all()
    patch_all()

    # Should not raise
    try:
        from kagglesdk.kaggle_http_client import KaggleHttpClient
        assert getattr(KaggleHttpClient, "_dfp_user_agent_patched", False)
    except ImportError:
        pytest.skip("kagglesdk not installed")
