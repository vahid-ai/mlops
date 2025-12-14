def test_kagglesdk_user_agent_patch_prevents_none_header():
    # Import inside the test to avoid side effects at collection time.
    from engines.dlt_engine.dfp_dlt.kaggle_compat import patch_kagglesdk_user_agent

    patch_kagglesdk_user_agent()

    from kagglesdk.kaggle_http_client import KaggleHttpClient

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
