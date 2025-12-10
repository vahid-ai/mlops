"""Component: Feast → LakeFS → MLflow training set builder."""

def run(repo: str, branch: str, table: str) -> str:
    return f"lakefs://{repo}/{branch}/{table}"
