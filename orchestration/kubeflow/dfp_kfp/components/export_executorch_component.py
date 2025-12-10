"""Component: export PyTorch weights to ExecuTorch."""

def run(model_path: str) -> str:
    return model_path.replace(".pth", ".pte")
