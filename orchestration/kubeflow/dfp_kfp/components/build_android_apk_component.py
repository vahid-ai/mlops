"""Component: Bazel build of runtime comparison APK."""

def run(apk_target: str = "//apps/android/runtime_comparison_app") -> str:
    return f"bazel-bin/{apk_target}".replace("//", "")
