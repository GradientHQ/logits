from importlib.metadata import PackageNotFoundError, version as _dist_version

__title__ = "logits-sdk"

try:
    # Resolved from the installed distribution metadata, which hatch-vcs
    # stamps from the git tag at build time.
    __version__ = _dist_version(__title__)
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"
