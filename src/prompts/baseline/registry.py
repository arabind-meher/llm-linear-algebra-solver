from . import v1, v2, v3

# Ordered list — last entry is always the default (latest)
_VERSIONS: list = [v1, v2, v3]

_REGISTRY: dict[str, any] = {
    "v1": v1,
    "v2": v2,
    "v3": v3,
}

LATEST: str = "v3"


def get_prompt(matrix: list[list[float]], dimension: int, version: str = LATEST) -> str:
    """Return a formatted prompt string for the given matrix and dimension."""
    module = _REGISTRY.get(version)
    if module is None:
        available = list(_REGISTRY.keys())
        raise ValueError(f"Unknown prompt version '{version}'. Available: {available}")
    return module.prompt(matrix, dimension)


def list_versions() -> list[str]:
    """Return all registered version keys in registration order."""
    return [m.__name__.split(".")[-1] for m in _VERSIONS]
