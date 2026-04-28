from vla_bench.models.base import VLAModel
from vla_bench.models.mock import MockVLA

REGISTRY: dict[str, type[VLAModel]] = {
    "mock": MockVLA,
}


def load_model(name: str, **config) -> VLAModel:
    if name not in REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[name](**config)
