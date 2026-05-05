from vla_bench.models.base import VLAModel
from vla_bench.models.mock import MockVLA
from vla_bench.models.openvla import OpenVLAOFTModel
from vla_bench.models.pi05 import Pi05Model

REGISTRY: dict[str, type[VLAModel]] = {
    "mock": MockVLA,
    "openvla": OpenVLAOFTModel,
    "pi05": Pi05Model,
}


def load_model(name: str, **config) -> VLAModel:
    if name not in REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[name](**config)
