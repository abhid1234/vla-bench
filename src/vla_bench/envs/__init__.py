from vla_bench.envs.base import Env, RolloutResult
from vla_bench.envs.mock import MockLIBEROEnv

REGISTRY: dict[str, type[Env]] = {
    "mock-libero": MockLIBEROEnv,
}


def load_env(name: str, **config) -> Env:
    if name not in REGISTRY:
        raise ValueError(f"Unknown env {name!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[name](**config)
