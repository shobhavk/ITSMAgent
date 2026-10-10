"""Shared pytest fixtures. Reuses the Step 14 evaluation fixtures and harness:
a throwaway database, two test API keys, datasets A/B and the shared KB docs."""
import asyncio
import json
import logging

import pytest

from evaluation.harness import FIXTURES, setup_environment

setup_environment()   # must run before any `app` import

from evaluation.harness import load_dataset, seed_knowledge_base, shared_kb_files  # noqa: E402
from app.services import observability  # noqa: E402


@pytest.fixture(scope="session")
def datasets():
    async def build():
        _, df_a = await load_dataset(FIXTURES / "incidents_A.csv")
        _, df_b = await load_dataset(FIXTURES / "incidents_B.csv")
        await seed_knowledge_base(shared_kb_files())
        return df_a, df_b
    return asyncio.run(build())


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []
        self.setFormatter(observability._JsonFormatter())

    def emit(self, record):
        self.lines.append(self.format(record))

    def events(self):
        return [json.loads(x) for x in self.lines]


@pytest.fixture(autouse=True)
def obs_clean():
    """Fresh metrics + a log capture for every test."""
    observability.reset_metrics()
    cap = _Capture()
    old_level, old_prop = observability.OBS_LOGGER.level, observability.OBS_LOGGER.propagate
    observability.OBS_LOGGER.addHandler(cap)
    observability.OBS_LOGGER.setLevel(logging.INFO)
    observability.OBS_LOGGER.propagate = False
    yield cap
    observability.OBS_LOGGER.removeHandler(cap)
    observability.OBS_LOGGER.setLevel(old_level)
    observability.OBS_LOGGER.propagate = old_prop
