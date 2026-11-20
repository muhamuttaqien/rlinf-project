# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate the FR3 Hydra configuration without constructing an env or worker."""

import argparse
import json
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.errors import HydraException
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException


def load_config(overrides: list[str] | None = None) -> DictConfig:
    """Compose only the repository's offline FR3 primary config."""
    directory = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        config = compose(
            config_name="realworld_fr3_collect_data", overrides=overrides or []
        )
    OmegaConf.resolve(config)
    return config


def validate_config(config: DictConfig) -> dict:
    """Check schema, native factory and Gym registration without opening resources."""
    import gymnasium as gym

    from rlinf.envs import get_env_cls
    from rlinf.envs.real.fr3.config import ENV_ID, validation_report
    from rlinf.envs.real.fr3.offline import FR3OfflineRealWorldEnv

    report = validation_report(config)
    if config.cluster.num_nodes != 1 or config.cluster.get("node_groups"):
        raise ValueError(
            "Expected one local CPU configuration with no physical hardware allocation"
        )
    selected = get_env_cls(config.env.eval.env_type, config.env.eval)
    if (
        selected is not FR3OfflineRealWorldEnv
        or gym.spec(ENV_ID).entry_point != "rlinf.envs.real.fr3:create_FR3OfflineTask"
    ):
        raise ValueError("Expected the fixed offline FR3 environment registration")
    report.update(
        valid=True,
        registration=selected.__name__,
        resources_opened=False,
        native_collection_compatible=False,
        collection_blocker="Collector hardcodes RealWorldEnv/numeric actions and lacks typed provenance/unscored episode integration",
    )
    return report


def main() -> None:
    """Accept Hydra overrides and emit a compact validation result, never a rollout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "overrides", nargs="*", help="Hydra key=value overrides for offline validation"
    )
    args = parser.parse_args()
    try:
        report = validate_config(load_config(args.overrides))
    except (
        ValueError,
        TypeError,
        KeyError,
        HydraException,
        OmegaConfBaseException,
    ) as exc:
        sys.stdout.write(
            json.dumps(
                {
                    "valid": False,
                    "error": str(exc),
                    "qualification_gate": "NO_GO",
                    "physical_command_authorization": "NOT_GRANTED",
                }
            )
            + "\n"
        )
        raise SystemExit(1) from exc
    sys.stdout.write(json.dumps(report, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
