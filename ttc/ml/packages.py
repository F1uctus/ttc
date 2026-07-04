"""Discovery and layout of installed ttc model packages."""

import json
import os
import warnings
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path

from ttc.ml.tokenizer import Tokenizer, make_tokenizer

ENTRY_POINT_GROUP = "ttc_models"


@dataclass
class ModelPackage:
    dir: Path
    meta: dict

    @classmethod
    def from_dir(cls, d: Path) -> "ModelPackage":
        return cls(dir=d, meta=json.loads((d / "meta.json").read_text("utf-8")))

    @property
    def name(self) -> str:
        return self.meta["name"]

    @property
    def langs(self) -> list[str]:
        return list(self.meta.get("langs", []))

    def graph_path(self, key: str) -> Path:
        if key == "encoder":
            return self.dir / self.meta["encoder"]["graph"]
        return self.dir / self.meta["heads"][key]

    def make_tokenizer(self) -> Tokenizer:
        return make_tokenizer(self.meta["encoder"]["tokenizer"], self.dir)


def find_model_package(lang: str) -> ModelPackage | None:
    if env_dir := os.environ.get("TTC_MODEL_DIR"):
        pkg = ModelPackage.from_dir(Path(env_dir))
        return pkg if lang in pkg.langs else None
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            pkg = ModelPackage.from_dir(Path(ep.load()()))
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"skipping model package {ep.name}: {e}", stacklevel=2)
            continue
        if lang in pkg.langs:
            return pkg
    return None
