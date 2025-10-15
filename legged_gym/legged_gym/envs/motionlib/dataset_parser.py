"""Utilities for inspecting motion dataset files.

This module mirrors the assumptions made by the various ``MotionLib``
implementations in :mod:`legged_gym.legged_gym.envs.motionlib`.  The goal is to
provide a lightweight way to load the YAML configuration files that reference
``.pt`` motion clips and to summarise the tensors stored inside each clip.

Example
-------
>>> parser = DatasetParser("resources/config/loco.yaml")
>>> description = parser.parse()
>>> print(description.format_summary())

The parser only touches metadata – it never mutates the underlying tensors –
so it is safe to use for debugging or visualisation pipelines.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import yaml


@dataclass
class MotionRecord:
    """Metadata describing a single motion clip.

    Attributes
    ----------
    skill:
        Name of the skill bucket in the YAML file (e.g. ``"loco"``).
    name:
        Human readable name derived from the file name.
    file:
        Absolute path to the ``.pt`` file that stores the tensors.
    weight:
        Sampling weight used by :class:`MotionLib`.
    rsi_skipped_range:
        Range specified by the dataset author to skip RSI initialisation
        (see the original YAML config for details).
    num_frames:
        Number of frames contained in this motion clip.
    tensor_shapes:
        Mapping from tensor key to its shape.
    extra_keys:
        Any non-tensor entries that were encountered while loading the clip.
    """

    skill: str
    name: str
    file: str
    weight: float
    rsi_skipped_range: Tuple[float, float]
    num_frames: int
    tensor_shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    extra_keys: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, object]:
        return {
            "skill": self.skill,
            "name": self.name,
            "file": self.file,
            "weight": self.weight,
            "rsi_skipped_range": list(self.rsi_skipped_range),
            "num_frames": self.num_frames,
            "tensor_shapes": {k: list(v) for k, v in self.tensor_shapes.items()},
            "extra_keys": list(self.extra_keys),
        }


@dataclass
class DatasetDescription:
    """Container returned by :class:`DatasetParser.parse`."""

    config_path: str
    skills: Dict[str, List[MotionRecord]] = field(default_factory=dict)

    @property
    def total_frames(self) -> int:
        return sum(record.num_frames for record in self.iter_records())

    def iter_records(self) -> Iterable[MotionRecord]:
        for records in self.skills.values():
            for record in records:
                yield record

    def to_dict(self) -> Dict[str, object]:
        return {
            "config_path": self.config_path,
            "total_frames": self.total_frames,
            "skills": {
                skill: [record.to_dict() for record in records]
                for skill, records in self.skills.items()
            },
        }

    def format_summary(self) -> str:
        """Return a multi-line human readable summary."""
        lines: List[str] = []
        lines.append(f"Dataset config: {self.config_path}")
        lines.append(f"Total frames: {self.total_frames}")
        lines.append("")
        for skill, records in sorted(self.skills.items()):
            lines.append(f"[Skill] {skill} (clips={len(records)})")
            for record in records:
                tensor_info = ", ".join(
                    f"{key}: {shape}" for key, shape in record.tensor_shapes.items()
                )
                extra = f", extras={record.extra_keys}" if record.extra_keys else ""
                lines.append(
                    f"  - {record.name} (frames={record.num_frames}, weight={record.weight})"
                    f" -> {tensor_info}{extra}"
                )
            lines.append("")
        return "\n".join(lines).strip()


class DatasetParser:
    """Parse YAML motion configs and gather tensor metadata."""

    def __init__(self, config_path: str, mapping_file: Optional[str] = None):
        self.config_path = config_path
        self.mapping_file = mapping_file
        self._mapping: Optional[Dict[str, int]] = None

    @property
    def mapping(self) -> Optional[Dict[str, int]]:
        if self.mapping_file is None:
            return None
        if self._mapping is None:
            self._mapping = self._load_mapping(self.mapping_file)
        return self._mapping

    def parse(self) -> DatasetDescription:
        config_path = os.path.abspath(self.config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            motion_config = yaml.load(f, Loader=yaml.SafeLoader)

        if "motions" not in motion_config:
            raise ValueError(f"Config {config_path} does not contain a 'motions' section")

        base_dir = os.path.dirname(config_path)
        description = DatasetDescription(config_path=config_path)

        for skill, entries in motion_config["motions"].items():
            records: List[MotionRecord] = []
            for entry in entries:
                file_rel = entry["file"]
                file_path = os.path.abspath(os.path.join(base_dir, file_rel))
                weight = float(entry.get("weight", 1.0))
                skipped = entry.get("rsi_skipped_range", [])
                if not skipped:
                    skipped = [float("inf"), float("-inf")]
                if len(skipped) != 2:
                    raise ValueError(
                        f"Entry {file_rel} in skill '{skill}' has invalid rsi_skipped_range: {skipped}"
                    )

                tensors, extra_keys = self._load_motion_file(file_path)
                num_frames = self._infer_num_frames(tensors, file_rel)

                if self.mapping is not None:
                    self._validate_mapping(tensors, file_rel)

                record = MotionRecord(
                    skill=skill,
                    name=os.path.splitext(os.path.basename(file_rel))[0],
                    file=file_path,
                    weight=weight,
                    rsi_skipped_range=(float(skipped[0]), float(skipped[1])),
                    num_frames=num_frames,
                    tensor_shapes={k: v.shape for k, v in tensors.items()},
                    extra_keys=tuple(sorted(extra_keys)),
                )
                records.append(record)
            description.skills[skill] = records
        return description

    def _load_motion_file(self, file_path: str) -> Tuple[Dict[str, torch.Tensor], Tuple[str, ...]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Motion file '{file_path}' does not exist")
        data = torch.load(file_path, map_location="cpu")
        if not isinstance(data, dict):
            raise TypeError(f"Motion file '{file_path}' should contain a dict, got {type(data)!r}")
        tensors: Dict[str, torch.Tensor] = {}
        extra_keys: List[str] = []
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                tensors[key] = value
            else:
                extra_keys.append(key)
        return tensors, tuple(extra_keys)

    @staticmethod
    def _infer_num_frames(tensors: Dict[str, torch.Tensor], file_rel: str) -> int:
        if not tensors:
            raise ValueError(f"Motion file '{file_rel}' does not contain any tensors")
        frame_lengths = {tensor.shape[0] for tensor in tensors.values() if tensor.ndim > 0}
        if len(frame_lengths) != 1:
            raise ValueError(
                f"Inconsistent leading dimensions in '{file_rel}': {sorted(frame_lengths)}"
            )
        return frame_lengths.pop()

    @staticmethod
    def _load_mapping(mapping_file: str) -> Dict[str, int]:
        if not os.path.exists(mapping_file):
            raise FileNotFoundError(f"Mapping file '{mapping_file}' does not exist")
        mapping: Dict[str, int] = {}
        with open(mapping_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                value, key = line.split(" ", 1)
                mapping[key] = int(value)
        return mapping

    def _validate_mapping(self, tensors: Dict[str, torch.Tensor], file_rel: str) -> None:
        joint_pos = tensors.get("joint_position")
        joint_vel = tensors.get("joint_velocity")
        if joint_pos is None or joint_vel is None:
            missing = [k for k in ("joint_position", "joint_velocity") if k not in tensors]
            raise KeyError(
                f"Motion file '{file_rel}' is missing required joint tensors: {missing}"
            )
        expected = max(self.mapping.values()) + 1  # type: ignore[arg-type]
        if joint_pos.shape[1] < expected or joint_vel.shape[1] < expected:
            raise ValueError(
                f"Joint tensors in '{file_rel}' have insufficient width. "
                f"Expected at least {expected}, got {joint_pos.shape[1]}"
            )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse motion dataset files")
    parser.add_argument("config", help="Path to the YAML config that lists the motion clips")
    parser.add_argument(
        "--mapping-file",
        help="Optional joint mapping file used to validate joint tensor widths",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        help="Optional file path to dump the parsed metadata as JSON",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    parser = DatasetParser(args.config, mapping_file=args.mapping_file)
    description = parser.parse()
    print(description.format_summary())
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump(description.to_dict(), f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
