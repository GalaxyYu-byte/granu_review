"""Shared aspect order, class encoding and model label configuration.

CSV readers, training and evaluation must use these definitions together.
Only Python integers and ASCII integer strings are accepted as raw labels;
floats and booleans are never coerced into valid classes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from types import MappingProxyType


ASPECT_COLUMNS = (
    "location_traffic_convenience",
    "location_distance_from_business_district",
    "location_easy_to_find",
    "service_wait_time",
    "service_waiters_attitude",
    "service_parking_convenience",
    "service_serving_speed",
    "price_level",
    "price_cost_effective",
    "price_discount",
    "environment_decoration",
    "environment_noise",
    "environment_space",
    "environment_cleaness",
    "dish_portion",
    "dish_taste",
    "dish_look",
    "dish_recommendation",
    "others_overall_experience",
    "others_willing_to_consume_again",
)
EXPECTED_COLUMNS = ("id", "content", *ASPECT_COLUMNS)

# Position is the model class ID. Derive both encoding and saved metadata here.
CLASS_DEFINITIONS = (
    (-2, "not_mentioned"),
    (-1, "negative"),
    (0, "neutral"),
    (1, "positive"),
)
CLASS_VALUES = tuple(value for value, _ in CLASS_DEFINITIONS)
LABEL_MAPPING = MappingProxyType({value: index for index, value in enumerate(CLASS_VALUES)})
RAW_LABEL_MEANINGS = MappingProxyType(dict(CLASS_DEFINITIONS))
NUM_ASPECTS = len(ASPECT_COLUMNS)
NUM_CLASSES = len(CLASS_VALUES)
INTEGER_PATTERN = re.compile(r"-?[0-9]+\Z")


class InvalidLabelError(ValueError):
    """A raw label is missing, has the wrong type or is outside its domain."""

    def __init__(self, value: object, reason: str):
        self.reason = reason
        super().__init__(f"非法标签 {value!r}：{reason}")


def parse_label(value: object, allowed_labels: Sequence[int] = CLASS_VALUES) -> int:
    """Validate before conversion; preserve the CSV validator's reason codes."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise InvalidLabelError(value, "missing")
    if type(value) is int:
        parsed = value
    elif isinstance(value, str) and INTEGER_PATTERN.fullmatch(value):
        try:
            parsed = int(value)
        except ValueError as error:
            raise InvalidLabelError(value, "not_integer_literal") from error
    else:
        raise InvalidLabelError(value, "not_integer_literal")
    if parsed not in allowed_labels:
        raise InvalidLabelError(value, "out_of_range")
    return parsed


def encode_labels(row: Mapping[str, object]) -> list[int]:
    """Read a raw CSV row by aspect name and encode it in the canonical order.

    Call this once on raw values, not on already encoded model class IDs.
    No input values are changed, filled in or silently truncated.
    """
    encoded = []
    for aspect in ASPECT_COLUMNS:
        if aspect not in row:
            raise ValueError(f"缺少标签列：{aspect}")
        try:
            raw_value = parse_label(row[aspect])
        except InvalidLabelError as error:
            raise ValueError(f"标签列 {aspect}：{error}") from error
        encoded.append(LABEL_MAPPING[raw_value])
    return encoded


def get_label_config() -> dict[str, list]:
    """Return fresh JSON-safe metadata to save alongside model weights.

    class_values[i] is the raw label represented by model class i.
    """
    return {
        "aspect_columns": list(ASPECT_COLUMNS),
        "class_values": list(CLASS_VALUES),
    }


def validate_label_config(config: object) -> None:
    """Reject absent or incompatible saved metadata before using a model."""
    if not isinstance(config, Mapping):
        raise ValueError("模型缺少有效的标签配置。")
    aspects = config.get("aspect_columns")
    if (not isinstance(aspects, list)
            or any(type(aspect) is not str for aspect in aspects)
            or aspects != list(ASPECT_COLUMNS)):
        raise ValueError("模型的 aspect_columns 与当前标签顺序不一致。")
    classes = config.get("class_values")
    if (not isinstance(classes, list)
            or any(type(value) is not int for value in classes)
            or classes != list(CLASS_VALUES)):
        raise ValueError("模型的 class_values 与当前类别映射不一致。")
