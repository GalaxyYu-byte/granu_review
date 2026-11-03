"""统一定义二十个方面的顺序、原始标签与模型类别的映射及模型标签配置。

读取数据、训练模型和评估预测时必须共用本模块，避免方面顺序或类别解释发生偏移。
原始标签只接受严格的整数或半角整数字符串，不将浮点数、布尔值隐式转换成合法类别。
模型保存时需同时保存标签配置，加载时再核对配置，防止权重与当前代码的标签语义不一致。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from types import MappingProxyType


# 方面顺序同时规定了输入列顺序和模型输出向量顺序，调整时必须同步处理已有模型。
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

# 定义中的位置就是模型类别编号；映射和持久化元数据从同一来源生成，避免双份定义漂移。
CLASS_DEFINITIONS = (
    (-2, "not_mentioned"),
    (-1, "negative"),
    (0, "neutral"),
    (1, "positive"),
)
CLASS_VALUES = tuple(value for value, _ in CLASS_DEFINITIONS)
# 只读映射防止调用方意外修改全局编码规则；返回给调用方的配置则另行复制。
LABEL_MAPPING = MappingProxyType({value: index for index, value in enumerate(CLASS_VALUES)})
RAW_LABEL_MEANINGS = MappingProxyType(dict(CLASS_DEFINITIONS))
NUM_ASPECTS = len(ASPECT_COLUMNS)
NUM_CLASSES = len(CLASS_VALUES)
INTEGER_PATTERN = re.compile(r"-?[0-9]+\Z")


class InvalidLabelError(ValueError):
    """表示原始标签缺失、字面量格式不合法或取值超出允许范围。

    同时保留结构化原因码，供校验报告区分缺失、格式错误与越界，而非解析异常文案。
    """

    def __init__(self, value: object, reason: str):
        self.reason = reason
        super().__init__(f"非法标签 {value!r}：{reason}")


def parse_label(value: object, allowed_labels: Sequence[int] = CLASS_VALUES) -> int:
    """先严格验证类型与整数字面量，再检查取值是否属于当前标签域。

    使用精确的整数类型检查以排除布尔值；字符串只接受半角数字及可选负号，
    不自动去空格、不取整，也不接受科学计数法。允许调用方指定原始或映射后的标签域。
    所有失败保持稳定的原因码，便于数据校验器统计同类问题。
    """
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
    """按固定方面名称读取一条原始记录，返回顺序一致的模型类别编号。

    列的读取依赖方面名称而非字典插入顺序，因此训练集和验证集使用同一编码契约。
    该函数只能对原始标签执行一次，不能对已经映射的类别编号再次调用。
    缺列或非法值立即报错并携带方面名，不补默认值、不截断、不改写输入记录。
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
    """返回可序列化、与调用方后续修改互不影响的模型标签元数据。

    方面列表给出输出向量的轴顺序，类别列表的第几个元素对应同编号模型类别的原始标签。
    两份列表都重新创建，调用方可直接保存为配置而不会污染模块中的全局定义。
    """
    return {
        "aspect_columns": list(ASPECT_COLUMNS),
        "class_values": list(CLASS_VALUES),
    }


def validate_label_config(config: object) -> None:
    """在使用模型前拒绝缺失、类型不符或与当前标签定义不兼容的保存配置。

    不只检查元素集合，还严格检查列表类型、元素类型与顺序。即使两份配置含有相同标签，
    一旦方面或类别位置交换，预测的解释也会变化，因此不能宽松接受。
    """
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
