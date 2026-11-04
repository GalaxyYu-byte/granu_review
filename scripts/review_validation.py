"""继承原始校验配置，并将已批准问题投影到处理后的完整剩余记录组。

本模块只判断后续扫描发现能否追溯到原复核授权，不读取或写入 CSV。
输入凭据、动作合法性和实际数据仍由处理器及独立审计核验；这里不能把
“某些成员曾经复核过”扩大解释成任何新重复问题都已获准保留。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import math

try:
    from .label_config import ASPECT_COLUMNS
    from .validate_dataset import normalize_for_comparison
except ImportError:
    from label_config import ASPECT_COLUMNS
    from validate_dataset import normalize_for_comparison


DUPLICATE_RULES = {
    "within_split_exact_duplicate", "within_split_normalized_duplicate",
    "cross_split_exact_overlap", "cross_split_normalized_overlap",
}


def validation_options(report: dict) -> dict:
    """提取并验证可继承的扫描参数，防止处理后校验悄悄退回默认口径。

    阶段和根目录由调用方明确提供，不从旧报告继承。缺省阈值表示未启用，
    不能根据本次数据自行选择新阈值；非法配置直接拒绝，不能静默忽略。
    """
    configuration = report.get("configuration", {})
    if not isinstance(configuration, Mapping):
        raise ValueError("原始校验配置必须是对象。")
    options = {"id_scope": configuration.get("id_scope", "split"),
               "max_text_chars": configuration.get("max_text_chars"),
               "min_class_fraction": configuration.get("min_class_fraction")}
    if not isinstance(options["id_scope"], str) or options["id_scope"] not in {"split", "global"}:
        raise ValueError("原始校验的 ID 作用域不合法。")
    maximum = options["max_text_chars"]
    if maximum is not None and (type(maximum) is not int or maximum <= 0):
        raise ValueError("原始校验的文本长度阈值必须是正整数。")
    fraction = options["min_class_fraction"]
    if fraction is not None and (isinstance(fraction, bool)
                                 or not isinstance(fraction, (int, float))
                                 or not math.isfinite(fraction) or not 0 < fraction < 1):
        raise ValueError("原始校验的类别占比阈值必须位于 0 与 1 之间。")
    return options


def _index(items) -> dict:
    """兼容列表和已建索引的对象，同时拒绝重复编号，避免覆盖掉互相冲突的授权。"""
    values = list(items.values()) if isinstance(items, Mapping) else list(items)
    result = {item["issue_id"]: item for item in values}
    if len(result) != len(values):
        raise ValueError("复核匹配输入包含重复问题编号。")
    return result


def _key(record: dict) -> tuple[str, int]:
    """使用处理前的集合和记录序号定位隔离与清洗，不受输出行号移动影响。"""
    return record["split"], record["record_number"]


def _references(records: list[dict]) -> dict:
    """比较决定的定位引用；重复引用不能因字典去重而获得有效授权。"""
    result = {_key(record): record["id"] for record in records}
    if len(result) != len(records):
        raise ValueError("复核匹配输入包含重复记录引用。")
    return result


def _duplicate_rule(records: list[dict]) -> str | None:
    """根据完整组当前的正文和集合边界，推导扫描器实际应报告的重复规则。"""
    if len(records) < 2:
        return None
    splits = {record["split"] for record in records}
    if not splits <= {"train", "validation"}:
        return None
    texts = {record["text"] for record in records}
    if len(texts) == 1:
        comparison = "exact"
    elif len({normalize_for_comparison(text) for text in texts}) == 1:
        comparison = "normalized"
    else:
        return None
    return (f"cross_split_{comparison}_overlap" if len(splits) > 1
            else f"within_split_{comparison}_duplicate")


def _signature(records: list[dict]) -> Counter:
    """以多重集合核对集合、ID、正文和全部方面标签，保留重复成员的数量信息。"""
    return Counter((record["split"], record["id"], record["text"],
                    tuple(record["labels"][aspect] for aspect in ASPECT_COLUMNS))
                   for record in records)


def match_review_approvals(issue: dict, source_issues, decisions,
                           quarantined_keys, edited_texts: Mapping) -> list[str]:
    """返回能够完整解释本项处理后 REVIEW 的原始已批准问题编号。

    原组先应用全部已批准隔离和精确正文清洗，再与扫描发现比较。规则变化
    必须来自这次完整投影：例如训练成员移出后，跨集合重叠变为验证集内部
    重复；原组的排版差异随隔离消失后，归一化重复也可能变成完全重复。
    剩余成员不足两条、无关新成员、任意子组和未知规则均不能继承授权。
    """
    if issue.get("severity") != "REVIEW" or issue.get("rule_id") not in DUPLICATE_RULES:
        return []
    records = issue["records"]
    if _duplicate_rule(records) != issue["rule_id"]:
        return []
    target = _signature(records)
    by_decision = _index(decisions)
    matches = []
    for issue_id, source in _index(source_issues).items():
        decision = by_decision.get(issue_id)
        if (not decision or source.get("rule_id") not in DUPLICATE_RULES
                or _duplicate_rule(source["records"]) != source["rule_id"]
                or decision.get("status") != "reviewed_pending_application"
                or decision.get("action_applied") is not False or decision.get("label_changes") != []
                or decision.get("rule_id") != source["rule_id"]
                or decision.get("original_severity") != source.get("severity")
                or decision.get("source_sha256") != source.get("source_sha256")
                or _references(decision["affected_records"]) != _references(source["records"])):
            continue
        action = decision["action"]
        if action["type"] == "retain_reviewed_records":
            # 训练集冲突不属于保留策略；验证冲突只保留原标，不在此裁决标签。
            if (source["rule_id"].startswith("cross_split_") or action.get("quarantine")
                    or action.get("target")
                    or (any(record["split"] == "train" for record in source["records"])
                        and any(len({record["labels"][aspect] for record in source["records"]}) > 1
                                for aspect in ASPECT_COLUMNS))):
                continue
        elif action["type"] == "quarantine_training_overlap":
            expected = {key: identity for key, identity in _references(source["records"]).items()
                        if key[0] == "train"}
            if (not source["rule_id"].startswith("cross_split_") or not expected
                    or _references(action["quarantine"]) != expected
                    or not expected.keys() <= set(quarantined_keys)):
                continue
        else:
            continue
        # 必须投影整组，而不是只挑当前发现中出现的成员；否则新子组可能被误放行。
        projected = [{**record, "text": edited_texts.get(_key(record), record["text"])}
                     for record in source["records"] if _key(record) not in quarantined_keys]
        if (_duplicate_rule(projected) == issue["rule_id"]
                and len(projected) >= 2 and _signature(projected) == target):
            matches.append(issue_id)
    return sorted(matches)
