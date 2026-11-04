"""统一检查复核理由与原文引文，阻止没有实际依据的复核记录获准应用。

复核生成、处理执行和独立审计共用相同的最低证据要求，避免手工改写决定后绕过生成阶段。
这里只确认理由非空且引文确实来自对应原文，不判断理由是否充分，也不替代语义复核。
"""

from __future__ import annotations


def validate_review_evidence(item: dict, records: list[dict]) -> None:
    """要求有效的复核理由，以及逐段能在当前问题原文中原样找到的非空引文。

    去除首尾空白仅用于判断内容是否为空，不将处理后的字符串用于匹配或写回。
    引文必须是非空列表，且每个元素都必须独立通过类型、非空与原文匹配检查；
    这样空字符串恒为任意文本子串的语言行为就不能成为批准复核的依据。
    """
    reason = item.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("复核理由必须是非空且不全为空白的字符串。")

    quotes = item.get("evidence_quotes")
    if not isinstance(quotes, list) or not quotes:
        raise ValueError("证据引文必须是非空列表。")
    for quote in quotes:
        if not isinstance(quote, str) or not quote.strip():
            raise ValueError("每段证据引文必须是非空且不全为空白的字符串。")
        # 保持原有空格、换行与特殊字符逐字匹配，不能先归一化再接受原文中并不存在的引文。
        if not any(isinstance(record.get("text"), str) and quote in record["text"] for record in records):
            raise ValueError("证据引文未在对应问题的原始文本中找到。")
