"""统一读取 JSONL，并在项目迁移时解析报告和处理凭据中的文件路径。

项目内文件以根目录相对路径保存，项目外输入保留绝对路径。旧版绝对路径
只有在报告位置能够确定旧项目根目录时才重定位，避免误读其他数据目录。
本模块仅负责读取和路径换算；文件摘要及内容一致性由调用方核验。
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath, PureWindowsPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    """按文件实际行边界读取 JSONL，忽略空白行，并返回解码后的记录列表。

    不使用字符串的 splitlines()：该方法还会识别 U+0085、U+2028、U+2029，
    而这些字符可能合法地出现在 JSON 字符串内部。按文件行迭代可以保留它们，
    不会把同一条记录拆开。JSON 格式或编码错误直接交给调用方处理。
    """
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def receipt_path(path: Path, root: Path | None = None) -> str:
    """将实际文件路径转换为可写入凭据的字符串，不访问或修改文件内容。

    root 内的文件使用带正斜杠的相对路径，使整份项目副本可以迁移。
    root 外的文件仍保存绝对路径，避免把外部数据错误地视为项目内部资源。
    未传 root 时，以当前脚本所在项目为基准。
    """
    path = Path(path).resolve()
    root = Path(root or PROJECT_ROOT).resolve()
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)


def _portable_path(value: str):
    """按记录中的路径格式解析旧路径，不依赖当前运行系统的路径规则。

    这里只进行纯路径运算，不查询文件系统；因此可以在另一种操作系统上
    识别旧报告里的 Windows 盘符路径或 POSIX 绝对路径。
    """
    windows = PureWindowsPath(value)
    return windows if windows.is_absolute() else PurePosixPath(value)


def receipt_origin(recorded_report_path: str, actual_report_path: Path,
                   root: Path) -> str | None:
    """根据报告的新旧位置推导旧项目根目录，无法可靠匹配时返回 None。

    recorded_report_path 是凭据中保存的报告位置，actual_report_path 是本次
    实际读取的位置，root 是本次项目根目录。先求实际报告相对 root 的完整
    后缀，再要求旧绝对路径也具有该后缀，才截取旧根目录。

    不能只搜索名为 data 的目录：项目上层或外部数据目录也可能使用这个名字。
    已经采用相对路径的报告不需要推导旧根目录。
    """
    recorded = _portable_path(recorded_report_path)
    if not recorded.is_absolute():
        return None
    try:
        suffix = Path(actual_report_path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return None
    flavor = type(recorded)
    count = len(suffix.parts)
    if not count or flavor(*recorded.parts[-count:]) != flavor(*suffix.parts):
        return None
    return str(recorded.parents[count - 1])


def resolve_receipt_path(value: str, root: Path | None = None,
                         base_dir: Path | None = None,
                         legacy_root: str | Path | None = None) -> Path:
    """将凭据路径解析为本次读取的绝对路径，兼容相对路径及旧版绝对路径。

    相对路径默认以 root 为基准；显式传入 base_dir 时改用该目录。
    绝对路径只有属于 legacy_root 时才换算到新 root，且优先读取新副本，
    不会因为旧目录仍存在就回退读取旧文件。其他外部绝对路径保持原定位。

    legacy_root 应由 receipt_origin 绑定到已知报告位置，不能靠目录名称猜测。
    路径解析成功不代表文件可信，调用方仍须检查文件是否存在及 SHA-256。
    """
    root = Path(root or PROJECT_ROOT).resolve()
    recorded = _portable_path(value)
    if recorded.is_absolute():
        if legacy_root is not None:
            try:
                relative = recorded.relative_to(_portable_path(str(legacy_root)))
            except ValueError:
                pass
            else:
                return root.joinpath(*relative.parts).resolve()
        return Path(value).resolve()
    candidate = (Path(base_dir) if base_dir is not None else root) / Path(value)
    return candidate.resolve()
