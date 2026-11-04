"""统一 CSV 字段上限、文件摘要、JSON/JSONL 读写及可迁移的报告凭据路径。

项目内文件以根目录相对路径保存，项目外输入保留绝对路径。旧版绝对路径
只有在报告位置能够确定旧项目根目录时才重定位，避免误读其他数据目录。
本模块提供文件操作工具；摘要是否匹配及内容是否一致仍由调用方核验。
"""

from __future__ import annotations

from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys
import tempfile
from threading import RLock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 直接脚本导入 dataset_io 与包导入 scripts.dataset_io 可能产生两份模块实例。
# 字段上限属于同一个标准库 csv 模块，锁也绑定到该对象，避免混合导入创建两把锁。
_CSV_FIELD_LOCK = csv.__dict__.setdefault("_granu_review_field_size_lock", RLock())


@contextmanager
def large_csv_fields():
    """持共享锁覆盖完整 CSV 解析，结束或异常时恢复调用前的进程级上限。

    读取器与校验器必须共同使用该入口，不能各自设置和恢复上限。锁覆盖
    yield 内的解析过程；同线程嵌套可重入，不同线程等待当前解析完成。
    """
    with _CSV_FIELD_LOCK:
        previous_limit = csv.field_size_limit()
        try:
            field_limit = sys.maxsize
            while True:
                try:
                    # 平台回退候选也不能降低调用方已经设置的有效上限。
                    csv.field_size_limit(max(previous_limit, field_limit))
                    break
                except OverflowError:
                    field_limit //= 10
                    if not field_limit:
                        raise
            yield
        finally:
            csv.field_size_limit(previous_limit)


def file_sha256(path: Path) -> str:
    """分块计算文件摘要，避免将整份数据一次读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value, *, jsonl: bool = False):
    """先在目标目录写临时文件，再原子替换，避免读者看到半份报告。

    每个 JSONL 对象只追加一个真正的换行；字符串内部字符由 JSON 编码器
    处理。调用方负责判断目标是否允许覆盖，本函数仅保证单文件发布完整。
    """
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", suffix=".tmp",
                                         prefix=path.stem + "_", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            if jsonl:
                for item in value:
                    stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
            else:
                json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


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
