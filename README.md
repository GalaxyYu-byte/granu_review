# 细粒度评论数据校验

提供原始数据只读校验、逐条异常复核，以及按复核决定生成清洗和隔离副本的流程。原始数据始终保留；标签映射是后续独立步骤。

## 运行

需要 Python 3.10 或更新版本，无需安装第三方依赖。在项目根目录执行：

```powershell
python .\scripts\validate_dataset.py
```

默认输入：

- `data/raw/sentiment_analysis_trainingset.csv`
- `data/raw/sentiment_analysis_validationset.csv`

默认输出：

- `data/reports/raw/validation_report.json`：总体状态、扫描完整性、输入文件 SHA-256、配置、统计和规则版本。
- `data/reports/raw/validation_issues.jsonl`：每行一个问题组，包含全部关联记录的原文、标签、定位信息、逐方面标签冲突和待复核字段。

脚本默认路径相对于项目定位，从其他目录运行也能找到默认输入。可以显式指定路径：

```powershell
python .\scripts\validate_dataset.py --input-dir .\data\raw --output-dir .\data\reports\raw
```

`--train-csv`、`--validation-csv` 可以分别覆盖两个文件路径。报告目录必须位于所有输入文件所在目录之外。重复运行只替换上述两份生成报告，不触碰其他文件。

## 规则和状态

| 级别 | 规则 | 要求 |
| --- | --- | --- |
| ERROR | 文件缺失、UTF-8/CSV 解析失败、表头缺失/重复/多余/乱序、坏列宽、空 ID、集合内重复 ID、空白文本、非法标签、空数据集、扫描期间输入变化 | 修复或隔离，再重新校验 |
| REVIEW | 跨集合文本重叠、归一化疑似重复、相同文本标签冲突、配置为全局 ID 时的跨集合 ID 重复 | 记录复核结论及处理动作，再用于正式评估 |
| WARN | 集合内文本和标签一致的完全重复、缺失类别、替换字符或异常控制字符，以及显式阈值触发的长度/类别占比问题 | 根据内容与用途判断 |

状态按优先级输出：

1. 有 ERROR：`failed`。
2. 无 ERROR、有 REVIEW：`needs_review`。
3. 仅 WARN：`passed_with_warnings`。
4. 没有问题：`passed`。

CLI 退出码：`0` 表示通过（可能带告警），`1` 表示阻断错误，`2` 表示待复核。参数或输出路径配置错误也由 argparse 返回 `2`，此时终端会显示用法错误，而非数据校验汇总。自动化调用应结合报告的 `status` 判断结果。报告 `validation_gate_passed` 仅表达本次所选校验范围是否通过，不替代模型验证或来源审计。

解析失败、不可解释的表头或源文件变化会设置 `scan_complete=false`。统计只能视为已读取部分的结果；合法宽度记录数和每个方面的有效标签数提供各自分母。`issue_counts` 和 `rule_counts` 按问题条目/问题组计数，并非受影响样本数。

## 字段、标签和 ID

- 固定检查 `id`、`content` 和 `scripts/label_config.py` 的 `ASPECT_COLUMNS` 中的 20 个标签列，顺序也必须一致。保留原始拼写 `environment_cleaness`。
- 使用 `csv.reader(strict=True)`、`utf-8-sig` 和 `newline=""` 解析；兼容 UTF-8 BOM、逗号、引号和文本内部换行。
- `record_number` 为不包含表头、从 1 开始的数据记录序号；`physical_line_end` 单独记录该记录结束的物理行号。
- 原始标签允许 `-2、-1、0、1`。`-2` 是“未提及”，属于有效类别。
- 合法十进制整数字符串会解析为整数用于检查，不修改 CSV。`1.0`、空值、`nan`、带首尾空白或越界标签会报错，不擅自修正。
- 默认 ID 作用域为 `split`，按 `(split, id)` 定位。报告记录 ID 是否与本文件从 0 开始的记录序号吻合；这只是观察到的编号约定，不证明原始评论没有全局标识。
- 确认 ID 是全局标识时，使用 `--id-scope global`，跨集合相同 ID 会进入待复核。

## 训练和评估共用标签配置

`scripts/label_config.py` 是标签定义的唯一来源。`ASPECT_COLUMNS` 决定方面顺序；`CLASS_DEFINITIONS` 同时派生 `CLASS_VALUES`、`LABEL_MAPPING` 和 `RAW_LABEL_MEANINGS`。映射是只读的，模型配置中的类别顺序也由 `CLASS_VALUES` 生成，不另写一份列表。现有校验、复核、清洗脚本均直接引用这个模块；旧的 `scripts.validate_dataset` 常量导入仍兼容。

后续训练集、验证集使用同一个编码入口，按列名读取每个原始 CSV 记录：

```python
from scripts.label_config import (
    ASPECT_COLUMNS, NUM_ASPECTS, NUM_CLASSES, encode_labels,
    get_label_config, validate_label_config,
)

# row 为 csv.DictReader 读取的一条原始记录；两个集合调用同一函数。
labels = encode_labels(row)
# labels[i] 对应 ASPECT_COLUMNS[i]，类别编号为 0～NUM_CLASSES-1。
```

编码前调用共用的 `parse_label` 严格校验原始值。只接受 Python `int`（排除 `bool`）和符合 ASCII `-?[0-9]+` 的整数字符串，且必须属于原始标签域。浮点数（包括 `1.0`）、布尔值、空值、`nan`、带首尾空白的数字、小数或指数文本、越界值和缺失方面列都会报错，错误指出具体方面；不使用 `int(float_value)` 截断、不填默认值。沿用原有 CSV 规则，`"01"`、`"-0"` 等合法整数文字仍可解析。编码仅对原始值执行一次，不对已经映射的类别编号重复编码。

模型输出约定为 `[batch_size, NUM_ASPECTS, NUM_CLASSES]`，第 `i` 个方面对应 `ASPECT_COLUMNS[i]`；评估用同一顺序将预测列与真实标签列逐一比较，并保持样本对应关系。原始 `-2` 是“未提及”类别，会编码为 `0`。

保存模型时，将 `get_label_config()` 的结果随权重保存，或保存为同目录的 `label_config.json`。以下示例假定 `model_dir` 已建立：

```python
import json
from pathlib import Path

config_path = Path(model_dir) / "label_config.json"
config_path.write_text(
    json.dumps(get_label_config(), ensure_ascii=False, indent=2),
    encoding="utf-8",
)

# 加载模型时，先检查配置，再使用模型权重进行预测或评估。
saved_config = json.loads(config_path.read_text(encoding="utf-8"))
validate_label_config(saved_config)
```

配置包含 `aspect_columns` 和 `class_values`；`class_values[i]` 表示模型类别编号 `i` 对应的原始标签。缺少配置、方面顺序或类别顺序不一致、类别类型不合法时，加载检查抛出 `ValueError`，避免仅凭输出维度相同就继续评估。配置返回独立副本，可直接进行 JSON 保存读取。当前仓库尚无模型训练与评估脚本，这些接口供后续脚本共用。

## 文本比较与冲突

两层检测：

1. 原文完全一致。
2. Unicode NFC 规范化，将连续 Unicode 空白折叠为一个 ASCII 空格，再去除首尾空白。

第二层仅生成比较键，保留原文和原始词语边界，不去掉全部空白、不去引号、不做繁简转换。引号、正常换行、制表符及表情本身不视为异常。

归一化后相同的文本仍需要复核。逐方面标签冲突保存在 `label_conflicts` 中，每个方面列出所有成员的集合、ID、记录序号和标签值。一个组可能包含同一集合的多个成员，不能简化为一对样本。

同一批成员若原文完全一致，不再重复生成归一化问题。一个更大的归一化组可能包含已经报告过的完全重复子组，因此两种组数不能直接相加当作唯一受影响样本数。

## 阈值

默认统计文本长度（原始 Unicode 字符数）和各方面的标签分布，不对长度或类别占比设置主观阈值。缺失类别属于客观 WARN。可以显式开启阈值：

```powershell
python .\scripts\validate_dataset.py --max-text-chars 4000 --min-class-fraction 0.01
```

以上数值仅为调用示例，不代表本项目的推荐标准。长度分位数使用 nearest-rank；类别占比的分母为该集合、该方面的有效标签数量。零占比由“缺失类别”规则处理，不再重复触发占比规则。

## 复核记录

每条问题包含规则版本、输入文件摘要和稳定的 `issue_id`。相同文件、相同规则、相同问题的编号在重复运行时保持一致；运行时间不参与问题编号计算。

所有 REVIEW 初始状态为 `pending`。复核记录应单独保存为 `data/reports/raw/review_decisions.jsonl`，每行包含：

- `issue_id` 和对应输入的 `source_sha256`；
- `status`、`conclusion`、`reason`、`action`；
- `affected_records`，使用集合、ID 和记录序号定位；
- `reviewed_at`，使用带时区的时间。

脚本不会创建、覆盖或自动应用复核决定；重新运行不会将原始扫描发现改成“已解决”。“已查看”不代表问题已关闭。对确认的跨集合重复，后续处理需要记录隔离策略，再校验处理后的数据；不直接用训练标签覆盖验证标签。

## 清洗和映射后二次校验

后续生成处理后的 CSV 时，可以复用相同的结构、文本、ID 和重复检查，仅切换合法标签范围：

```powershell
python .\scripts\validate_dataset.py --stage mapped --input-dir .\data\processed --output-dir .\data\reports\mapped
```

`mapped` 模式允许 `0、1、2、3`，对应约定映射 `-2→0、-1→1、0→2、1→3`。输入文件名默认与原始文件相同，也可用两个 CSV 路径参数覆盖。

当前代码负责检查 CSV 本身。逐样本验证映射是否正确、样本数变化是否与去重/隔离日志对应，以及训练张量类型检查，需要在后续清洗/映射流程提供来源记录后完成；`mapped` 通过不代表这些来源对账检查已完成，报告 `limitations` 会明确保留这一范围。JSON 报告不替代原始数据集，JSONL 问题明细也不是训练输入。

## 测试

```powershell
python -m unittest discover -s tests -v
```

测试使用临时 CSV，不依赖大数据集，覆盖正常 CSV、多行文本、结构和标签错误、扫描中断、ID 作用域、归一化重叠及逐方面冲突、阈值告警和原始数据写保护。

## 本批异常的逐条复核

`data/reports/raw/review_decisions.jsonl` 保存本批 15 组 REVIEW 和 13 条字符 WARN 的复核建议；每项包含输入摘要、原文证据、冲突方面或字符位置，以及具体处理动作。`review_summary.json` 保存汇总和全量只读预演。

所有建议标记为 `reviewed_pending_application`，`action_applied=false`。此次仅复核和预演，没有生成 processed 数据，也没有将原始校验问题改为“已解决”。固定验证集仍保留全部样本和原标签；两条重叠验证记录的标签争议另行保留。

可复算代码位于 `scripts/review_dataset_issues.py`，逐组解释的输入为同目录下的三份 `review_notes_*.json`。代码会核对输入文件摘要、问题明细摘要、每条原始记录及每段证据引文，并拒绝将旧注释套用到新问题。复算 Notebook 为 `notebooks/review_raw_issues.ipynb`，其代码只调用只读函数，不覆盖现有决定。

如需将复算结果另存到新目录：

```powershell
python .\scripts\review_dataset_issues.py --output-dir .\data\reports\raw_review_rerun
```

脚本拒绝覆盖已有复核决定。预演仅检查候选记录的条数、空文本、已定义的重复及异常字符规则、标签分布，不替代后续实际落地后的再次校验，也不宣称完成全量语义去重或标签重标。

## 落实已复核的处理方案

```powershell
python .\scripts\process_dataset.py
```

该步骤先核对原始文件、原始校验报告、异常明细、复核决定和复核汇总的 SHA-256，再执行明确记录的隔离或控制字符删除。当前批次的结果：

- `data/processed/sentiment_analysis_trainingset.csv`：104,969 条，隔离 29 条重复冲突记录和 2 条损坏记录；保留记录中 10 条移除已定位控制字符。
- `data/processed/sentiment_analysis_validationset.csv`：15,000 条，全部保留；其中 1 条移除已定位控制字符。
- `data/quarantine/sentiment_analysis_trainingset_quarantine.csv`：完整保留 31 条原始记录，便于后续追溯或恢复。
- `data/reports/processed/processing_changes.jsonl`：11 条文本编辑的来源、字符数量、前后长度及摘要，共删除 21 个控制字符。
- `data/reports/processed/application_results.jsonl`：28 项复核建议的实际应用结果。
- `data/reports/processed/processing_report.json`：条数对账、全部输入/产物摘要、隔离来源和已知验证标签争议。
- `data/reports/processed/validation_report.json`、`validation_issues.jsonl`：处理后的二次校验报告和问题明细。当前批次为 `passed`，ERROR/REVIEW/WARN 均为 0。

保留记录的 ID、全部 20 个标签值、原始顺序均逐条与来源对照；文本只允许发生复核决定中的控制字符删除。隔离记录不重标，保留记录不重新编号。正常换行、表情和 `/流泪` 等情感文本都保留。

此时标签仍是 `-2、-1、0、1`，尚未映射。处理后二次校验使用 `stage=raw` 表示原始标签域，并不表示检查对象还是 raw 目录。两个固定验证样本的标签争议仍记录在处理报告中，不能因结构校验通过就认为这些标签已修复。

`data/reports/raw` 下的报告和 `reviewed_pending_application` 复核记录是处理前的快照，保持不变。实际处理状态以 `data/reports/processed/application_results.jsonl` 和 `processing_report.json` 为准；隔离只解除训练使用风险，不代表底层标注或缺字已恢复。

二次校验通过、来源逐条对账通过且输入摘要未变化后才发布最终 CSV；处理报告最后发布作为完成凭据。脚本拒绝覆盖已有产物，并拒绝把输出写入原始输入目录。需要重新处理时，使用新的 `--output-dir`、`--quarantine-dir` 和 `--reports-dir`。

若要单独重复校验现有处理结果，使用另一个报告目录，以保留此次处理凭据中的产物摘要：

```powershell
python .\scripts\validate_dataset.py --stage raw --input-dir .\data\processed --output-dir .\data\reports\processed_recheck
```

## 清洗后独立复核（2026-11-03）

本轮重新读取全部原始、处理后和隔离 CSV，不以之前的 `passed` 作为文本未误删的证据。可复算代码为 `scripts/audit_processed_dataset.py`，只读 Notebook 为 `notebooks/review_processed_data.ipynb`。复核产物另存于 `data/reports/processed_review`，原数据和原处理凭据均保留。

- 全量对照 104,969 条训练记录、15,000 条验证记录，共 2,399,380 个标签单元，ID、原始顺序、全部标签均一致。
- 119,958 条保留记录的文本完全一致；另 11 条仅删除预先复核的 21 个控制字符，未发现计划外文本修改。31 条隔离记录与原始记录逐字段一致。
- 全部 11 条编辑记录另行阅读全文和每个删除位置，未发现汉字、数字、标点、普通空格、换行、表情文字或情感词误删。U+001C/U+001E 的 4 处删除也逐处检查，没有数字或英文拼接。
- `train/27384` 有轻微表情 token 边界变化：删除 U+0003 后形成 `/色/色得出结论`、`/流泪基本上`。表情文字与正文均完整，人类可读含义未变；后续模型预处理需要明确表情 token 识别规则，不能据此声称任意分词器输出不变。
- 新目录中的完整规则校验再次得到 `passed`，ERROR/REVIEW/WARN 均为 0。这不关闭下述语义标签争议。

两条验证记录 `13036`、`14365` 的全文与原始 CSV 完全一致，全部 20 个方面均另行分析。原始标签含义：`-2` 未提及、`-1` 负面、`0` 中性、`1` 正面。原有冲突方面的结论如下：

| 验证 ID | 方面 | 原标签 | 本轮判断与证据 |
| --- | --- | --- | --- |
| 13036 | 上菜速度 | -2 | 高置信修订候选 -1；“点了米饭，催了好几次才上来”。 |
| 13036 | 噪音 | 0 | 高置信修订候选 -2；全文没有噪音评价，临马路不等于评价嘈杂。 |
| 13036 | 整体体验 | 0 | 高置信修订候选 -1；“这顿饭吃的实在很窝火”。 |
| 13036 | 口味 | 0 | 保留中性合理；“还可以”“不错”与“单调”“失望”并存，不能直接照搬训练标签 1。 |
| 13036 | 价格水平 | 0 | 0/-1 需裁决；“不然这么一份要68块还真不便宜”有负面倾向，也存在条件句解释。 |
| 14365 | 价格水平 | 1 | 倾向 0，保留 0/1 歧义；明确报价，但“性价比还是很不错的”不能单独证明价格便宜。 |
| 14365 | 分量 | -2 | 疑似漏标；“两个人吃有点撑了”明确提及分量，倾向 0，是否取正面 1 需统一规则。 |

新增两项标注口径疑点：`13036` 的“吃鸡送面”是否计入价格优惠，以及 `14365` 的“值得一去”是否可推断再次消费意愿。它们原来没有同组标签冲突，也不代表原标签已获验证。

以上为助手基于文本证据提出的修订候选；仓库未提供完整的原始标注细则，不把训练标签当作真值，不修改固定验证标签。若后续重标，应另存有版本说明的审校数据，并同时保留原验证集的评估结果。

具体证据：

- `lineage_audit.json`：全量对账、新校验状态、输入与原产物摘要。
- `text_edit_evidence.json`、`text_edit_semantic_review.json`：全部实际编辑、删除位置和逐条语义判断。
- `validation_dispute_evidence.json`、`validation_13036_analysis.json`、`validation_14365_analysis.json`：两条原文、训练对照和逐方面判断。
- `review_summary.json`：本轮结论、未决事项与复核产物摘要。

后续可复算到新目录，拒绝覆盖已经存在的复核产物：

```powershell
python .\scripts\audit_processed_dataset.py --output-dir .\data\reports\processed_review_rerun
```

自动化部分复算所有记录差异与来源；语义意见仍需结合原文阅读，不能从脚本通过推导出全部标签正确。

## 复核流程的执行约束

处理前必须满足 `review_complete=true`，并且每条决定均为
`status=reviewed_pending_application`、`action_applied=false`。待复核、被否决、
已经应用或缺失状态的决定都会被拒绝，不生成处理产物。

重复问题按已复核证据区分处理：集合内标签一致的完全或归一化重复使用
`retain_reviewed_records` 保留所有记录；训练集标签冲突仍隔离；跨集合重叠仍只隔离训练侧。
验证集内部的冲突保留原文和标签，并继续记录为争议，不能视为已经重标。
争议成员从实际冲突组推导，不固定写成两条。

对明确批准保留的重复组，二次校验只允许成员、原文及标签与复核方案一致的问题通过。
校验报告保留扫描发现的 `issue_counts`，另写 `scan_status` 和 `review_resolutions`
说明哪些问题引用了原复核决定；新的未批准 REVIEW 仍阻止发布。
独立审计会重新核对原始问题、复核状态、汇总摘要、动作目标、应用日志、文本编辑日志和实际 CSV，
即使同步更新某一份日志的文件摘要，也不能绕过这些交叉检查。

三个复核、处理和审计脚本共用按物理文件行读取的 JSONL 函数。
评论内的 U+0085、U+2028、U+2029 不会再把一条 JSON 记录误拆成多行。
字符是否删除仍由逐条复核决定控制，JSONL 读取本身不清洗文本。

新凭据将项目内文件保存为相对于项目根目录的路径，同时保留 SHA-256；
项目外的明确输入保留绝对路径。迁移时复制整个项目的 `data` 目录即可，
旧绝对路径凭据也会根据报告原位置绑定到新根目录，不依赖旧目录是否仍存在。
四个命令都支持 `--root`，默认使用脚本所在项目的根目录，例如只读复核数据副本：

```powershell
python .\scripts\audit_processed_dataset.py --root D:\dataset_copy --output-dir D:\dataset_copy\data\reports\processed_review_rerun
```

原始数据、既有决定、清洗结果和验证集标签均作为历史版本保留。
上述修改修复后续执行和审计规则；标签修订候选仍须在口径确认后另存版本。

### 复核依据与决定覆盖

复核笔记和决定中的 `reason` 必须是非空、非纯空白字符串；`evidence_quotes`
必须是非空列表，其中每段引文都必须非空、非纯空白，并在对应问题原文中原样出现。
复核生成、处理执行和独立审计都会检查这些条件。有效引文不做空格归一化，
避免把并非来自原文的内容当成证据。检查只证明最低证据格式及来源有效，
不能代替对理由是否充分的人工判断。
