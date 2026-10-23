# AGENTS.md

面向 Codex 的项目指南。**先读这一页，再动代码。**

ai-recruiter
## 项目一句话

用 LangGraph 把招聘域的完整链路编排起来：`用户需求 → 意图识别 → Tool → RAG → ES/Milvus 混合检索 → Rerank → 多维评分 → Reflection → 最终回答`。
目标是**答案可溯源、数字不出错、不满足条件的候选人不被推荐**。

---
## 文档管理
都放入 docs 文件夹里，以 日期+关键词命名，如 20206-10-10-优化精排.md

## 环境（重要）

项目已自带 venv，**不要用系统 python，也不要重新 pip install**：

```bash
cd /Users/andy_mac/Documents/CodeSpace/BackEnd/ai-recruiter
./venv/bin/python <脚本>          # Python 3.11.12，依赖已装齐
```

常用命令：

```bash
./venv/bin/python scripts/init_data.py     # 建 SQLite + 演示数据
./venv/bin/python scripts/index_data.py    # 向量入库（自动选后端）
./venv/bin/python run_demo.py --demo       # 7 个场景全链路
./venv/bin/python run_demo.py -q "问题" --trace   # 单次问答 + Trace
./venv/bin/python tests/test_all.py        # 17 项测试
./venv/bin/python scripts/run_eval.py      # 10 条评测集
./venv/bin/python -m uvicorn api:app --port 8000
```

> `Makefile` 里的 `PY := python` 指向系统解释器，**优先直接用 `./venv/bin/python`**。

---

## 核心不变式（改代码时必须守住）

这 6 条是这个项目区别于「随便串个 LLM」的关键。破坏其中任何一条，都会出现"看起来能跑、实际在骗人"的问题。

### 1. 零依赖必须能跑通
没有 API Key / 没有 ES / 没有模型权重时，整条链路**仍然要真实执行**：
- LLM → `llm.py` 的 Mock（确定性）
- Embedding → `HashEmbedder`（signed hashing + **IDF**，需先 `fit()` 语料）
- 向量库 → `MemoryVectorStore`（纯 Python BM25 + 点积）

只有「语言表达」这一步退化为模板渲染。检索、评分、反思都是真的。
**因此：新增依赖必须是可选的，且必须有回退分支。**

### 2. 硬约束走 filter，不走向量
城市 / 年限 / 薪资 / 学历这类**可精确判定**的条件，必须在 `build_filters()` 里转成向量库 filter，在**召回阶段就过滤掉**。
不要指望"语义相似度"能保证「必须在北京」。

特别注意 `rag/retriever.py::profile_search()`：它的候选池来自**无 filter 的宽召回**（否则命中数为 0 就无从打分），所以**必须自行执行硬约束**。这里漏过一次，导致杭州/北京的候选人漏进"上海后端"的结果里。

### 3. 一票否决优先于分数线
`scoring.py`：硬技能 0 命中，或资历得分 < 0.3 → `vetoed=True`。
`passed = (not vetoed) and total >= pass_score`。

面试 93 分但 Java 完全不会的候选人加权分仍有 66 分，**不能**因此被推荐。
新增评分维度时要想清楚：它是"加权项"还是"否决项"。

### 4. 三后端共用同一套 RRF
`vectorstore/base.py::rrf_fuse` 是 ES / Milvus / 内存**唯一**的融合实现。
不要在某个后端里另写一套融合逻辑，否则换后端会得到不一致的排序，评测集也就失去横向对比价值。

### 5. 答案的每个数字都要能追溯
链路是：**确定性模板渲染事实底稿** → LLM 可选改写 → Reflection 硬检查 → 输出侧正则兜底。

- `answer_node.py::build_facts()` 是答案的**唯一事实来源**
- `reflect_node.py` 的 C2（数值一致性）/C3（约束遵守）即使 LLM 说 `accept` 也强制 `revise`
- `_sanitize()` 兜底修正幻觉数字 —— **改这个正则时必须加负向后视断言**，否则会把 `98.95` 的 `95` 当独立分数，修成 `98.98.95`（踩过）

### 6. 工具失败必须把 plan 步骤标记 done
`tool_node.py` 里，无论成功、失败、缺参，都要把该步骤 `done=True`。
否则 `plan → tool → tool` 无限循环，直接 `GraphRecursionError`。
缺参时写 `pending_input`，由 answer 节点转成追问，**不要**继续往下跑。

---

## 编排图（改路由前先看这里）

`recruiter_agent/graph.py`，10 个节点：

```
START → intent ─┬(置信度<0.45)→ clarify → END
                └→ plan → tool ⟲(还有待办工具)
                        ├→ knowledge → answer      # policy_qa
                        ├→ retrieve → rerank → score → answer
                        └→ answer
                              ↓
                           reflect ─┬(accept)→ END
                                      └(revise)→ tool / retrieve / rerank / answer / knowledge
```

`route_after_reflect()` 的回流优先级：**补工具 > 重检索 > 重写**。
必须校验"确实还有待办工具"才路由到 `tool`，否则 reflect↔tool 会死循环（踩过）。

`policy_qa` 走独立的 `knowledge` 节点，结果存 `knowledge_chunks`（**不要**写进 `retrieved_chunks`，否则反思节点会把制度文档当候选人做数值一致性检查，产生假告警）。

---

## 代码地图

| 文件 | 职责 | 改动频率 |
|---|---|---|
| `graph.py` | 节点注册 + 条件路由 | 改流程时 |
| `state.py` | `RecruitState` 状态契约 | **加节点写字段时必须同步声明**，否则 LangGraph 静默丢弃 |
| `intent_rules.py` | 意图规则 + 中文实体抽取 | 加意图/槽位时 |
| `nodes/tool_node.py` | `PLANS` 意图→工具计划、参数绑定 | 加意图时同步 |
| `nodes/retrieval_node.py` | retrieve / rerank / score / knowledge | 检索逻辑 |
| `nodes/reflect_node.py` | 5 项硬检查 + 定向回流 | 质量规则 |
| `nodes/answer_node.py` | 事实底稿 + 各意图渲染 | 改输出格式 |
| `vectorstore/` | 三后端 + RRF + filter 映射 | 换后端时 |
| `scoring.py` | 七维评分 + 否决 | 加维度时 |

**`state.py` 忘记声明字段是本项目最容易犯的错**——LangGraph 不会报错，只会悄悄丢掉数据，表现为"节点明明写了但下游拿不到"。

---

## 扩展方式

| 想加什么 | 改哪里 | 别忘了 |
|---|---|---|
| 新意图 | `intent_rules.py` 的 `INTENTS` + `_RULES`；`tool_node.py` 的 `PLANS` | 加 `CLARIFY_QUESTIONS`（intent_node）、评测集 case |
| 新工具 | 继承 `BaseTool` + `@register` | 自动进注册表；`required` 必填项要有缺参处理 |
| 新评分维度 | `scoring.py` 的 `DIMENSION_FNS` | `config.scoring_weights` 加权重；判断是加权还是否决 |
| 换向量库 | 实现 `VectorStore` 协议 | 复用 `rrf_fuse`；filter 字段名映射要写对 |
| 加多轮对话 | LangGraph `checkpointer`（`SqliteSaver`） | 当前 `RecruitAgent` 是无状态的 |

**验证标准**：任何改动后 `./venv/bin/python tests/test_all.py` 必须 17/17，且 `scripts/run_eval.py` 的意图准确率不能下降。

---

## 当前基线（改动后应对照）

```
意图准确率        : 100.0%   (10 条评测集)
实体抽取 P/R/F1   : 1.00 / 1.00 / 1.00
检索 nDCG@5       : 0.801
硬约束违反率      : 0.0%
事实一致性        : 100.0%   (校验 20 处数值，0 处不符)
无幻觉候选人      : 100.0%
延迟 P50/P95      : 3 / 5 ms
```

---

## 已知限制（不是 bug，是尚未做）

1. **hash embedding 语义泛化有限** — 相关文本余弦 0.48 vs 无关 0.00（可分性 OK），但远不如 BGE。生产用 `RECRUITER_EMBEDDING_PROVIDER=sentence_transformers`。
2. **ES / Milvus 后端未经真实服务验证** — 开发环境无 Docker，仅有 stub 单测覆盖 mapping / filter 映射 / RRF 对齐。首次接入真实服务时，**先单独跑 `scripts/index_data.py --backend elasticsearch` 验证建索引与召回**，再跑全链路。
3. **中文姓名靠姓氏启发式** — 「欧阳/司马」等复姓与生僻姓氏会漏判（已在 `_NOT_NAMES` 里排除一批动词误判）。
4. **内存后端不持久化** — 重启进程需重新 `index_data.py`。
5. **无状态、无鉴权** — `RecruitAgent` 每次调用都是独立会话；`api.py` 未做认证，生产需补。

---

## 提交前自检

- [ ] `./venv/bin/python tests/test_all.py` → 17/17
- [ ] `./venv/bin/python scripts/run_eval.py` → 意图准确率未下降
- [ ] 若改了 `state.py`：确认新增字段**已声明**
- [ ] 若改了 filter 映射：确认映射到的字段**在 ES mapping / Milvus schema 中真实存在**
      （`max_salary` → `expected_salary` 是逻辑名，不是字段名；`city` 已声明 keyword，不能加 `.keyword`）
- [ ] 若改了解析/渲染正则：**加了负向后视断言防部分匹配**
- [ ] 新增依赖是可选的，且有回退分支
