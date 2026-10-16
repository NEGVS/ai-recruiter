# AI 招聘 Agent · LangGraph 编排

> 用 LangGraph 把 **用户需求 → 意图识别 → Tool → RAG → ES/Milvus → Rerank → 评分 → Reflection → 最终回答**
> 这条完整链路编排起来，并且**开箱即跑**：不需要 GPU、不需要向量数据库、不需要 LLM API Key。

```bash
python scripts/init_data.py     # 初始化数据（25 候选人 / 8 岗位 / 6 制度文档）
python scripts/index_data.py    # 向量入库
python run_demo.py              # 交互式体验
```

---

## 一、为什么是 LangGraph

招聘 Agent 的难点不在单点能力，而在**这些能力的正确组合顺序**：

| 环节 | 单独做的问题 | 本项目的处理 |
|---|---|---|
| 意图识别 | 「面试改期要提前多久」被当成排期动作而不是制度问答 | 疑问句式 + 制度性措辞联合消歧，置信度低时反问而非硬猜 |
| 检索 | 纯向量搜不准「张伟」这类精确词，纯 BM25 抓不住语义 | **三路召回 + RRF 融合**：向量(语义) + BM25(精确) + 画像分(结构化) |
| 硬约束 | 靠语义相似度无法保证「必须在北京」 | 城市/年限/薪资走 filter 通道，**召回阶段就过滤** |
| Rerank | 通用 rerank 模型不懂「求职意向」「到岗时间」 | 业务感知精排：硬技能命中、意向、到岗、稳定性直接进排序分 |
| 评分 | 面试 93 分的高分掩盖了硬技能 0 命中 | **一票否决**：硬技能零命中 / 经验严重不足 → 直接不通过 |
| 答案 | 模型容易编造数字、把不匹配的人推荐出去 | 事实底稿 + 规则校验 + 输出侧数值兜底修正 |

LangGraph 的价值在于：**Reflection 发现问题后能定向回流到具体节点**，而不是整条链重跑。
- 召回不足 → 回到 `retrieve`，渐进放宽约束（并在答案中披露放宽了什么）
- 信息缺失 → 回到 `tool`，补一次工具调用
- 只是措辞问题 → 只回到 `answer` 重新组织

---

## 二、架构

```
                    ┌──────────┐
   用户需求 ───────▶│  intent  │ 规则 + LLM 意图识别 / 实体抽取
                    └────┬─────┘
              置信度低    │    置信度高
            ┌────────────┘    └───────────┐
            ▼                             ▼
      ┌──────────┐                  ┌──────────┐
      │ clarify  │                  │  plan    │ Planner：意图 → 工具调用计划
      └────┬─────┘                  └────┬─────┘
           │                             ▼
           │                      ┌──────────┐
           │                      │   tool   │◀──┐ 失败/缺参时标记 done，
           │                      └────┬─────┘   │ 避免 plan→tool 死循环
           │                           │ 还有待办  │
           │                           └──────────┘
           │                             ▼
           │                      ┌──────────┐
           │                      │ retrieve │ 三路召回 + RRF 融合
           │                      └────┬─────┘
           │                           ▼  ┌──────────────┐
           │                      ┌──────────┐           │
           │                      │  rerank  │ 业务感知精排 │
           │                      └────┬─────┘           │
           │                           ▼                 │
           │                      ┌──────────┐           │
           │                      │  score   │ 多维评分+否决│
           │                      └────┬─────┘           │
           │                           ▼                 │ 定向回流
           │                      ┌──────────┐           │
           └─────────────────────▶│  answer  │◀──────────┤
                                  └────┬─────┘           │
                                       ▼                 │
                                  ┌──────────┐           │
                                  │ reflect  │───────────┘
                                  └────┬─────┘
                             accept │  │ revise
                                    │  └──────────▶ 回流到 tool/retrieve/answer
                                    ▼
                                  END
```

`knowledge` 节点（制度问答走独立知识库）由 `tool` 之后条件分流进入。

### 目录结构

```
ai-recruiter/
├── run_demo.py                    # CLI Demo（含完整 Trace 展示）
├── api.py                         # FastAPI 服务
├── recruiter_agent/
│   ├── graph.py                   # ★ LangGraph 编排：节点注册 + 条件路由
│   ├── state.py                   # Agent 状态契约（RecruitState）
│   ├── config.py                  # 全局配置（环境变量覆盖）
│   ├── intent_rules.py            # 意图规则 + 实体抽取（含中文姓名/岗位识别）
│   ├── llm.py                     # LLM 抽象：无 Key 自动回退确定性 Mock
│   ├── embedding.py               # Embedding 抽象：真实模型 / hash+IDF 回退
│   ├── scoring.py                 # 七维可解释评分 + 一票否决
│   ├── vectorstore/
│   │   ├── base.py                # 协议 + RRF 融合 + filter 构造
│   │   ├── es_store.py            # Elasticsearch（BM25 + dense_vector knn）
│   │   ├── milvus_store.py        # Milvus（BM25 Function + dense + RRF）
│   │   └── memory_store.py        # 纯 Python BM25 + 点积（零依赖兜底）
│   ├── rag/
│   │   ├── retriever.py           # Query 改写 / 三路召回 / 硬约束 filter
│   │   └── rerank.py              # 规则 / CrossEncoder / LLM 三级精排
│   ├── tools/
│   │   ├── base.py                # Tool 协议 + 注册表 + 异常归一
│   │   └── business_tools.py      # 9 个业务工具
│   ├── nodes/                     # 各 LangGraph 节点实现
│   └── eval/benchmark.py          # 评测集 + 指标
├── scripts/
│   ├── init_data.py               # 建库 + 演示数据
│   ├── index_data.py              # 向量化入库（ES/Milvus/内存）
│   └── run_eval.py                # 跑评测
├── data/
│   ├── recruit.db                 # SQLite（结构化查询源）
│   └── knowledge/*.md             # 制度知识库
└── tests/test_all.py              # 11 项测试
```

---

## 三、核心设计

### 3.1 混合检索：为什么不能只用向量

招聘场景里两类信号缺一不可：

- **精确词**：候选人姓名、学校、证书、岗位编号 → BM25 强
- **语义**：「大模型」↔「LLM/Agent/智能体」→ 向量强

因此 `retrieve` 节点做三路召回后 RRF 融合：

| 通道 | 打分方式 | 解决的问题 |
|---|---|---|
| `dense` | 余弦相似度 | 语义相近但用词不同 |
| `bm25` | 经典 BM25（k1=1.5, b=0.75） | 精确词命中 |
| `profile` | 结构化字段加权分 | 不靠文本相似度，直接按硬条件打分 |

`profile` 通道的候选池来自**无 filter 的宽召回**，因此它**必须自行执行硬约束**——
否则会出现「语义召回到了，但城市/年限不满足」的候选人漏进结果（这是本项目开发中实际踩到并修复的 bug）。

### 3.2 Rerank：业务感知优先于通用相关性

即使不用任何 rerank 模型，规则 rerank 也能给出**有业务含义**的排序：

```python
rerank_score = RRF融合分 × 2
             + 0.9 × 硬技能命中率
             + 0.25 × 经验达标
             + 0.2  × 城市匹配
             + 0.15 × 求职意向匹配
             + 0.12 × 快速到岗
             - 0.2  × 三年内多次跳动
```

配置 `RECRUITER_RERANK_PROVIDER=cross_encoder` 可切换到 BGE-Reranker，`=llm` 交给模型打分（会自动叠加上述业务信号）。

### 3.3 评分：一票否决 + 全维度可解释

七个维度加权求和，每一维都产出**具体依据**（不是黑箱分数）：

| 维度 | 权重 | 说明 |
|---|---|---|
| 硬技能 | 30% | 命中率，零命中触发否决 |
| 领域经验 | 20% | JD 领域是否覆盖 |
| 资历匹配 | 15% | 过低不够、过高易流失（超配扣分） |
| 面试表现 | 15% | 无面试记录按中性分并标注 |
| 稳定性 | 8% | 平均在职时长 + 三年跳动次数 |
| 薪酬匹配 | 7% | 与岗位带宽比对，超预算明确提示 |
| 求职意向 | 5% | active/passive + 到岗时间 |

**一票否决规则**：硬技能 0 命中，或资历得分 < 0.3 → `vetoed=True`，无论加权分多高都不算「推荐」。

这解决了一个真实问题：面试 93 分但 Java 完全不会的候选人，加权分仍有 66 分，
在普通实现里会被推荐出去；本项目会明确告诉你「**3/3 位候选人触发一票否决，当前候选池中无完全匹配人选**」。

### 3.4 Reflection：五项硬检查 + 定向回流

反思不是让模型泛泛自评，而是**可验证的检查项**：

| 检查 | 内容 | 失败后果 |
|---|---|---|
| C1 证据充分性 | 结论是否可追溯到检索片段/工具结果 | 无证据 → 必须 revise |
| C2 数值一致性 | 答案中的分数是否等于实际计算值 | 不符 → 强制 revise |
| C3 约束遵守 | 是否违反城市/年限约束 | 违反 → 强制 revise |
| C4 完整性 | 多子问题是否都回答了 | 缺项 → revise |
| C5 风险披露 | 候选人风险是否如实说明 | 未披露 → revise |

即使 LLM 说 `accept`，只要规则检查发现 C2/C3 违背，仍然强制 `revise`。

答案生成后还有**输出侧兜底**：正则匹配「姓名…N 分」模式，与真实评分比对，不符则修正
（`98.95` 曾被误改成 `98.98.95`，现已加负向后视断言修复）。

---

## 四、运行

### 4.1 零依赖模式（默认）

```bash
python scripts/init_data.py       # 建 SQLite 库 + 演示数据
python scripts/index_data.py      # 入内存向量库（自动探测 ES/Milvus，不可用则回退）
python run_demo.py                # 交互
python run_demo.py -q "找 5 个北京 3 年以上 LLM 和 RAG 方向的候选人" --trace
python run_demo.py --demo         # 跑内置示例集
```

### 4.2 接入真实 ES / Milvus

```bash
docker run -d --name es -p 9200:9200 \
  -e "discovery.type=single-node" -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" docker.elastic.co/elasticsearch/elasticsearch:8.13.0

export RECRUITER_VECTOR_STORE_BACKEND=elasticsearch
export RECRUITER_ES_URL=http://localhost:9200
python scripts/index_data.py --backend elasticsearch
```

Milvus 同理（`docker-compose up milvus`），设 `RECRUITER_MILVUS_URI=http://localhost:19530`。

三后端的融合逻辑**完全一致**（都是应用层 RRF），因此可以用同一套评测集横向对比排序质量。

### 4.3 接入真实 LLM / Embedding

```bash
cp .env.example .env
# 在 .env 中配置：
# RECRUITER_LLM_PROVIDER=openai
# RECRUITER_LLM_API_KEY=sk-xxx
# RECRUITER_LLM_BASE_URL=https://api.deepseek.com/v1   # 或 任意 OpenAI 兼容端点
# RECRUITER_EMBEDDING_PROVIDER=sentence_transformers
# RECRUITER_RERANK_PROVIDER=cross_encoder
```

未配置 API Key 时自动使用确定性 Mock，**编排链路、检索、评分、反思全部真实执行**，
只有「语言表达」这一步不同——所以回归测试可以在无网络环境下跑。

### 4.4 API 服务

```bash
python -m uvicorn api:app --port 8000
```

| 接口 | 说明 |
|---|---|
| `GET /health` | 健康检查 + 后端信息 |
| `POST /ask` | 问答（`with_trace=true` 返回完整链路） |
| `GET /graph` | 导出 Mermaid 编排图 |
| `GET /candidates` `GET /jobs` | 原始数据查询 |

### 4.5 测试与评测

```bash
python tests/test_all.py        # 15 项测试
python scripts/run_eval.py      # 10 条评测集
```

当前基线（内存后端 + hash embedding + Mock LLM）：

```
意图准确率        : 100.0%
实体抽取 P/R/F1   : 0.89 / 0.89 / 0.89
检索 Recall@5     : 0.722
检索 MRR@5        : 1.0
检索 nDCG@5       : 0.801
硬约束违反率      : 0.0%
事实一致性        : 100.0% (校验 20 处数值，0 处不符)
无幻觉候选人      : 100.0%
延迟 P50/P95      : 4 / 5 ms
```

---

## 五、九个业务工具

| 工具 | 作用 | 关键设计 |
|---|---|---|
| `query_candidates` | 结构化条件预筛 | 纯 SQL，返回命中总量用于判断是否需要放宽 |
| `get_candidate` | 单人完整档案 | 姓名/ID 双索引，模糊匹配 |
| `get_job` | JD 详情 | 三级模糊：完全匹配 → LIKE → 关键词交集 |
| `list_jobs` | 在招岗位 | 支持城市/急招过滤 |
| `schedule_interview` | 面试排期 | 面试官档期冲突检测 → 无冲突时段 → 二次确认写库 |
| `list_interviews` | 日程查询 | — |
| `calc_offer` | Offer 测算 | 默认给**真能达成目标**的结构；对比档案实际期望给出谈判缺口 |
| `market_insight` | 市场分析 | 存量、薪资 P25/P50/P75、意向分布、供需评估 |
| `manage_shortlist` | 人才清单 | 去重 + 容量校验 |

工具是**确定性**的，模型只决定「调哪个、传什么参」。涉及写库的动作（排期）采用
「先返回候选时段 → 用户确认 → 再写入」的两阶段设计。

---

## 六、可扩展点

| 想加什么 | 改哪里 |
|---|---|
| 新意图 | `intent_rules.py` 的 `INTENTS` + `_RULES` + `PLANS`（`tool_node.py`） |
| 新工具 | 继承 `BaseTool` + `@register`，自动进入注册表与 LLM schema |
| 新评分维度 | `scoring.py` 的 `DIMENSION_FNS` + `config.scoring_weights` |
| 换向量库 | 实现 `VectorStore` 协议，`vectorstore/__init__.py` 注册 |
| 接真实简历解析 | 替换 `scripts/index_data.py` 的 `build_candidate_chunks` |
| 加持久化/多轮 | LangGraph `checkpointer`（`SqliteSaver` / `PostgresSaver`），当前为无状态 |

---

## 七、已知限制

1. **hash embedding 质量有限**。它是确定性的 signed hashing + IDF，能保证可分性（相关 0.48 vs 无关 0.00），
   但语义泛化能力远不如 BGE。生产环境请配置 `sentence_transformers`。
2. **中文姓名识别基于姓氏启发式**，对「欧阳/司马」等复姓与生僻姓氏会漏判，此时应在 `get_candidate` 里做兜底。
3. **内存后端不持久化**，进程重启需重新 `index_data.py`。ES/Milvus 后端无此问题。
4. **Mock LLM 的语言组织能力有限**，答案的排版由确定性模板渲染。接入真实 LLM 后表达会显著更自然，
   但事实与数字仍受同一套校验约束。
5. **ES / Milvus 后端未经真实服务验证**（开发环境无 Docker）。两者已通过 stub 单测覆盖
   mapping 构造、filter 字段映射与 RRF 语义对齐；首次接入真实服务时建议先用
   `python scripts/index_data.py --backend elasticsearch` 验证建索引与召回。
