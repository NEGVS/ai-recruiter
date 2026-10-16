"""初始化 SQLite 数据库：建表 + 灌入演示数据（25 位候选人 / 8 个岗位 / 6 篇知识文档）。

运行： python -m scripts.init_data
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recruiter_agent.state import Candidate, JobRequisition  # noqa: E402
from recruiter_agent.tools.business_tools import DB_PATH  # noqa: E402

SCHEMA = """
DROP TABLE IF EXISTS candidates;
DROP TABLE IF EXISTS jobs;
DROP TABLE IF EXISTS interviews;
DROP TABLE IF EXISTS shortlist;
DROP TABLE IF EXISTS scorecard;

CREATE TABLE candidates (
    candidate_id      TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    title             TEXT,
    years             REAL,
    city              TEXT,
    education         TEXT,
    edu_rank          INTEGER,
    school            TEXT,
    major             TEXT,
    skills_json       TEXT,
    domains_json      TEXT,
    projects_json     TEXT,
    highlights_json   TEXT,
    tags_json         TEXT,
    job_intent        TEXT,
    expected_salary   INTEGER,
    current_salary    INTEGER,
    availability_days INTEGER,
    stability_years_per_role REAL,
    job_changes_3y    INTEGER,
    interview_json    TEXT,
    risks_json        TEXT,
    resume_text       TEXT
);

CREATE TABLE jobs (
    job_id        TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    city          TEXT,
    level         TEXT,
    min_years     REAL,
    max_years     REAL,
    salary_min    INTEGER,
    salary_max    INTEGER,
    hard_skills_json TEXT,
    nice_skills_json TEXT,
    domains_json  TEXT,
    education_req TEXT,
    headcount     INTEGER,
    urgency       TEXT,
    description   TEXT
);

CREATE TABLE interviews (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id  TEXT,
    candidate_name TEXT,
    interviewer   TEXT,
    round         TEXT,
    interview_date TEXT,
    start_hour    INTEGER,
    duration      INTEGER,
    status        TEXT
);

CREATE TABLE shortlist (
    job_id       TEXT,
    candidate_id TEXT,
    note         TEXT,
    created_at   TEXT,
    PRIMARY KEY (job_id, candidate_id)
);

CREATE TABLE scorecard (
    candidate_id TEXT,
    job_id       TEXT,
    total        REAL,
    dimensions_json TEXT,
    created_at   TEXT,
    PRIMARY KEY (candidate_id, job_id)
);
"""

# ---------------------------------------------------------------- 岗位
JOBS = [
    JobRequisition(
        job_id="JD-1001", title="AI平台工程师(LLM/RAG方向)", city="北京", level="3-5",
        min_years=3, max_years=6, salary_range=(45000, 70000),
        hard_skills=["Python", "LLM", "RAG", "向量检索", "Kubernetes"],
        nice_skills=["Agent", "vLLM", "Milvus", "LangGraph", "Prompt"],
        domains=["大模型", "AI平台"], education_req="本科", headcount=2, urgency="high",
        description="负责企业级 LLM 应用平台建设：RAG 检索链路、Agent 编排、推理服务优化；"
                    "对接算法与业务团队，落地 AI 能力。团队已有 Milvus/ES 基础设施。",
    ),
    JobRequisition(
        job_id="JD-1002", title="高级后端工程师", city="上海", level="5-8",
        min_years=5, max_years=9, salary_range=(55000, 80000),
        hard_skills=["Java", "分布式", "MySQL", "Redis", "高并发"],
        nice_skills=["Kubernetes", "Kafka", "DDD", "Elasticsearch", "Go"],
        domains=["电商", "金融科技"], education_req="本科", headcount=1, urgency="normal",
        description="负责交易/支付核心链路的高并发服务设计与优化，参与微服务治理。",
    ),
    JobRequisition(
        job_id="JD-1003", title="推荐算法工程师", city="北京", level="3-5",
        min_years=3, max_years=6, salary_range=(50000, 75000),
        hard_skills=["Python", "推荐系统", "机器学习", "召回排序"],
        nice_skills=["PyTorch", "Flink", "Spark", "A/B测试", "多模态"],
        domains=["推荐", "广告"], education_req="硕士", headcount=1, urgency="high",
        description="负责首页信息流召回与排序模型迭代，优化 CTR/CVR 与时长指标。",
    ),
    JobRequisition(
        job_id="JD-1004", title="前端工程师", city="杭州", level="1-3",
        min_years=2, max_years=4, salary_range=(30000, 45000),
        hard_skills=["Vue", "TypeScript", "前端工程化"],
        nice_skills=["React", "Node", "可视化", "性能优化"],
        domains=["SaaS", "中台"], education_req="本科", headcount=2, urgency="low",
        description="负责数据中台前端应用开发与体验优化，参与组件库建设。",
    ),
    JobRequisition(
        job_id="JD-1005", title="数据平台工程师", city="深圳", level="3-5",
        min_years=3, max_years=7, salary_range=(40000, 65000),
        hard_skills=["Python", "Spark", "Flink", "数据仓库", "SQL"],
        nice_skills=["Kafka", "ClickHouse", "dbt", "Airflow", "Kubernetes"],
        domains=["大数据", "数仓"], education_req="本科", headcount=1, urgency="normal",
        description="负责实时数仓建设与指标平台开发。",
    ),
    JobRequisition(
        job_id="JD-1006", title="大模型应用研发(Agent方向)", city="北京", level="3-5",
        min_years=3, max_years=8, salary_range=(50000, 80000),
        hard_skills=["Python", "Agent", "LLM", "Prompt"],
        nice_skills=["LangGraph", "RAG", "MCP", "多模态", "向量检索"],
        domains=["大模型", "智能体"], education_req="本科", headcount=3, urgency="high",
        description="负责智能体产品核心链路研发：工具编排、记忆管理、多轮任务规划。",
    ),
    JobRequisition(
        job_id="JD-1007", title="DevOps/SRE", city="深圳", level="3-5",
        min_years=3, max_years=6, salary_range=(38000, 58000),
        hard_skills=["Kubernetes", "Docker", "Linux", "CI/CD", "Prometheus"],
        nice_skills=["Terraform", "Istio", "Go", "Ansible"],
        domains=["云原生", "运维"], education_req="本科", headcount=1, urgency="normal",
        description="负责容器平台建设与稳定性保障，完善可观测体系。",
    ),
    JobRequisition(
        job_id="JD-1008", title="校招-算法工程师", city="北京", level="校招",
        min_years=0, max_years=2, salary_range=(25000, 40000),
        hard_skills=["Python", "机器学习"],
        nice_skills=["PyTorch", "推荐系统", "NLP", "论文"],
        domains=["算法"], education_req="硕士", headcount=4, urgency="normal",
        description="2027 届校招算法岗，培养方向为推荐/NLP。",
    ),
]

# ---------------------------------------------------------------- 候选人
CANDIDATES: list[Candidate] = [
    Candidate(
        "C001", "张伟", "AI平台工程师", 6.0, "北京", "硕士", "清华大学", "计算机",
        skills=["Python", "LLM", "RAG", "向量检索", "Milvus", "LangGraph", "FastAPI", "Docker"],
        domains=["大模型", "AI平台"],
        projects=[{"name": "企业知识库问答平台", "role": "技术负责人", "detail":
                   "基于 Milvus + bge 搭建 RAG 链路，支撑 200+ 部门，QPS 峰值 300，回答准确率从 61% 提升至 84%"}],
        job_intent="active", expected_salary=65000, current_salary=52000, availability_days=15,
        stability_years_per_role=3.0, job_changes_3y=1,
        interview={"score": 88, "rounds": 3, "last": "2026-09-28", "reviewer": "李工"},
        highlights=["RAG 落地经验完整（切分/召回/重排/评估）", "主导过 Agent 编排框架落地", "有团队带人经验"],
        risks=[],
        tags=["大厂", "核心项目"],
    ),
    Candidate(
        "C002", "李娜", "大模型应用研发", 4.0, "北京", "硕士", "北京大学", "软件工程",
        skills=["Python", "Agent", "LLM", "Prompt", "LangGraph", "RAG", "向量检索"],
        domains=["大模型", "智能体"],
        projects=[{"name": "客服智能体", "role": "主力开发", "detail":
                   "基于 LangGraph 构建多轮任务 Agent，接入 20+ 工具，替代率 63%"}],
        job_intent="active", expected_salary=60000, current_salary=45000, availability_days=7,
        stability_years_per_role=2.0, job_changes_3y=1,
        interview={"score": 91, "rounds": 3, "last": "2026-09-30", "reviewer": "王工"},
        highlights=["Agent 编排经验与 JD1006 高度吻合", "面试各轮评分均靠前"],
        risks=[],
        tags=["大厂", "高潜力"],
    ),
    Candidate(
        "C003", "王强", "高级后端工程师", 8.0, "上海", "本科", "复旦大学", "软件工程",
        skills=["Java", "分布式", "MySQL", "Redis", "高并发", "Kafka", "SpringCloud", "Elasticsearch"],
        domains=["电商", "金融科技"],
        projects=[{"name": "支付核心链路重构", "role": "架构师", "detail":
                   "重构支付网关，峰值 QPS 从 8000 提升到 42000，P99 延迟下降 61%"}],
        job_intent="passive", expected_salary=78000, current_salary=65000, availability_days=60,
        stability_years_per_role=4.5, job_changes_3y=1,
        interview={"score": 82, "rounds": 2, "last": "2026-09-20", "reviewer": "陈工"},
        highlights=["高并发经验为团队标杆", "架构设计能力强"],
        risks=["被动求职者，需激励", "到岗周期较长"],
        tags=["大厂"],
    ),
    Candidate(
        "C004", "刘洋", "AI平台工程师", 3.5, "上海", "本科", "上海交通大学", "自动化",
        skills=["Python", "LLM", "RAG", "向量检索", "vLLM", "Elasticsearch"],
        domains=["大模型", "搜索"],
        projects=[{"name": "推理服务优化", "role": "开发", "detail":
                   "使用 vLLM + PagedAttention 将推理吞吐提升 3.2 倍"}],
        job_intent="active", expected_salary=48000, current_salary=38000, availability_days=14,
        stability_years_per_role=3.5, job_changes_3y=1,
        interview={"score": 76, "rounds": 2, "last": "2026-09-25", "reviewer": "李工"},
        highlights=["推理优化经验稀缺", "性价比高"],
        risks=["Kubernetes 经验偏浅"],
        tags=[],
    ),
    Candidate(
        "C005", "陈静", "推荐算法工程师", 5.0, "北京", "硕士", "中国科学院大学", "计算机",
        skills=["Python", "推荐系统", "机器学习", "召回排序", "PyTorch", "Flink", "多模态"],
        domains=["推荐", "广告"],
        projects=[{"name": "信息流排序模型", "role": "算法负责人", "detail":
                   "DIN→多目标 MMoE 演进，首页时长 +12%，CVR +8.7%"}],
        job_intent="active", expected_salary=70000, current_salary=55000, availability_days=20,
        stability_years_per_role=2.5, job_changes_3y=1,
        interview={"score": 90, "rounds": 3, "last": "2026-09-29", "reviewer": "赵工"},
        highlights=["推荐全链路经验，指标可验证", "有论文产出"],
        risks=[],
        tags=["大厂", "核心项目"],
    ),
    Candidate(
        "C006", "赵磊", "高级后端工程师", 6.0, "上海", "本科", "同济大学", "计算机",
        skills=["Go", "分布式", "MySQL", "Redis", "高并发", "Kubernetes", "gRPC"],
        domains=["云原生", "金融科技"],
        projects=[{"name": "微服务治理平台", "role": "主力开发", "detail":
                   "服务网格落地，接口 P99 下降 35%"}],
        job_intent="urgent", expected_salary=70000, current_salary=52000, availability_days=7,
        stability_years_per_role=1.8, job_changes_3y=3,
        interview={"score": 74, "rounds": 2, "last": "2026-09-18", "reviewer": "陈工"},
        highlights=["可快速到岗", "云原生落地经验"],
        risks=["3年内3次跳动，稳定性需重点评估", "Java 经验缺失，JD1002 需评估转型成本"],
        tags=[],
    ),
    Candidate(
        "C007", "孙悦", "数据平台工程师", 4.0, "深圳", "硕士", "中山大学", "统计学",
        skills=["Python", "Spark", "Flink", "数据仓库", "SQL", "Kafka", "Airflow", "ClickHouse"],
        domains=["大数据", "数仓"],
        projects=[{"name": "实时指标平台", "role": "技术负责人", "detail":
                   "Flink 实时链路，端到端延迟从分钟级降到秒级，覆盖 800+ 指标"}],
        job_intent="active", expected_salary=52000, current_salary=42000, availability_days=10,
        stability_years_per_role=4.0, job_changes_3y=1,
        interview={"score": 85, "rounds": 3, "last": "2026-09-27", "reviewer": "周工"},
        highlights=["实时数仓端到端经验", "指标体系设计能力"],
        risks=[],
        tags=["大厂"],
    ),
    Candidate(
        "C008", "周涛", "DevOps/SRE", 5.0, "深圳", "本科", "华南理工", "网络工程",
        skills=["Kubernetes", "Docker", "Linux", "CI/CD", "Prometheus", "Terraform", "Go", "Jenkins"],
        domains=["云原生", "运维"],
        projects=[{"name": "容器平台建设", "role": "负责人", "detail":
                   "K8s 多集群管理平台，支撑 800+ 微服务，故障恢复时间下降 70%"}],
        job_intent="active", expected_salary=55000, current_salary=45000, availability_days=15,
        stability_years_per_role=5.0, job_changes_3y=1,
        interview={"score": 86, "rounds": 3, "last": "2026-09-26", "reviewer": "吴工"},
        highlights=["可观测体系从 0 搭建经验", "IaC 熟练"],
        risks=[],
        tags=["大厂"],
    ),
    Candidate(
        "C009", "吴敏", "大模型应用研发", 3.0, "杭州", "硕士", "浙江大学", "人工智能",
        skills=["Python", "Agent", "LLM", "RAG", "Prompt", "多模态", "MCP"],
        domains=["大模型", "智能体"],
        projects=[{"name": "多模态问答助手", "role": "主力开发", "detail":
                   "结合 OCR+多模态模型处理复杂文档问答"}],
        job_intent="active", expected_salary=45000, current_salary=35000, availability_days=10,
        stability_years_per_role=3.0, job_changes_3y=1,
        interview={"score": 79, "rounds": 2, "last": "2026-09-24", "reviewer": "王工"},
        highlights=["多模态方向有差异化", "性价比高"],
        risks=["工程化与上线规模经验偏轻", "LangGraph 未在生产使用"],
        tags=[],
    ),
    Candidate(
        "C010", "郑凯", "AI平台工程师", 7.0, "北京", "硕士", "北京大学", "计算机",
        skills=["Python", "LLM", "RAG", "Agent", "向量检索", "Kubernetes", "Go", "微服务"],
        domains=["大模型", "AI平台"],
        projects=[{"name": "AI 平台中台", "role": "架构师", "detail":
                   "统一模型网关与 Agent 编排，服务 30+ 业务线"}],
        job_intent="passive", expected_salary=82000, current_salary=70000, availability_days=90,
        stability_years_per_role=3.5, job_changes_3y=1,
        interview={"score": 92, "rounds": 3, "last": "2026-09-30", "reviewer": "李工"},
        highlights=["技术深度与广度俱佳", "架构能力突出"],
        risks=["期望薪资超 JD1001 带宽", "被动且到岗周期长"],
        tags=["大厂", "核心项目"],
    ),
    Candidate(
        "C011", "冯雪", "前端工程师", 3.0, "杭州", "本科", "浙江工业大学", "计算机",
        skills=["Vue", "TypeScript", "前端工程化", "React", "Node", "可视化", "性能优化"],
        domains=["SaaS", "中台"],
        projects=[{"name": "数据中台前端", "role": "主力开发", "detail":
                   "可视化大屏 + 低代码搭建器"}],
        job_intent="active", expected_salary=35000, current_salary=28000, availability_days=20,
        stability_years_per_role=3.0, job_changes_3y=1,
        interview={"score": 80, "rounds": 2, "last": "2026-09-22", "reviewer": "钱工"},
        highlights=["与 JD1004 高度匹配", "可视化经验加分"],
        risks=[],
        tags=[],
    ),
    Candidate(
        "C012", "何军", "高级后端工程师", 7.0, "北京", "本科", "北京邮电大学", "通信",
        skills=["Java", "分布式", "MySQL", "Redis", "高并发", "Dubbo", "Elasticsearch"],
        domains=["电商"],
        projects=[{"name": "订单中心", "role": "技术负责人", "detail":
                   "订单中心重构，支撑大促峰值 12 万单/分钟"}],
        job_intent="urgent", expected_salary=68000, current_salary=58000, availability_days=7,
        stability_years_per_role=3.5, job_changes_3y=1,
        interview={"score": 84, "rounds": 2, "last": "2026-09-23", "reviewer": "陈工"},
        highlights=["大促高并发经验扎实", "可快速到岗"],
        risks=["城市与 JD1002(上海)不一致，需确认异地意愿"],
        tags=["大厂"],
    ),
    Candidate(
        "C013", "马晓丽", "推荐算法工程师", 3.0, "北京", "硕士", "北京理工大学", "计算机",
        skills=["Python", "推荐系统", "机器学习", "PyTorch", "召回排序", "NLP"],
        domains=["推荐"],
        projects=[{"name": "短视频召回", "role": "开发", "detail": "多路召回融合优化"}],
        job_intent="active", expected_salary=50000, current_salary=40000, availability_days=15,
        stability_years_per_role=3.0, job_changes_3y=1,
        interview={"score": 75, "rounds": 2, "last": "2026-09-19", "reviewer": "赵工"},
        highlights=["基础扎实", "薪资期望在预算内"],
        risks=["工业级大规模训练经验不足", "无多模态经验"],
        tags=[],
    ),
    Candidate(
        "C014", "高鹏", "数据平台工程师", 6.0, "杭州", "硕士", "武汉大学", "计算机",
        skills=["Java", "Spark", "Flink", "数据仓库", "Hive", "dbt", "Python"],
        domains=["大数据", "数仓"],
        projects=[{"name": "离线数仓分层", "role": "架构师", "detail": "数仓分层重构，任务时长 -40%"}],
        job_intent="passive", expected_salary=60000, current_salary=50000, availability_days=45,
        stability_years_per_role=6.0, job_changes_3y=0,
        interview={"score": 83, "rounds": 2, "last": "2026-09-21", "reviewer": "周工"},
        highlights=["数仓架构经验成熟", "稳定性极佳"],
        risks=["城市与 JD1005(深圳)不一致", "被动"],
        tags=[],
    ),
    Candidate(
        "C015", "林芳", "大模型应用研发", 2.0, "北京", "本科", "北京邮电大学", "信息管理",
        skills=["Python", "LLM", "RAG", "Prompt", "向量检索"],
        domains=["大模型"],
        projects=[{"name": "内部知识助手", "role": "开发", "detail": "基于 Dify 搭建内部知识助手"}],
        job_intent="active", expected_salary=32000, current_salary=25000, availability_days=7,
        stability_years_per_role=2.0, job_changes_3y=1,
        interview={"score": 68, "rounds": 2, "last": "2026-09-17", "reviewer": "王工"},
        highlights=["学习能力强", "到岗快"],
        risks=["经验 2 年，低于 JD1006 门槛 3 年", "无 Agent 生产经验"],
        tags=[],
    ),
    Candidate(
        "C016", "徐峰", "AI平台工程师", 5.0, "北京", "本科", "哈尔滨工业大学", "软件工程",
        skills=["Python", "LLM", "向量检索", "Elasticsearch", "Docker", "FastAPI", "RAG"],
        domains=["AI平台", "搜索"],
        projects=[{"name": "搜索平台 AI 化", "role": "主力开发", "detail": "搜索重排接入 LLM 改写"}],
        job_intent="active", expected_salary=52000, current_salary=44000, availability_days=10,
        stability_years_per_role=5.0, job_changes_3y=1,
        interview={"score": 81, "rounds": 3, "last": "2026-09-28", "reviewer": "李工"},
        highlights=["稳定性好", "搜索+AI 交叉背景"],
        risks=["Agent/K8s 经验薄弱"],
        tags=[],
    ),
    Candidate(
        "C017", "胡斌", "DevOps/SRE", 3.0, "深圳", "本科", "深圳大学", "软件工程",
        skills=["Kubernetes", "Docker", "Linux", "CI/CD", "Prometheus", "Jenkins", "Shell"],
        domains=["运维", "云原生"],
        projects=[{"name": "CI/CD 流水线", "role": "开发", "detail": "流水线自建，发布效率提升 3 倍"}],
        job_intent="active", expected_salary=40000, current_salary=32000, availability_days=20,
        stability_years_per_role=3.0, job_changes_3y=1,
        interview={"score": 72, "rounds": 2, "last": "2026-09-16", "reviewer": "吴工"},
        highlights=["成本低，流水线经验实用"],
        risks=["IaC(Terraform) 缺失", "可观测深度不足"],
        tags=[],
    ),
    Candidate(
        "C018", "苏晴", "推荐算法工程师", 4.0, "上海", "硕士", "复旦大学", "计算机",
        skills=["Python", "推荐系统", "机器学习", "召回排序", "Spark", "A/B测试"],
        domains=["推荐", "电商"],
        projects=[{"name": "商详页推荐", "role": "算法工程师", "detail": "点击率提升 9%"}],
        job_intent="active", expected_salary=58000, current_salary=46000, availability_days=12,
        stability_years_per_role=2.0, job_changes_3y=2,
        interview={"score": 78, "rounds": 2, "last": "2026-09-20", "reviewer": "赵工"},
        highlights=["电商推荐经验", "A/B 实验设计规范"],
        risks=["城市与 JD1003(北京)不一致", "3年内2次跳动"],
        tags=[],
    ),
    Candidate(
        "C019", "唐宇", "高级后端工程师", 5.0, "深圳", "硕士", "哈尔滨工业大学", "软件工程",
        skills=["Go", "分布式", "MySQL", "Redis", "高并发", "gRPC", "DDD"],
        domains=["金融科技", "云原生"],
        projects=[{"name": "清结算系统", "role": "主力开发", "detail": "日终批量处理性能提升 5 倍"}],
        job_intent="active", expected_salary=60000, current_salary=48000, availability_days=15,
        stability_years_per_role=5.0, job_changes_3y=1,
        interview={"score": 87, "rounds": 3, "last": "2026-09-27", "reviewer": "陈工"},
        highlights=["稳定性优秀", "金融级严苛环境经验"],
        risks=["无 Java 经验，JD1002 若硬性要求 Java 需评估"],
        tags=[],
    ),
    Candidate(
        "C020", "曹雪松", "AI平台工程师", 4.5, "上海", "硕士", "同济大学", "电子信息",
        skills=["Python", "LLM", "RAG", "Agent", "Milvus", "LangGraph", "向量检索"],
        domains=["大模型"],
        projects=[{"name": "合同审查 Agent", "role": "主力开发", "detail": "多模态文档解析 + Agent 审查"}],
        job_intent="active", expected_salary=55000, current_salary=43000, availability_days=10,
        stability_years_per_role=2.2, job_changes_3y=2,
        interview={"score": 77, "rounds": 2, "last": "2026-09-23", "reviewer": "李工"},
        highlights=["LangGraph 生产实践", "RAG+Agent 组合经验"],
        risks=["城市与 JD1001(北京)不一致", "跳动略频繁"],
        tags=[],
    ),
    Candidate(
        "C021", "彭飞", "校招-算法工程师", 0.0, "北京", "硕士", "清华大学", "人工智能",
        skills=["Python", "机器学习", "PyTorch", "NLP", "推荐系统"],
        domains=["算法"],
        projects=[{"name": "硕士课题：多模态检索", "role": "研究者", "detail": "顶会论文 1 篇"}],
        job_intent="urgent", expected_salary=32000, current_salary=0, availability_days=30,
        stability_years_per_role=0.0, job_changes_3y=0,
        interview={"score": 84, "rounds": 2, "last": "2026-09-29", "reviewer": "赵工"},
        highlights=["清华在读，论文产出", "符合校招门槛"],
        risks=["无工业界经验", "毕业时间为 2027 年 6 月"],
        tags=["应届"],
    ),
    Candidate(
        "C022", "董洁", "前端工程师", 2.0, "杭州", "本科", "杭州电子科技大学", "软件工程",
        skills=["Vue", "TypeScript", "前端工程化", "React", "Webpack"],
        domains=["SaaS"],
        projects=[{"name": "CRM 前端", "role": "开发", "detail": "CRM 核心模块开发"}],
        job_intent="active", expected_salary=28000, current_salary=23000, availability_days=15,
        stability_years_per_role=2.0, job_changes_3y=1,
        interview={"score": 70, "rounds": 2, "last": "2026-09-15", "reviewer": "钱工"},
        highlights=["Vue 熟练", "到岗快"],
        risks=["缺少组件库建设经验", "可视化能力一般"],
        tags=[],
    ),
    Candidate(
        "C023", "袁磊", "数据平台工程师", 3.0, "北京", "本科", "华北电力大学", "自动化",
        skills=["Python", "Spark", "SQL", "数据仓库", "Hive", "Airflow", "dbt"],
        domains=["大数据"],
        projects=[{"name": "报表平台", "role": "开发", "detail": "报表查询性能优化"}],
        job_intent="passive", expected_salary=38000, current_salary=32000, availability_days=45,
        stability_years_per_role=3.0, job_changes_3y=1,
        interview={"score": 66, "rounds": 1, "last": "2026-09-14", "reviewer": "周工"},
        highlights=["成本低"],
        risks=["无 Flink 实时经验", "被动，激励成本未知"],
        tags=[],
    ),
    Candidate(
        "C024", "范晓东", "大模型应用研发(Agent)", 5.0, "北京", "硕士", "中国科学院大学", "软件工程",
        skills=["Python", "Agent", "LLM", "LangGraph", "RAG", "MCP", "向量检索", "多模态"],
        domains=["智能体", "大模型"],
        projects=[{"name": "运维智能体", "role": "技术负责人", "detail":
                   "基于 LangGraph 的运维故障定位 Agent，故障定位时间 -60%"}],
        job_intent="active", expected_salary=68000, current_salary=56000, availability_days=8,
        stability_years_per_role=2.5, job_changes_3y=1,
        interview={"score": 93, "rounds": 3, "last": "2026-09-30", "reviewer": "王工"},
        highlights=["Agent 方向最匹配人选", "LangGraph+MCP 双栈经验", "面试最高分"],
        risks=["薪资接近 JD1006 带宽上限"],
        tags=["大厂", "核心项目", "高潜力"],
    ),
    Candidate(
        "C025", "崔天佑", "AI平台工程师", 2.0, "北京", "硕士", "北京航空航天大学", "计算机",
        skills=["Python", "LLM", "RAG", "向量检索", "FastAPI"],
        domains=["大模型"],
        projects=[{"name": "毕业设计：RAG 问答", "role": "作者", "detail": "课程项目"}],
        job_intent="active", expected_salary=35000, current_salary=0, availability_days=15,
        stability_years_per_role=0.0, job_changes_3y=0,
        interview={"score": 64, "rounds": 1, "last": "2026-09-12", "reviewer": "李工"},
        highlights=["有 RAG 实际项目"],
        risks=["经验不足 3 年，不满足 JD1001 门槛", "无生产环境经验"],
        tags=["应届"],
    ),
]

# ---------------------------------------------------------------- 知识库文档
KNOWLEDGE_DOCS: list[dict] = [
    {
        "doc_id": "POL-001", "chunk_type": "policy", "title": "面试安排与改期政策",
        "text": "候选人面试改期需提前 24 小时通知 HR。若候选人临时无法参加，HR 应在 2 小时内重新安排，"
                "且一个月内累计改期不超过 2 次；超过 2 次需由招聘经理审批后方可继续安排。"
                "面试开始前 10 分钟，HR 需将会议室与面试官信息通过企业微信发送给候选人。"
                "急招岗位(high urgency)的候选人改期，HR 需同步通知用人部门负责人。",
        "tags": ["面试", "改期", "政策"],
    },
    {
        "doc_id": "POL-002", "chunk_type": "policy", "title": "Offer 审批与薪酬带宽",
        "text": "所有 Offer 需在系统提交审批。薪酬超过岗位带宽上限 110% 的，需 VP 级审批；"
                "在带宽内但处于 P90 分位以上的，需总监级审批。签字费单人不超过 3 个月薪资，"
                "且需在 Offer 中注明发放时间与违约回收条款。期权/股票按公司统一政策执行，不在本次谈判范围。",
        "tags": ["offer", "薪酬", "审批"],
    },
    {
        "doc_id": "POL-003", "chunk_type": "policy", "title": "背景调查规范",
        "text": "候选人接受 Offer 后 5 个工作日内需完成背景调查，核查项包含：任职经历真实性、"
                "薪资真实性、学历学位、离职原因、竞业限制。涉及前雇主商业秘密的信息不得收集。"
                "背调未通过且涉及学历造假的，直接取消 Offer 并报备 HRBP。",
        "tags": ["背调", "合规"],
    },
    {
        "doc_id": "POL-004", "chunk_type": "policy", "title": "招聘流程与时效要求",
        "text": "标准招聘流程为：简历筛选 → 业务面(2-3轮) → HR面 → Offer 审批 → 背调 → 入职。"
                "从简历投递到 Offer 发放的周期目标为 30 天。急招岗位可压缩至 15 天，"
                "但需保证至少 2 轮面试。HR 需在候选人每轮面试后 24 小时内录入面评。",
        "tags": ["流程", "时效"],
    },
    {
        "doc_id": "POL-005", "chunk_type": "policy", "title": "候选人信息使用与合规",
        "text": "候选人简历仅可用于本次招聘目的。招聘结束后 6 个月，未入职候选人的资料需归档或销毁。"
                "严禁将候选人简历上传至外部 AI 工具或公开平台；使用 AI 工具处理候选人信息时，"
                "必须完成脱敏处理。",
        "tags": ["合规", "数据安全"],
    },
    {
        "doc_id": "PRC-001", "chunk_type": "practice", "title": "技术面试评估维度",
        "text": "技术面试按四个维度评估：基础扎实度(占30%)、项目深度(占35%)、"
                "问题解决能力(占25%)、团队协作与表达(占10%)。"
                "对于 AI 方向岗位，基础维度重点考察 Python 能力、Transformer 原理理解、"
                "工程化落地经验；对 RAG 岗位需追问切分策略、召回与重排的权衡逻辑、效果评估方法。",
        "tags": ["面试", "评估", "AI"],
    },
]


def build_db(reset: bool = True) -> Path:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.executescript(SCHEMA)
        edu_rank = {"大专": 1, "本科": 2, "硕士": 3, "博士": 4}
        cands = []
        for c in CANDIDATES:
            cands.append((
                c.candidate_id, c.name, c.title, c.years, c.city, c.education,
                edu_rank.get(c.education, 2), c.school, c.major,
                json.dumps(c.skills, ensure_ascii=False),
                json.dumps(c.domains, ensure_ascii=False),
                json.dumps(c.projects, ensure_ascii=False),
                json.dumps(c.highlights, ensure_ascii=False),
                json.dumps(c.tags, ensure_ascii=False),
                c.job_intent, c.expected_salary, c.current_salary, c.availability_days,
                c.stability_years_per_role, c.job_changes_3y,
                json.dumps(c.interview, ensure_ascii=False),
                json.dumps(c.risks, ensure_ascii=False),
                c.resume_text or build_resume_text(c),
            ))
        con.executemany(
            "INSERT INTO candidates VALUES (" + ",".join("?" * 23) + ")", cands)

        jobs = []
        for j in JOBS:
            jobs.append((
                j.job_id, j.title, j.city, j.level, j.min_years, j.max_years,
                j.salary_range[0], j.salary_range[1],
                json.dumps(j.hard_skills, ensure_ascii=False),
                json.dumps(j.nice_skills, ensure_ascii=False),
                json.dumps(j.domains, ensure_ascii=False),
                j.education_req, j.headcount, j.urgency, j.description,
            ))
        con.executemany("INSERT INTO jobs VALUES (" + ",".join("?" * 15) + ")", jobs)

        con.executemany(
            "INSERT INTO interviews (candidate_id,candidate_name,interviewer,round,"
            "interview_date,start_hour,duration,status) VALUES (?,?,?,?,?,?,?,?)",
            [
                ("C001", "张伟", "技术面试官A", "三面", "2026-10-08", 14, 60, "scheduled"),
                ("C002", "李娜", "技术面试官A", "三面", "2026-10-08", 15, 60, "scheduled"),
                ("C005", "陈静", "技术面试官B", "三面", "2026-10-07", 10, 60, "scheduled"),
                ("C007", "孙悦", "技术面试官B", "二面", "2026-10-09", 11, 60, "scheduled"),
                ("C024", "范晓东", "技术面试官A", "二面", "2026-10-10", 16, 60, "scheduled"),
                ("C011", "冯雪", "前端面试官", "一面", "2026-10-09", 14, 60, "done"),
                ("C017", "胡斌", "技术面试官B", "一面", "2026-10-06", 10, 60, "done"),
            ],
        )
        con.commit()
    return DB_PATH


def build_resume_text(c: Candidate) -> str:
    """从结构化字段生成简历长文本 —— 这是向量检索与 BM25 的主要语料。"""
    proj = "；".join(f"{p['name']}（{p.get('role','')}）：{p.get('detail','')}" for p in c.projects)
    iv = c.interview or {}
    return (
        f"{c.name}，{c.years}年工作经验，{c.city}，{c.education}（{c.school} {c.major}）。"
        f"现任/曾任职位：{c.title}。核心技能：{'、'.join(c.skills)}。"
        f"领域：{'、'.join(c.domains)}。项目经历：{proj}。"
        f"亮点：{'；'.join(c.highlights)}。"
        f"求职意向：{c.job_intent}，期望薪资 {c.expected_salary//1000}K，"
        f"可到岗时间 {c.availability_days} 天。三年内跳动 {c.job_changes_3y} 次。"
        f"面试评分 {iv.get('score', '暂无')}（{iv.get('rounds', 0)} 轮）。"
        f"标签：{'、'.join(c.tags) if c.tags else '无'}。"
    )


KNOWLEDGE_DOCS_EXPORT = KNOWLEDGE_DOCS

if __name__ == "__main__":
    p = build_db()
    with sqlite3.connect(p) as con:
        n1 = con.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        n2 = con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        n3 = con.execute("SELECT COUNT(*) FROM interviews").fetchone()[0]
    print(f"✅ 数据库已初始化: {p}")
    print(f"   候选人 {n1} 位 | 岗位 {n2} 个 | 面试记录 {n3} 条 | 知识文档 {len(KNOWLEDGE_DOCS)} 篇")
