"""意图识别规则库。

这一层同时承担两个角色：
1. Mock LLM 模式下的确定性意图识别；
2. 真实 LLM 模式下的 few-shot 样例与「规则预筛」（先缩小候选意图集，再让 LLM 在
   候选集内做判别，可显著降低幻觉与延迟）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------- 意图定义
INTENTS: dict[str, str] = {
    "candidate_search": "按岗位/技能/城市/经验搜索候选人（人才检索）",
    "candidate_evaluate": "评估某个（或某几个）候选人与岗位的匹配度",
    "job_recommend": "给某位候选人推荐合适的岗位",
    "interview_schedule": "安排/调整面试时间、面试官、会议室",
    "offer_negotiation": "薪资谈判、offer 审批与对比",
    "market_insight": "人才市场洞察：某技能供需、薪资基准、流动趋势",
    "policy_qa": "招聘政策/流程/制度问答（须走 RAG 知识库）",
    "chitchat": "与招聘无关的闲聊或元问题",
}

# ---------------------------------------------------------------- 关键词规则
_RULES: list[tuple[str, float, list[str]]] = [
    ("candidate_search", 1.0, ["找人", "找候选人", "推荐几个候选人", "人才检索", "搜简历", "候选人库", "有哪些人"]),
    ("candidate_evaluate", 1.0, ["匹配度", "评估", "适不适合", "能不能做", "是否合适", "打分", "评估一下"]),
    ("interview_schedule", 1.0, ["排面试", "约面试", "面试安排", "安排面试", "面试时间", "面试官",
                                 "会议室", "面评", "改到", "一面", "二面", "三面", "初面", "终面",
                                 "安排面", "约个面"]),
    ("offer_negotiation", 1.0, ["offer", "薪资谈判", "涨薪", "期望薪资", "给多少钱", "薪酬", "定薪", "测算"]),
    ("market_insight", 1.0, ["市场行情", "人才市场", "供需", "薪资基准", "行情如何", "紧缺", "热度"]),
    ("policy_qa", 1.0, ["政策", "流程", "制度", "规定", "允许吗", "报销", "多久通知", "需要提前",
                        "背调", "背景调查", "竞业", "有效期"]),
    ("job_recommend", 1.0, ["推荐岗位", "适合的岗位", "有什么岗位", "匹配岗位", "适合他"]),
]

# 制度性提问标记：出现这些措辞时，即使命中动作类关键词也应归为 policy_qa
_POLICY_QUESTION = re.compile(
    r"(需要提前|提前多久|多久通知|要多久|几天|是否需要|能不能|可不可以|是否允许|规定是|"
    r"流程是|制度是|有什么要求|需要什么材料|怎么办理|如何办理|标准是|有什么规定)")


def _dampen_by_policy_question(text: str, scores: dict[str, float]) -> dict[str, float]:
    """疑问句式 + 制度性措辞 → 压制动作类意图，抬升 policy_qa。"""
    if not _POLICY_QUESTION.search(text or ""):
        return scores
    for act in ("interview_schedule", "offer_negotiation", "candidate_evaluate"):
        if act in scores:
            scores[act] *= 0.4
    scores["policy_qa"] = scores.get("policy_qa", 0) + 1.8
    return scores

_QUESTION_HINTS = ["?", "？", "怎么", "如何", "什么", "哪些", "为什么", "是否", "能否"]
_CHITCHAT = ["你好", "在吗", "hi", "hello", "谢谢", "你是谁", "自我介绍"]

# 岗位推荐判别：岗位名词 + 「推荐/适合/能投」类动词
_JOB_NOUN = re.compile(r"岗位|职位|职缺|机会|坑位|opening")
_JOB_REC_VERB = re.compile(r"推荐|适合|匹配|能投|可投|投递|合适|找个?合适|有哪些|有什么")


# ---------------------------------------------------------------- 实体抽取
_CITY = ["北京", "上海", "深圳", "广州", "杭州", "成都", "南京", "武汉", "西安", "苏州",
         "合肥", "长沙", "青岛", "厦门", "天津", "重庆", "东莞", "佛山", "宁波", "郑州"]

_SKILL_DICT: dict[str, list[str]] = {
    "语言": ["Python", "Java", "Go", "Golang", "C++", "Rust", "Scala", "Kotlin", "TypeScript",
             "JavaScript", "SQL", "Shell", "Rust", "PHP", "C#"],
    "AI/算法": ["LLM", "RAG", "Agent", "大模型", "大语言模型", "智能体", "Transformer", "推荐系统",
                "NLP", "CV", "机器学习", "深度学习", "PyTorch", "图神经网络", "多模态", "向量检索",
                "Embedding", "Prompt", "微调", "知识图谱", "语音识别", "MCP", "LangGraph",
                "vLLM", "AIGC", "思维链", "多智能体"],
    "数据/数仓": ["Hadoop", "Spark", "Flink", "Hive", "Kafka", "ClickHouse", "Doris",
                  "数据仓库", "数仓", "dbt", "Airflow", "数据治理", "实时计算"],
    "云原生/运维": ["Kubernetes", "K8s", "Docker", "Terraform", "CI/CD", "Jenkins",
                    "Prometheus", "Grafana", "Linux", "Istio", "SRE", "DevOps", "Ansible"],
    "后端/架构": ["微服务", "分布式", "高并发", "Spring", "SpringCloud", "MySQL", "Redis",
                  "Nginx", "gRPC", "DDD", "架构设计", "服务端", "性能优化", "Elasticsearch",
                  "Dubbo", "消息队列"],
    "前端": ["Vue", "React", "Angular", "Webpack", "前端工程化", "小程序", "TypeScript", "Node"],
    "移动端": ["iOS", "Android", "Flutter", "React Native", "Kotlin", "HarmonyOS"],
    "测试/质量": ["自动化测试", "Pytest", "JMeter", "性能测试", "CI/CD", "测试平台"],
    "产品/设计": ["需求分析", "PRD", "原型", "Figma", "用户增长", "数据驱动", "项目管理"],
    "搜索": ["搜索", "检索", "搜索引擎", "召回排序", "搜索算法"],
}

_YEARS_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*年"
    r"(?:以上|以下|\+|起|左右|工作经验|相关|开发|经验|经历)?"
)
_YEARS_NEG = re.compile(r"(\d+(?:\.\d+)?)\s*年(?:以下|以内|之内|以下)")
_YEARS_RANGE = re.compile(r"(\d+)\s*[-~到至]\s*(\d+)\s*年")

# 候选人姓名的上下文噪声词：这些词常出现在「…方向的候选人」这类短语里
_NAME_STOP = {
    "方向的", "方向的候", "以上", "以下", "相关", "有经", "经验的", "水平的", " senior",
    "以上年", "级别", "相关经", "的候", "资深", "初级", "中级",
}


def extract_skills(text: str) -> list[str]:
    low = text.lower()
    hits: list[str] = []
    for _, group in _SKILL_DICT.items():
        for skill in group:
            if skill.lower() in low:
                hits.append(skill)
    # 英文技能大小写归一（匹配到的原样保留）
    return list(dict.fromkeys(hits))


def extract_entities(text: str) -> dict[str, Any]:
    """从自然语言抽取招聘场景实体。"""
    ents: dict[str, Any] = {}

    cities = [c for c in _CITY if c in text]
    if cities:
        ents["cities"] = cities

    skills = extract_skills(text)
    if skills:
        ents["skills"] = skills

    years: list[float] = []
    for m in _YEARS_RE.finditer(text):
        try:
            years.append(float(m.group(1)))
        except ValueError:
            continue
    r = _YEARS_RANGE.search(text)
    if r:
        years.extend([float(r.group(1)), float(r.group(2))])
    if years:
        ents["min_years"] = min(years)
        # 只有出现「以下/不超过」时才设上限
        if _YEARS_NEG.search(text) or re.search(r"\d+\s*[-~到至]\s*\d+\s*年", text):
            ents["max_years"] = max(years)
        else:
            neg = [float(m.group(1)) for m in _YEARS_NEG.finditer(text)]
            if neg:
                ents["max_years"] = min(neg)
            else:
                ents["max_years"] = max(years) + 2.0
    if "校招" in text or "应届" in text or "毕业生" in text:
        ents["min_years"], ents["max_years"] = 0.0, 1.0
        ents["education"] = "本科"

    if "本科" in text:
        ents.setdefault("education", "本科")
    if "硕士" in text or "研究生" in text:
        ents["education"] = "硕士"
    if "博士" in text:
        ents["education"] = "博士"
    if "大厂" in text or "知名" in text:
        ents["tags"] = ["大厂"]
    if "活跃" in text or "正在看机会" in text or "在看机会" in text:
        ents["job_intent"] = "active"
    if "被动" in text or "没在看" in text:
        ents["job_intent"] = "passive"

    # 数字 -> 人数 / 薪资
    nums = [int(x) for x in re.findall(r"(\d+)\s*(?:个|名|人)", text)]
    if nums:
        ents["top_n"] = max(nums)

    # 薪资：区分「月薪 / 年薪 / 总包」，统一折算为年薪
    annual_vals: list[int] = []
    for m in re.finditer(
        r"(\d+(?:\.\d+)?)\s*([kK万])\s*(?:/|每)?\s*(月薪|年薪|总包|年薪包|package|年|个月薪)?",
        text,
    ):
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        val = val * 10000 if m.group(2) == "万" else val * 1000
        unit = m.group(3) or ""
        if "月" in unit:
            annual = val * 12
        else:  # 年薪 / 总包 / 裸数字，一律按「已是年值」处理
            annual = val
        annual_vals.append(int(annual))
    if annual_vals:
        ents["salary_kw"] = annual_vals

    # ---- JD 编号 ----
    m5 = re.search(r"([A-Z]{2,6}\d{3,6})", text)
    if m5:
        ents["job_id"] = m5.group(1)

    # ---- 候选人姓名 ----
    ents["candidate_name"] = _extract_person_name(text, skills, ents.get("cities", []))

    # ---- 岗位名称 ----
    ents["job_title"] = _extract_job_title(text, skills, ents.get("cities", []),
                                          ents.get("candidate_name"))

    # ---- 市场分析主题 ----
    topic = _extract_market_topic(text, skills, ents.get("cities", []))
    if topic:
        ents["_market_topic"] = topic

    return ents


# ---------------------------------------------------------------- 姓名抽取
_SURNAMES = (
    "王李张刘陈杨黄赵周吴徐孙马朱胡林郭何高罗郑梁谢宋唐许韩冯邓曹彭曾萧田董袁潘于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤"
)


# 姓氏 + 单字构成的常见动词/介词，不是人名
_NOT_NAMES = {
    "向候", "向人", "找人", "给人", "有人", "用人", "求人", "加人", "选人",
    "推人", "荐人", "来人", "去人", "出人", "收人", "代人", "专人", "个人",
    "公司", "同行", "各位", "大家", "本人", "别人", "他人", "多人", "几人",
    "方向", "匹配", "经验", "简历", "背景", "评估", "推荐", "岗位", "面试",
    "薪资", "以上", "以下", "测算", "行情", "分析", "市场", "该名", "这名",
}


def _valid_name(name: str | None) -> bool:
    if not name or not (2 <= len(name) <= 4):
        return False
    if name[0] not in _SURNAMES:
        return False
    if name in _NOT_NAMES:
        return False
    return not any(w in name for w in ("方向", "匹配", "经验", "简历", "背景", "评估",
                                       "推荐", "岗位", "面试", "薪资", "以上", "以下",
                                       "测算", "行情", "分析", "市场"))


def _extract_person_name(text: str, skills: list[str], cities: list[str]) -> str | None:
    """抽取候选人姓名：显式标记优先，其次姓氏 + 边界启发式。"""
    # 显式标记。用**候选切分 + 校验**而非贪婪正则：
    # 「候选人张伟测算」贪婪会吞成「张伟测算」，切分后再逐个校验才能取到「张伟」。
    for pat in (r"候选人[：:\s]*([一-龥]{2,5})",
                r"([一-龥]{2,5})[这]?位?候选人",
                r"([一-龥]{2,5})同学"):
        for m in re.finditer(pat, text):
            for cand in _name_candidates(m.group(1)):
                if _valid_name(cand):
                    return cand

    # 姓氏 + 1~3 字（惰性），后接边界词。
    # 用惰性量词是关键：贪婪会把「张伟和」整体当成名字。
    # 另外要求**前面有明确的「主语引入语」**（句首 / 给 / 评估 / 推荐 / 这位 等），
    # 否则「…方向的候选人」这类短语会产出「向候选人」这种伪姓名。
    tail = (r"(?=[的和与跟，,。；;？?！!\s]|评估|推荐|匹配|适合|简历|情况|背景|经历|表现|"
            r"能|是否|可以|测算|安排|约|看|一面|二面|三面|面试|明|今|后|下|上|周|明天|今天|"
            r"下一|这一|$)")
    lead = (r"(?:^|[，,。；;？?！!\s]|给|为|帮|请|评估|评价|分析|推荐|看看|查|找|联系|约|"
            r"安排|这位|那个|候选|叫做|名叫)")
    pattern = rf"{lead}\s*([{_SURNAMES}][一-龥]{{1,3}}?){tail}"
    for m in re.finditer(pattern, text):
        cand = m.group(1)
        if _valid_name(cand) and not _overlaps(cand, skills, cities):
            return cand
    return None


def _name_candidates(raw: str) -> list[str]:
    """把「张伟测算」这类粘连串切成可能的姓名（按长度从短到长）。

    以姓氏开头 + 逐步延长，因此「张伟测算」会依次产出 张伟 / 张伟测 / 张伟测算，
    由 _valid_name 决定取哪个。
    """
    if not raw or raw[0] not in _SURNAMES:
        return []
    return [raw[:n] for n in range(2, min(len(raw), 4) + 1)]


def _overlaps(name: str, skills: list[str], cities: list[str]) -> bool:
    return any(name in kw for kw in list(skills) + list(cities))


# ---------------------------------------------------------------- 岗位名抽取
_JOB_NOISE_TAIL = re.compile(
    r"(的匹配度|匹配度|怎么样|如何|的?情况|的?表现|合适|岗位|职位|方向|职责|要求|薪资|"
    r"工程师|开发|专家|架构师|师|岗|位|呢|吗|吧|了|和|与|跟|、|，|,|。|；|;|？|\?|！|!|"
    r"这个|那个|该|此|一个|一套|的)")


def _extract_job_title(text: str, skills: list[str], cities: list[str], name: str | None) -> str | None:
    # 先剥离括号补充说明：「AI平台工程师(LLM/RAG方向)」→「AI平台工程师」
    text_wo_paren = re.sub(r"[（(][^）)]*[）)]", " ", text)

    for m in re.finditer(r"(?:岗位|职位|职位名称|招聘|JD)\s*[:：]?\s*([A-Za-z\u4e00-\u9fff+#./\-_ ]{2,25})", text):
        t = _clean_job_title(m.group(1), skills, cities, name)
        if t:
            return t

    # 「和/与 <岗位名> 岗位/职位」以及「和/与 <岗位名> 的匹配度」
    for m in re.finditer(
        r"[和与跟]\s*([A-Za-z\u4e00-\u9fff+#./\-_]{2,20}?)\s*(?:岗位|职位|这个|该|的匹配度|的匹配|匹配度)",
        text_wo_paren,
    ):
        t = _clean_job_title(m.group(1), skills, cities, name)
        if t and len(t) >= 2:
            return t

    for m in re.finditer(r"([A-Za-z\u4e00-\u9fff+#]{2,15}?)方向的", text_wo_paren):
        t = _clean_job_title(m.group(1), skills, cities, name)
        if t:
            return t
    return None


def _clean_job_title(raw: str, skills: list[str], cities: list[str], name: str | None) -> str | None:
    t = (raw or "").strip()
    if not t:
        return None
    for _ in range(4):
        t2 = _JOB_NOISE_TAIL.sub("", t).strip()
        if t2 == t:
            break
        t = t2
    t = t.strip(" -_/·、，,。；;：: ")
    for c in cities:
        if t.startswith(c) and len(t) > len(c):
            t = t[len(c):]
            break
    if len(t) < 2:
        return None
    if name and t == name:
        return None
    if t in skills or t in cities:
        return t
    return t or None


# ---------------------------------------------------------------- 市场主题
def _extract_market_topic(text: str, skills: list[str], cities: list[str]) -> str | None:
    if not re.search(r"市场|行情|供需|紧缺|热度", text):
        return None
    cleaned = text
    for c in cities:
        cleaned = cleaned.replace(c, "", 1)
    for pat in (r"([\u4e00-\u9fffA-Za-z+#]{2,10}?)方向的?(?:人才)?(?:市场|行情|供需)",
                r"([\u4e00-\u9fffA-Za-z+#]{2,10}?)(?:市场|行情)(?:怎么样|如何|如何样)?",
                r"(?:分析|看看|查看|查)\s*([\u4e00-\u9fffA-Za-z+#]{2,10}?)(?:方向|领域|的)?(?:市场|行情|供需|情况)"):
        m = re.search(pat, cleaned)
        if m:
            t = _JOB_NOISE_TAIL.sub("", m.group(1).strip("的了和与跟 ")).strip()
            if t and len(t) >= 2 and t not in ("人才", "技术", "招聘"):
                return t
    return skills[0] if skills else None


# ---------------------------------------------------------------- 规则意图识别
def rule_based_intent(text: str) -> dict[str, Any]:
    """返回 {"intent":..., "confidence":..., "entities":{...}, "reason":...}"""
    low = (text or "").lower().strip()
    if not low:
        return {"intent": "chitchat", "confidence": 1.0, "entities": {}, "reason": "空输入"}

    entities = extract_entities(text)

    # 闲聊优先（短句 + 招呼词）
    if len(low) <= 8 and any(k in low for k in _CHITCHAT):
        return {"intent": "chitchat", "confidence": 0.9, "entities": entities,
                "reason": "命中闲聊词且为短句"}

    scores: dict[str, float] = {}
    reasons: dict[str, str] = {}
    for intent, w, kws in _RULES:
        hits = [k for k in kws if k.lower() in low]
        if hits:
            scores[intent] = scores.get(intent, 0) + w * len(hits) / max(len(kws), 1) * len(kws)
            reasons[intent] = f"命中关键词 {hits}"

    scores = _dampen_by_policy_question(text, scores)

    # 语义线索加权：出现技能/城市/经验 = 更像检索
    if entities.get("skills") or entities.get("cities") or "min_years" in entities:
        scores["candidate_search"] = scores.get("candidate_search", 0) + 0.9
        reasons["candidate_search"] = reasons.get("candidate_search", "") + "；含技能/城市/经验实体"
    # 「推荐几个/几个/若干 + Nouns + 候选人」= 人才检索
    if re.search(r"(推荐|找|筛|拉)\s*(几个|若干|一批|些|点|[0-9一二两三四五六七八九十]+\s*(个|名|位))?"
                 r"[^，。；]{0,25}?(候选人|人选|简历|人)", text):
        scores["candidate_search"] = scores.get("candidate_search", 0) + 1.6
        reasons["candidate_search"] = reasons.get("candidate_search", "") + "；含检索数量+候选人表述"

    if re.search(r"[一二三]面|初面|终面|安排.{0,3}面|约.{0,3}面", text):
        scores["interview_schedule"] = scores.get("interview_schedule", 0) + 1.4
        reasons["interview_schedule"] = reasons.get("interview_schedule", "") + "；含面试轮次/排期表述"

    # 岗位推荐：问「给某人推荐 / 某人适合哪些岗位」。
    # 与 candidate_evaluate（评估与**某个具体岗位**的匹配度）区分：
    #   · 出现「匹配度」→ 是评估，不是推荐，直接跳过；
    #   · 寻找目标是「候选人/人选/简历」→ 是人才检索，不是岗位推荐。
    if (_JOB_NOUN.search(text) and _JOB_REC_VERB.search(text)
            and "匹配度" not in text and not re.search(r"候选人|人选|简历", text)):
        scores["job_recommend"] = scores.get("job_recommend", 0) + 1.8
        reasons["job_recommend"] = "寻找适合某候选人的岗位（岗位名词+推荐/适合动词）"
        for competing in ("candidate_evaluate", "policy_qa", "chitchat", "candidate_search"):
            if competing in scores:
                scores[competing] *= 0.3
    if any(k in low for k in ["推荐候选人", "帮我找", "来几个", "人选"]):
        scores["candidate_search"] = scores.get("candidate_search", 0) + 1.2
    if entities.get("candidate_name") and any(k in low for k in ["评估", "匹配", "适合"]):
        scores["candidate_evaluate"] = scores.get("candidate_evaluate", 0) + 1.5
    if any(q in text for q in _QUESTION_HINTS) and not scores:
        scores["policy_qa"] = 0.7
        reasons["policy_qa"] = "疑问句式且无明确业务动作"

    if not scores:
        scores["chitchat"] = 0.5
        reasons["chitchat"] = "无业务动作匹配"

    best = max(scores, key=lambda k: scores[k])
    total = sum(scores.values()) or 1.0
    confidence = round(min(0.98, scores[best] / total * 0.5 + min(scores[best], 1.5) / 1.5 * 0.5), 2)
    return {
        "intent": best,
        "confidence": confidence,
        "entities": entities,
        "reason": reasons.get(best, "默认"),
        "scores": {k: round(v, 2) for k, v in scores.items()},
    }


def candidate_intents(text: str) -> list[str]:
    """给 LLM 的 few-shot：规则先缩小候选意图集。"""
    r = rule_based_intent(text)
    ranked = sorted(r.get("scores", {}), key=lambda k: r["scores"][k], reverse=True)
    return ranked[:3] or [r["intent"]]


INTENT_PROMPT = """你是企业招聘域的意图识别器。任务：把招聘专家的自然语言需求映射到以下意图之一。

意图列表：
{intent_list}

规则：
1. 只能输出 JSON，格式：{{"intent": "<意图>", "confidence": 0.0-1.0, "reason": "<20字内理由>", "entities": {{"skills": [], "cities": [], "min_years": 0, "max_years": 0, "top_n": 0, "job_title": "", "candidate_name": "", "job_id": "", "tags": [], "education": ""}}}}
2. entities 只填从原文能直接确认的字段，不要臆测。
3. 用户明确说"忽略/不考虑某条件"时，在 entities 顶层加 "negated": [条件名]。

示例：
用户：帮我找 5 个 北京 3年以上 LLM/RAG 方向的候选人
输出：{{"intent":"candidate_search","confidence":0.97,"reason":"索取候选人数+筛选条件","entities":{{"skills":["LLM","RAG"],"cities":["北京"],"min_years":3,"top_n":5}}}}

用户：张伟的简历和后端岗位匹配度怎么样
输出：{{"intent":"candidate_evaluate","confidence":0.95,"reason":"评估指定候选人","entities":{{"candidate_name":"张伟","job_title":"后端"}}}}

用户：面试改到明天下午三点
输出：{{"intent":"interview_schedule","confidence":0.96,"reason":"调整面试时间","entities":{{}}}}
"""
