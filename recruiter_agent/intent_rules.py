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
    "candidate_detail": "查看某个候选人的简历、档案、当前进展",
    "job_recommend": "给某位候选人推荐合适的岗位；或求职者描述自身情况找岗位",
    "job_detail": "查看某个岗位的详情、招聘进度、剩余 HC",
    "interview_schedule": "安排/调整面试时间、面试官、会议室",
    "interview_list": "查询已有的面试日程与安排",
    "offer_negotiation": "薪资谈判、offer 审批与对比",
    "market_insight": "人才市场洞察：某技能供需、薪资基准、流动趋势",
    "policy_qa": "招聘政策/流程/制度问答（须走 RAG 知识库）",
    "chitchat": "与招聘无关的闲聊或元问题",
    "confirm_interview": "确认面试时段（回复「确认第 N 个」「选第 N 个」「第 N 个时间」等）",
}

# ---------------------------------------------------------------- 关键词规则
_RULES: list[tuple[str, float, list[str]]] = [
    ("candidate_search", 1.0, ["找人", "找候选人", "推荐几个候选人", "人才检索", "搜简历", "候选人库", "有哪些人",
                               "候选人", "人才"]),
    ("candidate_evaluate", 1.0, ["匹配度", "评估", "适不适合", "能不能做", "是否合适", "打分", "评估一下"]),
    ("candidate_detail", 1.0, ["简历", "什么情况", "档案", "进展", "资料", "发我看看", "详细信息",
                               "在哪个阶段", "什么状态"]),
    ("interview_schedule", 1.0, ["排面试", "约面试", "面试安排", "安排面试", "面试时间", "面试官",
                                 "会议室", "面评", "改到", "一面", "二面", "三面", "初面", "终面",
                                 "安排面", "约个面"]),
    ("interview_list", 1.0, ["有哪些面试", "面试日程", "面试排期", "这周的面试", "下周面", "面试记录",
                             "面试情况", "几场面试", "面试列表"]),
    ("offer_negotiation", 1.0, ["offer", "薪资谈判", "涨薪", "期望薪资", "给多少钱", "薪酬", "定薪", "测算"]),
    ("market_insight", 1.0, ["市场行情", "人才市场", "供需", "薪资基准", "行情如何", "紧缺", "热度",
                             "薪资大概", "什么水平", "能招到", "好找工作", "薪资水平", "薪资范围",
                             "市场价", "多少钱"]),
    ("policy_qa", 1.0, ["政策", "流程", "制度", "规定", "允许吗", "报销", "多久通知", "需要提前",
                        "背调", "背景调查", "竞业", "有效期"]),
    ("job_recommend", 1.0, ["推荐岗位", "适合的岗位", "有什么岗位", "匹配岗位", "适合他",
                            "找工作", "求职", "适合什么", "能做什么", "能推荐岗位", "推荐几个岗位"]),
    ("job_detail", 1.0, ["hc", "招了多久", "岗位详情", "还招吗", "职位详情", "招聘进度",
                         "几个名额", "岗位要求", "岗位介绍", "jd详情"]),
    ("confirm_interview", 1.0, ["确认第", "选第", "第一个时段", "第二个时段", "第三个时段",
                                "选这个时间", "就选", "确认时间", "可以的", "好的第"]),
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

_YEARS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*年")
_YEARS_RANGE = re.compile(r"(\d+(?:\.\d+)?)\s*[-~至到]\s*(\d+(?:\.\d+)?)\s*年")

# 年限的「方向」判定：同一个「3年」可能是下限（5年以上）、上限（3年以下）
# 或中性（3年经验）。早前只靠 _YEARS_NEG 判断，导致「3年以下」被解析成
# min=3、max=3（一个点），「不超过5年」被解析成 min=5、max=7。
_YEARS_MAX_WORDS = ("以下", "以内", "之内", "及以下", "或以下", "封顶")
_YEARS_MIN_WORDS = ("以上", "+", "起")
_YEARS_MAX_PREFIX = ("不超过", "不到", "最多", "至多", "不高于", "不足", "不大于", "少于")
_YEARS_MIN_PREFIX = ("至少", "最少", "不低于", "不少于")

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

    # 面试时段确认序号：「确认第2个」「选第二个」→ _confirm_slot
    _CN_NUM = {"一": 1, "二": 2, "三": 3, "1": 1, "2": 2, "3": 3}
    m_slot = re.search(r"[确认选]第([一二三123])[个号时]", text)
    if m_slot:
        ents["_confirm_slot"] = _CN_NUM.get(m_slot.group(1), 1) - 1  # 0-indexed

    cities = [c for c in _CITY if c in text]
    if cities:
        ents["cities"] = cities

    skills = extract_skills(text)
    if skills:
        ents["skills"] = skills

    # 年限：按「方向」分别收集，而不是把出现的所有数字当成一个区间。
    # 期望语义：
    #   「5年以上」     → min=5（上限不设，交给岗位 JD / 默认值）
    #   「3年以下」     → max=3（下限不设）
    #   「不超过5年」   → max=5
    #   「3-5年」       → min=3, max=5
    #   「3年经验」     → min=3（中性表述按「至少 3 年」处理）
    range_span: tuple[int, int] | None = None
    r = _YEARS_RANGE.search(text)
    if r:
        ents["min_years"] = float(r.group(1))
        ents["max_years"] = float(r.group(2))
        range_span = r.span()

    mins: list[float] = []
    maxs: list[float] = []
    for m in _YEARS_RE.finditer(text):
        if range_span and m.start() < range_span[1] and m.end() > range_span[0]:
            continue  # 已被区间规则消费
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        after = text[m.end():m.end() + 4]
        before = text[max(0, m.start() - 4):m.start()]
        # 否定语境会把方向整体翻转：「不考虑 5 年以上的」= 年限 < 5，
        # 早前按字面判成 min_years=5，推荐出来的人正好是用户明确排除的那批。
        before_wide = text[max(0, m.start() - 10):m.start()]
        negated_clause = bool(re.search(r"不考虑|不要|排除|不限|忽略|不包括|除了", before_wide))
        is_max = any(after.startswith(w) for w in _YEARS_MAX_WORDS) or \
            any(before.endswith(w) for w in _YEARS_MAX_PREFIX)
        is_min = any(after.startswith(w) for w in _YEARS_MIN_WORDS) or \
            any(before.endswith(w) for w in _YEARS_MIN_PREFIX)
        if negated_clause:
            if is_min:          # 不考虑 X 年以上 → 只要 X 年以下
                maxs.append(val)
            elif is_max:        # 不考虑 X 年以下 → 只要 X 年以上
                mins.append(val)
            else:
                maxs.append(val)  # 「不考虑 5 年」（中性）→ 不超过 5 年
        elif is_max:
            maxs.append(val)
        elif is_min:
            mins.append(val)
        else:
            mins.append(val)  # 「3年经验」「3年Java」→ 至少 3 年

    if mins:
        ents["min_years"] = max(mins) if "min_years" not in ents else max(float(ents["min_years"]), max(mins))
    if maxs:
        ents["max_years"] = min(maxs) if "max_years" not in ents else min(float(ents["max_years"]), min(maxs))
    # 上下界自相矛盾（如同时说「5年以上、3年以下」）→ 保留更宽松的一侧，避免得到空集
    if "min_years" in ents and "max_years" in ents and ents["min_years"] > ents["max_years"]:
        ents.pop("max_years")
    # 应届/校招：只有**正面要求**时才收窄到 0~1 年。
    # 「不要应届生」是排除条件，早前无条件命中这一支，把用户要的「有经验的人」
    # 解析成 min=0、max=1，推荐出来正好是应届生。
    if "校招" in text or "应届" in text or "毕业生" in text:
        neg_fresh = bool(re.search(r"(不要|不考虑|排除|不收|不招|不招|别找)[^，。；,;]{0,6}(应届|校招|毕业生)", text))
        if neg_fresh:
            ents["min_years"] = max(float(ents.get("min_years") or 0), 1.0)
            ents.pop("max_years", None)
            ents["_exclude_fresh"] = True
        else:
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
    # 负向后视：「12000 一个月」「3 个月」里的「一/个」不是人数，
    # 否则「12000一个月」会被解析成「找 1 个人」。
    nums = [int(x) for x in re.findall(r"(?<![年月日\d])(\d+)\s*(?:个|名|人)(?![月薪级])", text)]
    # 中文数字：「找三个候选人」「帮我约两个人」——早前只认阿拉伯数字，
    # 导致「找三个」退化成默认 top_n=5，用户要 3 个却给了 5 个。
    cn_digit = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
                "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    for m in re.finditer(r"(?<![年月日])([一二两三四五六七八九十])\s*(?:个|名|位)(?![月薪级])", text):
        nums.append(cn_digit.get(m.group(1), 0))
    nums = [n for n in nums if n > 0]
    if nums:
        ents["top_n"] = max(nums)

    # 薪资：区分「月薪 / 年薪 / 总包」，统一折算为年薪
    # 同时记录「这几个数字是薪资」——姓名抽取要把它们屏蔽掉，
    # 否则「年薪 80 万以内」会把「万以内」当成一个人名（万 是姓氏）。
    annual_vals: list[int] = []
    salary_spans: list[tuple[int, int]] = []
    for m in re.finditer(
        r"(\d+(?:\.\d+)?)\s*([kK万])?\s*(?:块|元)?\s*(?:/|每)?\s*"
        r"(月薪|月|年薪|总包|年薪包|package|一个月|1个月|个月)?\s*"
        r"(以内|以下|之内|及以下|左右|封顶)?",
        text,
    ):
        # 「年」不作为薪资单位：它几乎总是年限（「6年经验」「3到5年」）。
        # 年薪语义靠前面的「年薪/总包/package」判断，见下方 unit 分支。
        # 排除「5万年」「3年」这类纯年份/纯年限写法：万后面紧跟「年」且前面不是薪资语境
        tail = text[m.end():m.end() + 2]
        if not m.group(2) and not m.group(3) and tail.startswith("年"):
            continue
        # 裸数字后紧跟「年」一律是年限，不是薪资（「3到5年」「6年经验」）
        if not m.group(2) and not m.group(3) and tail.startswith("年"):
            continue
        # 带「万/k」但紧跟着「年」时也要看语境：「5万年」是年限，不是 5 万年薪
        if m.group(2) and tail.startswith("年") and not m.group(3):
            continue
        # 没有单位又不是薪资语境（如「找3个」「5年以上」）→ 不是薪资数字
        if not m.group(2) and not m.group(3) and not re.search(
                r"薪|预算|工资|待遇|package|块|元", text[max(0, m.start() - 8):m.end() + 4]):
            continue
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        if m.group(2) == "万":
            val = val * 10000
        elif m.group(2) in ("k", "K"):
            val = val * 1000
        unit = m.group(3) or ""
        # 口径判定（招聘口语）：
        #   「60k」「月薪 2 万」「预算 2 万」「12000 一个月」→ 月薪，×12
        #   「年薪 80 万」「总包 60 万」「package 60 万」→ 已是年值
        #   裸「万」在没写明「年薪/总包」且有「预算/月薪」语境时按月薪理解
        monthly_context = bool(re.search(r"预算|月薪|每月|一个月|/月|块钱一个月", text))
        if unit in ("月薪", "月", "一个月", "1个月", "个月"):
            annual = val * 12
        elif unit in ("年薪", "总包", "年薪包", "package", "年"):
            annual = val
        elif m.group(2) in ("k", "K"):
            annual = val * 12
        elif monthly_context and (not m.group(2)):
            annual = val * 12
        elif monthly_context and m.group(2) == "万":
            annual = val * 12
        else:
            annual = val  # 裸「万」默认年薪
        annual_vals.append(int(annual))
        salary_spans.append(m.span())
    if annual_vals:
        ents["salary_kw"] = annual_vals
        # 「以内 / 以下 / 不超过」是上限；「以上 / 不低于」是下限。
        # 不能在整句上做布尔判断：一句话里可能同时出现年限的「5 年以上」
        # 和薪资的「60k 以内」，布尔判断会让 min_salary = max_salary = 720000，
        # 过滤出空集。改为**在每个薪资数字附近**就近判定方向。
        def _salary_direction(span: tuple[int, int]) -> str:
            s, e = span
            left = text[max(0, s - 6):s]
            right = text[e:e + 6]
            if re.search(r"不超过|最多|至多|不高于|不到|封顶|尽量(不)?超", left) or \
                    re.match(r"\s*(?:以内|以下|之内|及以下|封顶)", right):
                return "cap"
            if re.search(r"不低于|至少|最少|不低于", left) or \
                    re.match(r"\s*(?:以上|起)", right):
                return "floor"
            return "cap"  # 招聘语境里报一个数默认就是预算上限

        dirs = [_salary_direction(sp) for sp in salary_spans]
        if any(d == "cap" for d in dirs):
            ents["max_salary"] = min(v for v, d in zip(annual_vals, dirs) if d == "cap")
        if any(d == "floor" for d in dirs):
            ents["min_salary"] = max(v for v, d in zip(annual_vals, dirs) if d == "floor")

    # ---- JD 编号 ----
    m5 = re.search(r"([A-Z]{2,6}\d{3,6})", text)
    if m5:
        ents["job_id"] = m5.group(1)

    # ---- 候选人姓名 ----
    # 薪资片段先屏蔽：「年薪 80 万以内」的「万以内」会被姓氏启发式误判成人名。
    ents["candidate_name"] = _extract_person_name(_mask_salary(text), skills, ents.get("cities", []))

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
            r"安排|这位|那个|候选|叫做|名叫|和|与|跟|让|叫)")
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
    r"什么岗|哪种岗|什么职位|岗|位|呢|吗|吧|了|和|与|跟|、|，|,|。|；|;|？|\?|！|!|"
    r"这个|那个|该|此|一个|一套|的|（|）|\(|\))$")
# 职级/角色后缀属于岗位名本身，**不能**当噪声剥掉：
# 早前把「工程师」列为噪声，导致「AI平台工程师 的匹配度」被截成「AI平台」，
# 再拿去 get_job 模糊匹配时命中错误的 JD。
_JOB_ROLE_SUFFIX = ("工程师", "开发", "研发", "专家", "架构师", "经理", "总监", "主管",
                    "负责人", "专员", "分析师", "研究员", "科学家", "leader", "Leader")


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
    t = t.replace(" ", "")
    # 疑问/指标类片段不是岗位名：「招了多久」「薪资多少」「还有几个HC」。
    # 这类句子常以「这个岗位……」开头，粗正则会把后面的整段当成岗位名，
    # 再拿去 get_job 模糊匹配就会命中一个毫不相干的 JD。
    if re.search(r"多久|多少|几个|几天|几轮|怎么样|如何|呢|吗|HC|hc", t):
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
    # 疑问词残留在开头时（「上海有哪些后端」→ 剥城市后是「有哪些后端」）再剥一次。
    # 必须在剥城市之后做，否则「有哪些」藏在中间匹配不到。
    t = re.sub(r"^(?:有哪些|有什么|哪些|什么|多少|怎么|如何|是什么|找找|找|看看|查)+", "", t).strip()
    if len(t) < 2:
        return None
    if name and t == name:
        return None
    if t in skills or t in cities:
        return t
    return t or None


def _mask_salary(text: str) -> str:
    """把薪资片段替换成等长占位符，供姓名抽取使用。

    「年薪 80 万以内」里的「万以内」以「万」这个姓氏开头，会被姓氏启发式
    误判成候选人「万以内」；等长替换可以保持其余位置的偏移不变。
    """
    pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*[kK万]\s*(?:/|每)?\s*"
        r"(?:月薪|月|年薪|总包|年薪包|package|年)?\s*"
        r"(?:以内|以下|之内|及以下|左右|封顶)?")
    return pattern.sub(lambda m: "□" * len(m.group(0)), text)


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
    if re.search(r"(推荐|找|筛|拉|挑)\s*(几个|若干|一批|些|点|[0-9一二两三四五六七八九十]+\s*(个|名|位))?"
                 r"[^，。；]{0,25}?(候选人|人选|简历|人|后端|前端|算法|开发|工程师|测试|运维|产品|数据)",
                 text) and not re.search(r"找工作|求职", text):
        scores["candidate_search"] = scores.get("candidate_search", 0) + 1.6
        reasons["candidate_search"] = reasons.get("candidate_search", "") + "；含检索数量+对象表述"
        # 「推荐 3 个能马上安排面试的后端」里的「安排面试」是修饰语，不是排期动作；
        # 只要句子里有「推荐/找 + 数量」，就以人才检索为准，压掉排期。
        if re.search(r"(推荐|找|筛|拉|挑)\s*[0-9一二两三四五六七八九十]+\s*(个|名|位)", text):
            if "interview_schedule" in scores:
                scores["interview_schedule"] *= 0.25

    # 院校筛选（985/211/具体学校）= 人才检索的口语说法
    if re.search(r"985|211|双一流|毕业的|统招", text) and re.search(r"找|推荐|有没有|招", text):
        scores["candidate_search"] = scores.get("candidate_search", 0) + 1.5
        reasons["candidate_search"] = reasons.get("candidate_search", "") + "；含院校筛选表述"

    # 面试日程查询：「下周/明天/这周有哪几场面试」——问的是已有安排，不是要排新的
    if re.search(r"(下周|这周|本周|明天|今天|最近)[^，。；]{0,8}面试", text) or \
            re.search(r"有几(个|场)面试|面试日程|面试安排有哪些|面试列表", text):
        scores["interview_list"] = scores.get("interview_list", 0) + 2.2
        reasons["interview_list"] = "面试日程查询表述"
        for competing in ("interview_schedule", "candidate_search"):
            if competing in scores:
                scores[competing] *= 0.3

    # 求职者视角：「找工作」「我想找 XX 的工作」「我适合什么」= 让他看岗位，不是替他找人
    if re.search(r"找工作|求职|找.{0,6}的工作|我想找|我适合|我能做|能推荐岗位", text):
        scores["job_recommend"] = scores.get("job_recommend", 0) + 2.4
        reasons["job_recommend"] = "求职者视角（找工作/我适合）"
        for competing in ("candidate_search", "candidate_evaluate"):
            if competing in scores:
                scores[competing] *= 0.2

    # 「这个岗位薪资多少」= 岗位详情，不是制度问答
    if re.search(r"(这个|该|此)?岗位[^，。；]{0,6}(薪资|工资|待遇|多少钱|要求|职责)", text):
        scores["job_detail"] = scores.get("job_detail", 0) + 2.2
        reasons["job_detail"] = "岗位属性查询"
        for competing in ("chitchat", "policy_qa"):
            if competing in scores:
                scores[competing] *= 0.2

    # 面试确认：「确认第 N 个」「选第 N 个时段」→ confirm_interview，压制闲聊
    m_confirm = re.search(r"确认|选第[一二三1-3]|好的.*时段|可以.*时间", text)
    if m_confirm:
        scores["confirm_interview"] = scores.get("confirm_interview", 0) + 2.5
        reasons["confirm_interview"] = "面试时段确认表述"
        for competing in ("chitchat", "policy_qa"):
            if competing in scores:
                scores[competing] *= 0.1

    if re.search(r"[一二三]面|初面|终面|安排.{0,3}面|约.{0,3}面", text):
        scores["interview_schedule"] = scores.get("interview_schedule", 0) + 1.4
        reasons["interview_schedule"] = reasons.get("interview_schedule", "") + "；含面试轮次/排期表述"

    # 明确的人才检索动作覆盖排期：。「推荐 3 个能马上安排面试的后端」里
    # 「安排面试」只是对候选人的修饰语，真正动作是「推荐 N 个候选��」。
    _explicit_search = re.search(
        r"(推荐|找|筛|拉|挑)\s*[0-9一二两三四五六七八九十]+\s*(个|名|位)[^，。；]{0,10}"
        r"(候选人|人选|后端|前端|算法|开发|工程师|测试|运维|产品|数据|简历)", text)
    if _explicit_search:
        scores["candidate_search"] = scores.get("candidate_search", 0) + 2.0
        reasons["candidate_search"] = reasons.get("candidate_search", "") + "；显式「推荐 N 个对象」"
        if "interview_schedule" in scores:
            scores["interview_schedule"] *= 0.3

    # 市场行情：问「好不好找 / 薪资水平 / 能招到什么水平」——关注的是市场而非具体人
    if re.search(r"好找工作|好不好找|薪资(大概|水平|范围|多少)|什么水平|能招到|市场价|"
                 r"多少钱能|行情", text):
        scores["market_insight"] = scores.get("market_insight", 0) + 2.2
        reasons["market_insight"] = "市场行情/薪资水平表述"
        for competing in ("candidate_search", "job_recommend", "chitchat"):
            if competing in scores:
                scores[competing] *= 0.3

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
