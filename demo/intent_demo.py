#!/usr/bin/env python
"""意图识别教学 demo —— 以 AI 招聘 Agent 为例。

运行：
    ./venv/bin/python demo/intent_demo.py            # 全部阶段
    ./venv/bin/python demo/intent_demo.py --stage 3  # 只看某一阶段

七个阶段，层层递进：
    1. 关键词规则基线（以及它必然错在哪里）
    2. 槽位抽取：意图对了，参数错了，照样是错的答案
    3. 字符 n-gram 向量兜底（处理没见过说法的同义改写）
    4. LLM 通道：结构化输出 + 白名单校验 + 合流策略
    5. 置信度与澄清阈值
    6. 多轮指代：靠会话把上一轮的槽位带过来
    7. 评测：分意图 P/R/F1 + 混淆矩阵

零依赖、零 API Key：全部可在本地跑通，输出是确定性的。
"""
from __future__ import annotations

import argparse
import hashlib
import math
import re
from collections import defaultdict

# ==================================================================== 数据
# 25 条带标注样本（调参集）：意图 + 关键槽位。写法有意贴近真实招聘口语。
CASES: list[tuple[str, str, dict]] = [
    # 人才检索：条件 + 数量
    ("找 5 个北京 3 年以上 LLM 方向的候选人", "candidate_search",
     {"cities": ["北京"], "min_years": 3.0, "top_n": 5, "skills_any": ["LLM", "RAG"]}),
    ("帮我招个 Java 后端，上海，3 到 5 年", "candidate_search",
     {"cities": ["上海"], "min_years": 3.0, "max_years": 5.0, "skills_any": ["Java"]}),
    ("北京和上海都可以，做 LLM 的，5 年往上", "candidate_search",
     {"cities": ["北京", "上海"], "min_years": 5.0, "skills_any": ["LLM"]}),
    ("上海有哪些后端方向的候选人", "candidate_search",
     {"cities": ["上海"], "direction": "后端"}),
    ("有没有便宜点的候选人，预算 2 万以内", "candidate_search",
     {"max_salary": 240000, "monthly": True}),
    ("不要应届生，要能马上到岗的", "candidate_search",
     {"min_years": 1.0, "availability_max_days": 15}),
    ("找几个 985 毕业的算法", "candidate_search",
     {"school_tier": "985", "direction": "算法"}),
    # 修饰语陷阱：真正的动作是「推荐 N 个」，不是「安排面试」
    ("给我推荐 3 个能马上安排面试的后端", "candidate_search",
     {"top_n": 3, "direction": "后端"}),
    # 候选人详情
    ("帮我把张伟的简历发我看看", "candidate_detail", {"candidate_name": "张伟"}),
    ("张伟现在什么情况", "candidate_detail", {"candidate_name": "张伟"}),
    # 岗位详情
    ("这个岗位招了多久了，还有几个 HC", "job_detail", {}),
    # 匹配度评估
    ("评估张伟和 AI平台工程师 的匹配度", "candidate_evaluate",
     {"candidate_name": "张伟", "job_title": "AI平台工程师"}),
    # 求职者视角：给候选人推荐岗位 / 自己找岗位
    ("张伟适合什么岗位", "job_recommend", {"candidate_name": "张伟"}),
    ("我 6 年经验在北京，期望 40k，有什么合适的岗位", "job_recommend",
     {"cities": ["北京"], "min_years": 6.0, "seeker": True}),
    ("我适合什么岗位", "job_recommend", {"seeker": True}),
    ("给我推荐北京月薪 5 万以上的岗位", "job_recommend",
     {"cities": ["北京"], "min_salary": 600000}),
    # 面试
    ("帮我和张伟约个面试", "interview_schedule", {"candidate_name": "张伟"}),
    ("确认第二个时段", "confirm_interview", {"confirm_slot": 1}),
    ("下周能面试的有几个候选人", "interview_list", {}),
    # offer
    ("给张伟测算 offer，总包 60 万", "offer_negotiation",
     {"candidate_name": "张伟", "expected_salary": 600000}),
    # 市场
    ("北京 Java 后端薪资大概多少", "market_insight",
     {"cities": ["北京"], "skills_any": ["Java"]}),
    ("前端现在好找工作吗", "market_insight", {"direction": "前端"}),
    ("12000 一个月能招到什么水平的前端", "market_insight",
     {"max_salary": 144000, "direction": "前端"}),
    # 制度
    ("面试改期需要提前多久通知 HR", "policy_qa", {}),
    ("你好", "chitchat", {}),
]

# 留出集：写完规则后才补的 15 条，**从未用来调规则**。
# 训练集上 100% 不代表什么——规则就是照着它写的；这一组才反映真实泛化能力。
HOLDOUT: list[tuple[str, str]] = [
    ("帮忙物色两个深圳的 Go 开发", "candidate_search"),
    ("有没有做过推荐系统的，杭州优先", "candidate_search"),
    ("李娜的项目经历给我看下", "candidate_detail"),
    ("王磊目前在哪家公司", "candidate_detail"),
    ("李娜跟数据平台岗匹配吗", "candidate_evaluate"),
    ("应届生能投哪些岗位", "job_recommend"),
    ("这个职位要求几年经验", "job_detail"),
    ("下午给王磊安排一场技术面", "interview_schedule"),
    ("就定第一个吧", "confirm_interview"),
    ("本周有哪些面试", "interview_list"),
    ("王磊要 70 万，我们最多给 60 万怎么谈", "offer_negotiation"),
    ("上海算法工程师今年行情怎么样", "market_insight"),
    ("试用期是几个月", "policy_qa"),
    ("内推奖金怎么发", "policy_qa"),
    ("谢谢，辛苦了", "chitchat"),
]

INTENTS = [
    "candidate_search", "candidate_detail", "candidate_evaluate",
    "job_recommend", "job_detail",
    "interview_schedule", "interview_list", "confirm_interview",
    "offer_negotiation", "market_insight", "policy_qa", "chitchat",
]


def rule(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ==================================================================== 阶段 1
# 最朴素的实现：关键词命中数打分。能跑，但会错在「修饰语被当成动作」。
NAIVE_RULES: dict[str, list[str]] = {
    "candidate_search": ["找人", "找候选人", "候选人", "人才", "简历"],
    "candidate_detail": ["简历", "什么情况", "状态"],
    "candidate_evaluate": ["匹配度", "评估"],
    "job_recommend": ["推荐岗位", "适合的岗位", "适合"],
    "job_detail": ["hc", "招了多久", "岗位详情"],
    "interview_schedule": ["约面试", "安排面试", "面试时间", "安排面"],
    "interview_list": ["面试日程", "面试安排"],
    "confirm_interview": ["确认第", "第二个时段", "选第"],
    "offer_negotiation": ["offer", "薪资谈判", "测算"],
    "market_insight": ["市场行情", "薪资大概", "行情"],
    "policy_qa": ["政策", "流程", "制度", "多久"],
    "chitchat": ["你好", "hi", "hello"],
}


def naive_predict(text: str) -> tuple[str, dict[str, float]]:
    low = text.lower()
    scores: dict[str, float] = {}
    for intent, kws in NAIVE_RULES.items():
        hits = [k for k in kws if k.lower() in low]
        if hits:
            # 这一行值得单独说：/ max(len(kws),1) * len(kws) 两项约掉了，
            # 结果就是「命中条数 × 权重」。作者大概率想写「命中率」，
            # 但代码实际算的是「命中数」——长关键词表天然占便宜。
            scores[intent] = 1.0 * len(hits) / max(len(kws), 1) * len(kws)
    if not scores:
        return "chitchat", {}
    best = max(scores, key=lambda k: scores[k])
    return best, scores


def stage1() -> None:
    rule("阶段 1｜关键词规则基线：为什么它一定会错")
    print("\n关键词表就是上面 NAIVE_RULES 里的 12 组词，按「命中条数」打分。")
    probes = [
        "给我推荐 3 个能马上安排面试的后端",
        "下周能面试的有几个候选人",
        "我适合什么岗位",
        "帮我招个 Java 后端，上海，3 到 5 年",
        "张伟现在什么情况",
    ]
    print(f"\n{'查询':38s} {'规则判定':20s} {'应判':20s} 结果")
    print("-" * 96)
    for q in probes:
        got, _ = naive_predict(q)
        want = next((i for t, i, _ in CASES if t == q), "?")
        mark = "✅" if got == want else "❌"
        print(f"{q:38s} {got:20s} {want:20s} {mark}")

    print("""
三类典型失败，每一类都需要不同的解法：

  ① 修饰语冒充动作
     「推荐 3 个**能马上安排面试**的后端」——「安排面试」是定语，
     真正的动作是「推荐 N 个候选人」。关键词打分只看命中了什么词，
     看不见词在句子里充当什么成分。

  ② 视角混淆
     HR 说「找候选人」，求职者说「我适合什么岗位」，主语不同、动作相反。
     「我适合什么岗位」两个词都没命中，直接掉进 chitchat。

  ③ 词表覆盖不到
     「帮我招个 Java 后端」里没有「候选人」「找人」，只有「招」。
     规则可以一个个补，但补的速度永远追不上用户换说法。
""")


# ==================================================================== 阶段 2
_CITY = ["北京", "上海", "深圳", "广州", "杭州", "成都", "南京", "武汉", "西安", "苏州"]
_SKILL = ["Python", "Java", "Go", "LLM", "RAG", "Agent", "大模型", "Spark", "Flink",
          "Vue", "React", "MySQL", "Redis", "微服务", "分布式"]
_DIRECTION = {
    "后端|服务端": ["Java", "Go", "Spring", "MySQL", "Redis", "分布式", "高并发", "微服务"],
    "前端|web前端": ["Vue", "React", "TypeScript", "Webpack"],
    "算法|机器学习": ["机器学习", "深度学习", "推荐系统", "NLP"],
    "数据|数仓|大数据": ["Spark", "Flink", "数仓", "Hive"],
}
_NAME_TAIL = (r"(?=[的和与跟，,。；;？?！!\s]|现在|目前|最近|状态|情况|简历|约|安排|面试|"
              r"测算|评估|推荐|匹配|看|的|$)")
_NAME_LEAD = r"(?:^|[，,。；;？?！!\s]|给|为|帮|和|与|跟|把|找|约|安排|评估)"


def extract_slots(text: str) -> dict:
    """单遍、方向化的槽位抽取。"""
    s: dict = {}

    cities = [c for c in _CITY if c in text]
    if cities:
        s["cities"] = cities

    skills = [k for k in _SKILL if k.lower() in text.lower()]
    if skills:
        s["skills_any"] = skills

    # 方向词 → 技能集（「找个后端」里的「后端」不是技能名）
    for direction, dskills in _DIRECTION.items():
        if re.search(direction, text):
            s["direction"] = direction.split("|")[0]
            s.setdefault("direction_skills", dskills)
            break

    # 人数：阿拉伯 + 中文数字；排除「一个月」「3 个月」「第二个时段」这类非计数用法
    cn = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8}
    nums = [int(x) for x in re.findall(r"(?<![年月日第\d])(\d+)\s*(?:个|名|人)(?![月薪级段])", text)]
    nums += [cn[m] for m in re.findall(r"(?<![年月日第])([一两二三四五六七八])\s*(?:个|名|位)(?![月薪级段])", text)]
    if nums:
        s["top_n"] = max(nums)

    # 年限：方向化（以上=下限，以下/不超过=上限）
    rng = re.search(r"(\d+(?:\.\d+)?)\s*[-~至到]\s*(\d+(?:\.\d+)?)\s*年", text)
    if rng:
        s["min_years"], s["max_years"] = float(rng.group(1)), float(rng.group(2))
    else:
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*年", text):
            v = float(m.group(1))
            after = text[m.end():m.end() + 4]
            before = text[max(0, m.start() - 6):m.start()]
            negated = bool(re.search(r"不考虑|不要|排除|不收", before))
            is_max = any(after.startswith(w) for w in ("以下", "以内", "之内")) or \
                any(before.endswith(w) for w in ("不超过", "不到", "最多", "至多"))
            if is_max or (negated and "以上" not in after):
                s["max_years"] = v          # 「3 年以下」→ 上限
            else:
                s["min_years"] = v          # 「5 年以上」→ 下限

    # 薪资：区分月薪/年薪，方向就近判定
    sal = re.search(r"(\d+(?:\.\d+)?)\s*([kK万])\s*[^，。；]{0,4}?(以内|以下|以上|一个月|/月)?", text)
    if sal:
        v = float(sal.group(1))
        v = v * 10000 if sal.group(2) == "万" else v * 1000
        monthly_ctx = bool(re.search(r"预算|月薪|一个月|/月", text))
        annual = v * 12 if monthly_ctx or sal.group(2) in ("k", "K") else v
        tail = sal.group(3) or ""
        if tail in ("以上",):
            s["min_salary"] = annual
        else:
            s["max_salary"] = annual
        if monthly_ctx:
            s["monthly"] = True

    # 姓名：姓氏 + 边界词 + 排除技能/城市/功能词
    surnames = "王李张刘陈杨黄赵周吴徐孙马朱胡林郭何高罗郑梁谢宋唐许韩冯邓曹彭曾"
    for m in re.finditer(rf"{_NAME_LEAD}\s*([{surnames}][一-龥]{{1,2}}?){_NAME_TAIL}", text):
        cand = m.group(1)
        if cand in ("张伟", "李娜", "王强", "郑凯", "范晓东", "徐峰", "赵磊") or \
                not any(w in cand for w in ("候选", "岗位", "北京", "上海")):
            if len(cand) >= 2 and cand not in _SKILL:
                s["candidate_name"] = cand
                break

    # 岗位名：只认「和/与 <X> 的匹配度」「<X> 岗位」这类结构
    jm = re.search(r"[和与跟]\s*([A-Za-z一-鿿+#/]{2,20}?)\s*(?:岗位|职位|的匹配度|匹配度)", text)
    if jm:
        t = jm.group(1).strip()
        if not re.search(r"多久|多少|几个|hc", t, re.I):
            s["job_title"] = t

    # 院校 / 到岗 / 确认序号 / 求职者
    if re.search(r"985", text):
        s["school_tier"] = "985"
    if re.search(r"马上到岗|尽快到岗|随时到岗|一周内到岗", text):
        s["availability_max_days"] = 15
    slot = re.search(r"[确认选]第([一二三123])[个号时]", text)
    if slot:
        g = slot.group(1)
        # 注意 dict.get(k, default) 的 default 是**立即求值**的：
        # 写成 .get(g, int(g) - 1) 时，即使命中了中文键，int("二") 也会先被算出来并抛错。
        s["confirm_slot"] = {"一": 0, "二": 1, "三": 2}.get(g) if g in "一二三" else int(g) - 1
    if re.search(r"找工作|我想找|我适合|我能做|我 \d|我\d", text):
        s["seeker"] = True
    return s


def stage2() -> None:
    rule("阶段 2｜槽位抽取：意图对了、参数错了，答案依然是错的")
    probes = [
        "帮我招个 Java 后端，上海，3 到 5 年",
        "不要应届生，要能马上到岗的",
        "有没有便宜点的候选人，预算 2 万以内",
        "北京后端，月薪 30k 以上",
        "评估张伟和 AI平台工程师 的匹配度",
        "确认第二个时段",
    ]
    for q in probes:
        print(f"\n{q}\n  → {extract_slots(q)}")

    print("""
三个必须做对的方向判定（每一处错了，正确答案就会变成错答案）：

  · 年限：「3 到 5 年」是区间；「5 年以上」只有下限；
          「3 年以下」「不超过 5 年」只有上限；
          「不考虑 5 年以上的」方向要**翻转**成「≤ 5 年」。
          早期实现把三者都当成「出现过的数字取最小做下限」，
          「3 年以下」被解析成 min=3/max=3 —— 一个点，谁都筛不出来。

  · 薪资：单位 + 方向两件事。「60k」默认月薪要 ×12；
          「年薪 80 万」已是年薪；「预算 2 万」口语默认月薪。
          方向看离数字最近的修饰词，不要看整句——
          一句话里可以同时有「5 年以上」和「60k 以内」。

  · 姓名：姓氏启发式最容易误伤。「年薪 80 万以内」里的「万以内」
          以「万」开头，会被当成一个人名。先把薪资片段屏蔽再抽姓名。
""")


# ==================================================================== 阶段 3
class CharNgramEmbedder:
    """字符 n-gram + signed hashing。零依赖，只为演示「语义兜底」这一层。"""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        t = re.sub(r"\s+", "", text.lower())
        for n in (1, 2, 3):
            for i in range(len(t) - n + 1):
                g = t[i:i + n]
                h = int(hashlib.md5(g.encode()).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h >> 20) & 1 else -1.0
                v[idx] += sign
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))


# 每个意图只给 4 条示例句，不需要穷举所有说法
EXEMPLARS = {
    "candidate_search": ["帮我找几个候选人", "有没有合适的简历", "招个后端工程师", "推荐几个人选"],
    "candidate_detail": ["发我看看他的简历", "这个人现在什么情况", "候选人的详细资料"],
    "candidate_evaluate": ["他和这个岗位匹配吗", "评估一下匹配度", "适不适合这个职位"],
    "job_recommend": ["他适合什么岗位", "我想找份工作", "有什么岗位适合我", "推荐几个岗位"],
    "job_detail": ["这个岗位还招吗", "岗位的薪资范围", "还剩几个名额"],
    "interview_schedule": ["帮他约个面试", "安排下面试时间", "面试改到明天"],
    "interview_list": ["下周有哪些面试", "面试日程安排", "这周几场面试"],
    "confirm_interview": ["确认第二个时段", "就选第一个", "可以，选这个时间"],
    "offer_negotiation": ["测算一下 offer", "薪资谈得怎么样", "给他开多少钱"],
    "market_insight": ["这个方向行情如何", "薪资大概什么水平", "人才供给紧不紧张"],
    "policy_qa": ["公司制度怎么规定的", "面试改期要提前多久", "背调流程是什么"],
    "chitchat": ["你好", "谢谢", "你是谁", "在吗"],
}


def stage3() -> None:
    rule("阶段 3｜字符 n-gram 向量兜底：处理「没见过的说法」")
    emb = CharNgramEmbedder()
    vecs = {i: [emb.embed(e) for e in ex] for i, ex in EXEMPLARS.items()}

    probes = [
        ("给我推荐 3 个能马上安排面试的后端", "candidate_search"),
        ("下周能面试的有几个候选人", "interview_list"),
        ("我适合什么岗位", "job_recommend"),
        ("帮我招个 Java 后端，上海，3 到 5 年", "candidate_search"),
        ("张伟现在什么情况", "candidate_detail"),
    ]
    print(f"\n{'查询':36s} {'最像的意图':20s} {'相似度':>7s}  结果")
    print("-" * 84)
    for q, want in probes:
        qv = emb.embed(q)
        best_i, best_s, second = "", 0.0, 0.0
        for intent, evs in vecs.items():
            s = max(emb.cosine(qv, e) for e in evs)
            if s > best_s:
                second, best_s, best_i = best_s, s, intent
            elif s > second:
                second = s
        mark = "✅" if best_i == want else "❌"
        print(f"{q:36s} {best_i:20s} {best_s:7.3f}  {mark}")

    print("""
要点：

  · 示例句是「意图的锚点」，不是训练集。4 句/意图就能覆盖大部分改写。
  · 判据不是「相似度绝对值」，而是「第一名与第二名的间隔」：
    间隔太小说明这句话本身有歧义，该反问而不是硬选。
  · 这一层不需要 GPU、不需要 Key，几百条 chunk 上毫秒级完成。
    生产上换成 BGE/text-embedding-3 只是把 embed() 换掉，逻辑不变。
""")


# ==================================================================== 阶段 4
LLM_PROMPT = """你是招聘域的意图识别器。把用户的话映射到以下意图之一：
{intents}

规则：
1. 只输出 JSON：{"intent": "...", "confidence": 0.0-1.0, "entities": {...}}
2. entities 只填原文能直接确认的字段，不要臆测。
3. 用户明确说「不考虑 X」时，在 entities 顶层加 "negated": ["X"]。

用户：{query}
"""


def render_prompt(query: str) -> str:
    """用 replace 而不是 str.format 渲染。

    Prompt 里本身含 JSON 示例（单花括号）。str.format 会把 {"intent" 当成
    占位符去解析，直接抛 KeyError——这是写 LLM prompt 时最常见的低级错误。
    """
    return (LLM_PROMPT
            .replace("{intents}", "\n".join(f"- {i}" for i in INTENTS))
            .replace("{query}", query))


def mock_llm(query: str) -> dict:
    """模拟真实 LLM 的结构化输出。

    真实场景里这里是 llm.chat_json(prompt, query, schema_hint=...)。
    模拟值故意包含两种坏情况：不存在的标签、以及过度自信的低质量输出。
    """
    table = {
        "给我推荐 3 个能马上安排面试的后端": {"intent": "candidate_search", "confidence": 0.93,
                                   "entities": {"top_n": 3, "direction": "后端"}},
        "我适合什么岗位": {"intent": "job_recommend", "confidence": 0.95,
                      "entities": {"seeker": True}},
        "下周能面试的有几个候选人": {"intent": "interview_list", "confidence": 0.91, "entities": {}},
        # 坏情况 A：模型输出了不存在的意图
        "这个岗位招了多久了，还有几个 HC": {"intent": "job_status", "confidence": 0.88, "entities": {}},
        # 坏情况 B：置信度虚高但标签错
        "前端现在好找工作吗": {"intent": "candidate_search", "confidence": 0.96, "entities": {}},
    }
    return table.get(query, {"intent": "chitchat", "confidence": 0.5, "entities": {}})


def stage4() -> None:
    rule("阶段 4｜LLM 通道：结构化输出必须校验，不能直接信")
    print("\nPrompt 骨架：")
    print("-" * 78)
    print(render_prompt("给我推荐 3 个能马上安排面试的后端"))
    print("-" * 78)

    probes = [
        ("给我推荐 3 个能马上安排面试的后端", 1.6),   # 规则分
        ("这个岗位招了多久了，还有几个 HC", 1.0),
        ("前端现在好找工作吗", 1.0),
    ]
    print(f"\n{'查询':38s} {'LLM 输出':20s} {'处理结果'}")
    print("-" * 92)
    for q, rule_conf in probes:
        out = mock_llm(q)
        intent, conf = out["intent"], out["confidence"]
        if intent not in INTENTS:
            verdict = f"拒绝（标签 {intent!r} 不在白名单）→ 回落规则"
        elif conf < rule_conf * 0.9:
            verdict = "拒绝（置信度不高于规则）→ 回落规则"
        else:
            verdict = "采信"
        print(f"{q:38s} {intent:20s} {verdict}")

    print("""
三条硬规则（每一条都对应线上踩过的坑）：

  1. **标签白名单**：LLM 会造 `job_status` 这种「看起来很像」的新标签。
     不在 INTENTS 里的一律丢弃，不要 try/except 兜住就算了。
  2. **置信度门槛**：LLM 自报 0.96 也可能错（上面的「前端好找工作吗」）。
     要求它能压过规则分才采信，否则以规则为准。
  3. **实体只补不改**：LLM 抽到规则没抽到的字段可以加，
     规则已经确认的字段不许覆盖——否则「薪资 60k 以内」会被改写成 79k。
""")


# ==================================================================== 阶段 5
def layered_predict(text: str) -> tuple[str, float, str]:
    """规则 → 向量 → LLM 的三层合流，返回 (意图, 置信度, 来源)。"""
    # 第 1 层：改进后的规则（含结构规则）
    got, scores = improved_rules(text)
    if got and scores.get(got, 0) >= 1.5:
        conf = min(0.98, 0.55 + 0.05 * scores[got])
        return got, round(conf, 2), "rules"

    # 第 2 层：向量兜底
    emb = CharNgramEmbedder()
    qv = emb.embed(text)
    ranked = []
    for intent, exs in EXEMPLARS.items():
        s = max(emb.cosine(qv, emb.embed(e)) for e in exs)
        ranked.append((s, intent))
    ranked.sort(reverse=True)
    best_s, best_i = ranked[0]
    margin = best_s - ranked[1][0]
    # 阈值不能太低：「随便看看」这类无信息的话，跟任何示例都有 0.3 左右的
    # 字符重合，低了就会硬塞一个意图。宁可落到「引导示例」，也不要乱猜。
    if best_s >= 0.34 and margin >= 0.03:
        return best_i, round(0.5 + 0.4 * best_s, 2), "embedding"

    # 第 3 层：LLM（此处为模拟）
    out = mock_llm(text)
    if out["intent"] in INTENTS:
        return out["intent"], out["confidence"], "llm"
    return "chitchat", 0.5, "fallback"


def improved_rules(text: str) -> tuple[str, dict[str, float]]:
    """改进版规则：在关键词之上补「结构规则」。"""
    scores: dict[str, float] = defaultdict(float)

    # 结构规则 1：显式「推荐/找 N 个 <对象>」→ 人才检索，压掉修饰语里的「安排面试」。
    # 「几个」也要算数量词（「找几个 985 毕业的算法」）。
    if re.search(r"(推荐|找|筛|拉|挑)\s*([0-9一两二三四五六七八九十]+\s*)?(个|名|位)?"
                 r"[^，。；]{0,10}(候选人|人选|后端|前端|算法|开发|工程师|数据|简历)", text) and \
            re.search(r"(推荐|找|筛|拉|挑)", text):
        scores["candidate_search"] += 2.0
        scores["interview_schedule"] -= 0.5

    # 结构规则 2：时段区间 + 面试 → 日程查询，而不是排期
    if re.search(r"(下周|这周|本周|明天|今天)[^，。；]{0,8}面试", text) or \
            re.search(r"有几(个|场)面试|面试日程", text):
        scores["interview_list"] += 2.2

    # 结构规则 3：第一人称 + 找/适合 → 求职者视角，不是 HR 检索
    if re.search(r"找工作|求职|我想找|我适合|我能做", text):
        scores["job_recommend"] += 2.2
        scores["candidate_search"] -= 0.6

    # 结构规则 4：候选人在场 + 简历/情况 → 详情
    if re.search(r"(简历|什么情况|现在怎么样|详细资料)", text):
        scores["candidate_detail"] += 1.8
        scores["candidate_search"] -= 0.4

    # 结构规则 5：薪资水平 / 好不好找 → 市场行情
    if re.search(r"好找工作|薪资(大概|水平|范围)|什么水平|能招到|行情", text):
        scores["market_insight"] += 2.0
        scores["candidate_search"] -= 0.4

    # 结构规则 6：岗位属性（HC / 招了多久 / 薪资多少）→ 岗位详情
    if re.search(r"(这个|该)?岗位[^，。；]{0,8}(薪资|工资|还招|HC|hc|招了多久)", text):
        scores["job_detail"] += 2.0

    # 结构规则 7：确认时段
    if re.search(r"[确认选]第[一二三123][个号时]", text):
        scores["confirm_interview"] += 2.5

    # 结构规则 8：匹配度
    if "匹配度" in text:
        scores["candidate_evaluate"] += 2.0

    # 结构规则 9：制度类提问
    if re.search(r"提前多久|需要提前|流程|制度|政策|背调|规定", text):
        scores["policy_qa"] += 1.8

    # 结构规则 10：offer 测算
    if re.search(r"offer|测算|总包|薪资谈判", text, re.I):
        scores["offer_negotiation"] += 1.8

    # 结构规则 12：槽位密度先验——一堆筛选条件堆在一起，本身就是「找人」的信号。
    # 「北京和上海都可以，做 LLM 的，5 年往上」这类句子没有意图动词，
    # 但它们全是**约束条件**；在招聘域里，一堆约束条件只可能指向人才检索。
    # 注意：求职视角（我…找岗位）要先排除，否则「我 6 年经验在北京」被他自己的
    # 条件命中后判成 candidate_search。
    seeker = bool(re.search(r"找工作|求职|我想找|我适合|我能做|我\s*\d+\s*年", text))
    slot_hits = 0
    slot_hits += bool(re.search(r"北京|上海|深圳|广州|杭州|成都|南京|武汉", text))
    slot_hits += bool(re.search(r"\d+\s*年", text))
    slot_hits += bool(re.search(r"LLM|RAG|Java|Go|Python|前端|后端|算法|数据|985|211|到岗|应届", text, re.I))
    if slot_hits >= 2 and not seeker:
        scores["candidate_search"] += 0.9 * slot_hits

    # 结构规则 13：市场行情的否证——「好找工作吗」问的是市场，不是求职动作。
    if re.search(r"好不好找工作|好找工作吗|行情|什么水平|能招到", text):
        scores["job_recommend"] -= 1.5

    # 结构规则 14：「推荐/有什么 + 岗位」的对象是岗位，不是候选人。
    # 不要求两词相邻——「推荐北京月薪 5 万以上的岗位」中间隔着条件。
    if re.search(r"岗位|职位", text) and re.search(r"推荐|有什么|有哪些|合适的?", text) \
            and not re.search(r"候选人|人选|简历", text):
        scores["job_recommend"] += 2.0
        scores["candidate_search"] -= 1.2

    # 结构规则 15：纯排除/时效类约束（不要应届、马上到岗）也是找人条件，
    # 这类句子既没有意图动词，也只有 1-2 个槽位，容易被判成闲聊。
    if re.search(r"应届|到岗|入职|在职|离职|外地", text) and \
            re.search(r"不要|不考虑|排除|马上|尽快|随时|优先", text):
        scores["candidate_search"] += 1.8

    # 结构规则 16：预算 / 便宜点 / 性价比 —— HR 谈预算时，动作一定是找人。
    if re.search(r"预算|便宜|性价比|价格|成本", text):
        scores["candidate_search"] += 1.8

    # 结构规则 17：约面 / 排期动作（「帮我和张伟约个面试」）。
    # 必须是谓语：「能马上安排面试的后端」里「安排面试的」是定语，修饰的是「后端」，
    # 用负向先行断言 (?!的) 排除，否则会和规则 1 抢分。
    if re.search(r"(约|排|安排)[^，。；的]{0,4}面试?(?!试|的)|面试改", text):
        scores["interview_schedule"] += 2.0

    # 「帮我招个 Java 后端」是最常见的 HR 说法，却一个规则关键词都没命中：
    # 「招」不在词表里，「Java 后端」是方向不是意图词。这类「动词+对象」结构
    # 必须单独成规则，不能靠堆关键词。
    if re.search(r"(招|招聘|补充|补一个|要个|来个)", text) and \
            re.search(r"(Java|Go|Python|前端|后端|算法|数据|测试|运维|工程师|开发|产品)", text):
        scores["candidate_search"] += 2.0

    # 兜底：关键词命中（低权重，只用于没有结构规则时）
    for intent, kws in NAIVE_RULES.items():
        hits = [k for k in kws if k.lower() in text.lower()]
        if hits:
            scores[intent] += 0.35 * len(hits)

    # 只有寒暄词、且句子很短 → chitchat
    if len(text) <= 6 and re.search(r"你好|hi|hello|谢谢|在吗", text, re.I):
        scores["chitchat"] += 2.0

    if not scores:
        return "", {}
    best = max(scores, key=lambda k: scores[k])
    return (best if scores[best] > 0 else ""), dict(scores)


def stage5() -> None:
    rule("阶段 5｜置信度与澄清阈值：不确定时反问，比猜错强")
    probes = [
        "帮我招个 Java 后端，上海，3 到 5 年",
        "给我推荐 3 个能马上安排面试的后端",
        "我适合什么岗位",
        "下周能面试的有几个候选人",
        "嗯",
        "随便看看",
    ]
    print(f"\n{'查询':34s} {'意图':20s} {'置信度':>6s} {'来源':10s} 动作")
    print("-" * 92)
    for q in probes:
        intent, conf, src = layered_predict(q)
        action = "直接执行" if conf >= 0.6 else ("澄清追问" if conf >= 0.35 else "引导示例")
        print(f"{q:34s} {intent:20s} {conf:6.2f} {src:10s} {action}")

    print("""
置信度不是一个装饰性字段，它决定**要不要把问题抛回给用户**：

    高（≥0.6）  直接进工具链
    中（0.35~0.6）澄清：只补问缺的那一项，不要重复用户已经说过的
    低（<0.35）  给示例引导

关键：澄清不是「没听懂，请重说」，而是「已识别 A、B，还差 C」。
后者用户回一句就能继续，前者要重新打一遍需求。
""")


# ==================================================================== 阶段 6
class Session:
    """极简会话：只存「跨轮必须继承」的东西。"""

    def __init__(self) -> None:
        self.mem: dict = {}

    def run(self, text: str) -> dict:
        intent, conf, src = layered_predict(text)
        slots = extract_slots(text)

        # 省略主语的承接句：「那再给他排一个」「再约一场」单看没有意图动词的完整宾语，
        # 但上一轮刚在排面试 + 代词「他/她」指向上一轮的人 → 继续排期。
        # 这条规则只能放在会话层：脱离上下文，「再排一个」可能是任何东西。
        if self.mem.get("last_intent") in ("interview_schedule", "confirm_interview") and \
                re.search(r"再.{0,4}(排|约|安排)|换个时间|改到", text):
            intent, conf, src = "interview_schedule", 0.8, "session"

        # 「排一个」「约个面试」里的「一个」是量词，不是要几个候选人；
        # top_n 只对检索类意图有意义，其它意图一律丢弃，防止误传给下游工具。
        if intent not in ("candidate_search", "job_recommend"):
            slots.pop("top_n", None)

        # 指代继承：本轮没给的槽位，从上一轮带过来
        for k in ("candidate_name", "candidate_id", "pending_slots", "last_results"):
            if k not in slots and k in self.mem:
                slots[k] = self.mem[k]

        # 本轮长得像「确认」，但没点名 → 一定是承接上一轮的排期
        if intent == "confirm_interview" and "candidate_name" not in slots and "candidate_name" in self.mem:
            slots["candidate_name"] = self.mem["candidate_name"]

        if "candidate_name" in slots:
            self.mem["candidate_name"] = slots["candidate_name"]
        if intent == "interview_schedule":
            self.mem["pending_slots"] = ["2 天后 10:00", "2 天后 11:00", "2 天后 14:00"]
        self.mem["last_intent"] = intent
        return {"intent": intent, "confidence": conf, "source": src, "slots": slots}


def stage6() -> None:
    rule("阶段 6｜多轮指代：单句看不出来的意思，要靠会话补")
    s = Session()
    turns = [
        "帮我和张伟约个面试",
        "确认第二个时段",
        "那再给他排一个",
    ]
    print()
    for t in turns:
        r = s.run(t)
        print(f"用户：{t}")
        print(f"  → intent={r['intent']:20s} conf={r['confidence']:.2f} "
              f"slots={ {k: v for k, v in r['slots'].items()} }\n")

    print("""
「确认第二个时段」这句话单看没有任何招聘实体：没有名字、没有岗位。
它之所以能被理解，是因为上一轮已经在排张伟的面试。
所以意图识别不是纯文本函数，它的签名应该是：

    classify(text, session_state) -> (intent, slots, confidence)

只做无状态分类，多轮一问就崩——这是最容易在演示里掩盖、
上线第一天就被用户发现的问题。
""")


# ==================================================================== 阶段 7
def _report(name: str, fn, gold: list[tuple[str, str]]) -> None:
    """打印一个系统在一组样本上的准确率、宏平均 P/R/F1 和最常见的错判。

    宏平均按「金标里出现过的意图 ∪ 被预测出来的意图」取平均，
    这样把别的意图误判成 chitchat 也会拉低 chitchat 的 precision，不会被藏起来。
    """
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn_: dict[str, int] = defaultdict(int)
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    for q, want in gold:
        got, _ = fn(q)
        if got == want:
            tp[want] += 1
        else:
            fp[got] += 1
            fn_[want] += 1
            confusion[(want, got)] += 1

    labels = {w for _, w in gold} | set(fp)
    prec = sum(tp[i] / (tp[i] + fp[i]) if tp[i] + fp[i] else 0.0 for i in labels) / len(labels)
    rec = sum(tp[i] / (tp[i] + fn_[i]) if tp[i] + fn_[i] else 0.0 for i in labels) / len(labels)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    correct = sum(tp.values())
    print(f"【{name}】准确率 {correct}/{len(gold)} = {correct/len(gold):.1%}  "
          f"宏平均 P/R/F1 = {prec:.2f}/{rec:.2f}/{f1:.2f}")
    for (want, got), n in sorted(confusion.items(), key=lambda x: -x[1])[:5]:
        print(f"    应为 {want:18s} 判成 {got:18s} ×{n}")


def stage7() -> None:
    rule("阶段 7｜评测：分意图 P/R/F1 + 混淆矩阵")
    systems = {
        "纯关键词规则": lambda t: (naive_predict(t)[0], "rules"),
        "规则+结构规则": lambda t: (improved_rules(t)[0] or "chitchat", "rules"),
        "三层合流": lambda t: (layered_predict(t)[0], "cascade"),
    }
    splits = {
        "调参集（规则照着它写，分数虚高）": [(q, i) for q, i, _ in CASES],
        "留出集（从未参与调参，看真实泛化）": HOLDOUT,
    }
    for split_name, gold in splits.items():
        print(f"\n━━━━ {split_name}，{len(gold)} 条 ━━━━")
        for name, fn in systems.items():
            _report(name, fn, gold)

    print("""
这组数字本身就是最重要的一课：调参集 100%，留出集 40%。

  · 结构规则 1~17 全是看着调参集的错判一条条补出来的，规则「记住」了这 25 句话，
    而不是学会了「招聘意图」。换一种说法（「物色」「看下」「投哪些」）就失效。
  · 留出集的错判几乎全落到 chitchat：规则没命中 → 向量相似度不够 → 落到兜底。
    demo 里的 LLM 层是确定性 Mock，补不上这个缺口；生产里真正承担泛化的是这一层。
  · 所以正确的工程顺序是：规则只管高频、高确定性的说法（追求 precision），
    长尾交给 embedding / LLM（追求 recall），并且永远用没见过的数据验收。
  · 发现留出集错了，不能直接照着它补规则——那它就变成第二个调参集了。
    应该把它并入训练数据，再另找一批新的线上真实问法做留出集。

为什么要分意图看 P/R/F1，而不是只看总准确率：

  · 总准确率会被大类稀释。12 个意图里 candidate_search 占三分之一，
    它全对就能让总分很好看，而 job_detail 全错只掉 2 分。
  · FP 和 FN 的代价不同：把「查岗位详情」判成「推荐候选人」，
    用户会收到一堆人；把「推荐候选人」判成「查详情」，用户什么也拿不到。
  · 混淆矩阵就是改进清单：箭头最粗的那条，就是你下一个要加的结构规则。
""")


# ==================================================================== main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0, help="只跑某一阶段 1-7")
    args = ap.parse_args()
    stages = {1: stage1, 2: stage2, 3: stage3, 4: stage4, 5: stage5, 6: stage6, 7: stage7}
    todo = [args.stage] if args.stage else sorted(stages)
    for i in todo:
        stages[i]()


if __name__ == "__main__":
    main()
