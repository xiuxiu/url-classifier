"""
URL Classifier Inference — ML 模型 + 规则兜底，双保险

规则兜底逻辑：
  1. 高置信规则（精确 pattern）→ 直接返回，绕过模型
  2. 数字 ID 密集出现 → detail
  3. 低置信模型预测（conf < 0.6）→ 参考规则
  4. 规则与模型冲突（高置信冲突）→ 规则优先（人工经验更可靠）

用法:
  from inference import UrlClassifier
  clf = UrlClassifier("data/models/url_classifier.pkl")
  label, conf = clf.classify("https://github.com/facebook/react")
"""

import pickle
import re
from pathlib import Path


# ─────────────────────────────────────────────────
# 规则库
# ─────────────────────────────────────────────────

class RuleEngine:
    """纯规则兜底分类器，独立于 ML 模型"""

    # 列表页强标识（路径中出现 = 列表页）
    LIST_PATH_MARKERS = {
        "search", "browse", "list", "category", "tag", "archive",
        "page", "index", "trending", "popular", "latest", "recent",
        "top", "new-releases", "most-wished", "sitemap", "feed", "rss",
        "featured", "recommended", "explore", "discover", "community",
        "forum", "calendar", "hot", "rising", "best", "review",
        "all", "countries", "saved_hotel",
    }

    # 详情页强标识（路径中出现 = 详情页）
    DETAIL_PATH_MARKERS = {
        "article", "story", "product", "item", "video",
    }

    def classify(self, url: str) -> str | None:
        """
        返回 'list' | 'detail' | None
        None = 不确定，让模型决定
        """
        parsed = self._parse(url)
        path = parsed["path"]
        path_lower = path.lower()

        # ── 1. 数字 ID 密集（详情页特征）───────────
        digit_ratio = sum(c.isdigit() for c in path) / max(len(path), 1)
        path_only = path.split("?")[0]
        # 路径末尾是 6 位以上纯数字 → 详情页
        if re.search(r'/(\d{6,})$', path_only):
            return "detail"
        # 路径含大量数字（>30%）且有斜线分隔 → 详情页
        if digit_ratio > 0.35 and "/" in path:
            return "detail"

        # ── 2. 强列表标识（按路径分词匹配，防止 hotel→hot 误判）─
        # 把路径分成独立 segment："/news/technology" → ["", "news", "technology"]
        segments = [s.lower() for s in path.split("/")]
        for marker in self.LIST_PATH_MARKERS:
            for seg in segments:
                if seg == marker or seg.startswith(f"{marker}?") or seg.startswith(f"{marker}?"):
                    # "search?q=" or "search?" or exact "search"
                    return "list"
            # 也检查 query string 里有没有（?search=xxx 这种）
            if marker in parsed.get("query", "").lower():
                return "list"

        # ── 3. 强详情标识（按 path segment 精确匹配）───────
        for marker in self.DETAIL_PATH_MARKERS:
            if f"/{marker}" in path_lower:
                return "detail"

        # ── 4. 特定 path segment 模式 ─────────────
        # /dp/ASIN（亚马逊详情）
        if re.search(r'/dp/[A-Z0-9]', path):
            return "detail"
        # /itm/数字（eBay 详情）
        if re.search(r'/itm/\d', path):
            return "detail"
        # /Hotel_Review-...-d数字（TripAdvisor 详情）
        if re.search(r'Hotel_Review.*-d\d', path, re.IGNORECASE):
            return "detail"
        # /Restaurant_Review-...-d数字
        if re.search(r'Restaurant_Review.*-d\d', path, re.IGNORECASE):
            return "detail"
        # /Attraction_Review-...-d数字
        if re.search(r'Attraction_Review.*-d\d', path, re.IGNORECASE):
            return "detail"
        # /abs/数字（arxiv abstract）
        if re.search(r'/abs/\d', path):
            return "detail"
        # /pdf/数字（arxiv pdf）
        if re.search(r'/pdf/\d', path):
            return "detail"
        # /title/数字（IMDb 详情）
        if re.search(r'/title/\d', path):
            return "detail"
        # /name/nm数字（IMDb 人名页）
        if re.search(r'/name/nm\d', path):
            return "detail"
        # /watch?v=（YouTube 视频）
        if re.search(r'/watch\?v=', path):
            return "detail"
        # /shorts/（YouTube 短视频）
        if "/shorts/" in path:
            return "detail"
        # /project/包名（pypi 详情）
        # pypi /project/name（无版本号）= 详情页
        if re.match(r'^/project/[^/]+/?$', path) or re.match(r'^/pypi/[^/]+/?$', path):
            return "detail"
        # pypi /project/name/version
        if re.match(r'^/project/[^/]+/\d', path):
            return "detail"
        if re.match(r'^/pypi/[^/]+/\d', path):
            return "detail"

        # ── 5. github.com ──────────────────────────
        # /user/repo = 详情（repo 主页）
        # /search = 列表
        if parsed["domain"] == "github.com":
            # /{user}/{repo} 且 repo 名不含搜索关键词 → 详情
            segs = [s for s in path.split("/") if s]
            if len(segs) >= 2 and segs[0] and segs[1]:
                # 不是搜索页
                if "search" not in path_lower:
                    return "detail"
            if "/search" in path_lower:
                return "list"

        # 不确定
        return None

    @staticmethod
    def _parse(url: str) -> dict:
        try:
            from urllib.parse import urlparse
        except ImportError:
            return {"domain": "", "path": ""}
        parsed = urlparse(url)
        return {
            "scheme": parsed.scheme,
            "domain": parsed.netloc.lower(),
            "path": parsed.path,
            "query": parsed.query,
        }


# ─────────────────────────────────────────────────
# 主分类器
# ─────────────────────────────────────────────────

class UrlClassifier:
    def __init__(self, model_path: str, *, confidence_threshold: float = 0.55):
        """
        model_path: pickle 模型路径
        confidence_threshold: 规则与模型冲突时的置信度阈值
          - 模型置信 > 此值 → 信任模型
          - 模型置信 ≤ 此值 → 信任规则
        """
        with open(model_path, "rb") as f:
            self.vectorizer, self.clf = pickle.load(f)
        self.rule = RuleEngine()
        self.confidence_threshold = confidence_threshold

    # 高精度规则（人工验证过，极低误判率）→ 强制使用
    # 对 path 应用（相对于 URL scheme+domain 之后的部分）
    HIGH_PRIORITY_RULES = [
        (r'/dp/[A-Z0-9]',              "amazon detail"),
        (r'/itm/\d',                    "ebay detail"),
        (r'Hotel_Review.*-d\d',         "tripadvisor hotel_review"),
        (r'Restaurant_Review.*-d\d',   "tripadvisor restaurant"),
        (r'Attraction_Review.*-d\d',   "tripadvisor attraction"),
        (r'/abs/\d',                    "arxiv abstract"),
        (r'/pdf/\d',                    "arxiv pdf"),
        (r'/title/\d',                  "imdb title"),
        (r'/name/nm\d',                "imdb person"),
        (r'/watch\?v=',                "youtube video"),
        (r'/shorts/',                   "youtube shorts"),
        (r'/project/[^/]+/?$',         "pypi project"),
        (r'/pypi/[^/]+/?$',            "pypi pypi page"),
        (r'/comments/',                 "reddit comments"),
        (r'/item\?id=',                 "hackernews"),
        (r'/articles/',                 "medium articles"),
        (r'/\d{6,}$',                   "numeric ID end"),
    ]

    def classify(self, url: str) -> tuple[str, float]:
        """
        返回 (label, confidence)
        confidence 反映最终决策的确信程度（0.5-1.0）
        """
        rule_result = self.rule.classify(url)

        # ── 决策逻辑 ──────────────────────────────
        # 1. 规则命中 detail 高优先级模式 → 直接返回（不依赖 rule_result）
        # 应用到 path（不含 scheme+domain）
        url_path = url.split("?")[0]   # 去掉 query string
        for pattern, _name in self.HIGH_PRIORITY_RULES:
            if re.search(pattern, url_path, re.IGNORECASE):
                return "detail", 0.95

        # 2. 获得模型预测
        vec = self.vectorizer.transform([url])
        model_pred = self.clf.predict(vec)[0]
        model_proba = self.clf.predict_proba(vec)[0]
        model_conf = float(max(model_proba))

        if rule_result is None:
            # 规则不确定，完全信任模型
            return model_pred, model_conf

        if rule_result == model_pred:
            # 规则和模型一致 → 增强置信度
            final_conf = min(1.0, model_conf + 0.05)
            return model_pred, final_conf

        # 规则与模型冲突 → 模型高置信时信模型，否则信规则
        if model_conf >= self.confidence_threshold:
            return model_pred, model_conf
        else:
            return rule_result, 0.80

    def classify_batch(self, urls: list[str]) -> list[tuple[str, float]]:
        """批量分类"""
        vecs = self.vectorizer.transform(urls)
        preds = self.clf.predict(vecs)
        probas = self.clf.predict_proba(vecs)
        results = []
        for url, pred, proba in zip(urls, preds, probas):
            model_conf = float(max(proba))
            rule_result = self.rule.classify(url)
            if rule_result is None:
                results.append((pred, model_conf))
            elif rule_result == pred:
                results.append((pred, min(1.0, model_conf + 0.05)))
            elif model_conf >= self.confidence_threshold:
                results.append((pred, model_conf))
            else:
                results.append((rule_result, 0.80))
        return results


# ─────────────────────────────────────────────────
# 测试
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    MODEL = "C:/Users/windlx/Projects/url-classifier/data/models/url_classifier.pkl"
    clf = UrlClassifier(MODEL)

    tests = [
        # (url, expected)
        ("https://github.com/facebook/react", "detail"),
        ("https://github.com/search?q=fasttext", "list"),
        ("https://bbc.com/news/3439437565", "detail"),
        ("https://bbc.com/news/technology", "list"),
        ("https://arxiv.org/abs/2301.00001", "detail"),
        ("https://arxiv.org/list/cs/recent", "list"),
        ("https://amazon.com/dp/B09V3KXJPB", "detail"),
        ("https://amazon.com/s?k=laptop", "list"),
        ("https://youtube.com/watch?v=dQw4w9WgXcQ", "detail"),
        ("https://youtube.com/feed/subscriptions", "list"),
        ("https://tripadvisor.com/Hotel_Review-g1-d123456", "detail"),
        ("https://tripadvisor.com/Hotels-g1", "list"),
        ("https://pypi.org/project/requests", "detail"),
        ("https://pypi.org/simple/", "list"),
        ("https://imdb.com/title/tt1234567/", "detail"),
        ("https://imdb.com/search/?q=action", "list"),
        ("https://ebay.com/itm/123456789012", "detail"),
        ("https://ebay.com/sch/i.html?_nkw=laptop", "list"),
        # 未见域名（泛化测试）
        ("https://reddit.com/r/python/comments/abc123", "detail"),
        ("https://reddit.com/r/python", "list"),
        ("https://news.ycombinator.com/item?id=12345678", "detail"),
        ("https://news.ycombinator.com/", "list"),
    ]

    print(f"{'URL':55s}  {'EXPECT':6s}  {'PRED':6s}  {'CONF':5s}  {'OK?'}")
    print("-" * 90)
    correct = 0
    for url, expected in tests:
        pred, conf = clf.classify(url)
        ok = "OK" if pred == expected else "FAIL"
        if pred == expected:
            correct += 1
        print(f"{url:55s}  {expected:6s}  {pred:6s}  {conf:.2f}   {ok}")
    print(f"\nAccuracy: {correct}/{len(tests)} = {correct/len(tests)*100:.0f}%")
