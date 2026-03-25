"""
URL List/Detail Classifier — CPU-Optimized Pipeline
====================================================
模型: TfidfVectorizer (char 3-6 gram) + LogisticRegression
速度: ~100,000+ URLs/秒 (单核)，内存 ~50MB
对比 FastText C++ 版: 效果相当，推理更简单

数据格式 (JSON):
  {"url": "https://bbc.com/news/3439437565", "label": "detail", "domain": "bbc.com"}
  {"url": "https://bbc.com/news/technology", "label": "list", "domain": "bbc.com"}
"""

import json
import random
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score


# ─────────────────────────────────────────────────
# 1. 域名 → 列表页/详情页 路径模式
# ─────────────────────────────────────────────────
DOMAIN_LIST_PATTERNS = {
    # 新闻/媒体
    "bbc.com": {
        "list": [
            "/news/", "/sport/", "/business/", "/technology/", "/culture/",
            "/science/", "/world/", "/politics/", "/climate/", "/future/",
            "/newsround/", "/news/{category}", "/sport/{category}",
        ],
        "detail": [
            "/news/{id}", "/sport/{slug}-{id}", "/articles/{slug}-{id}",
            "/news/{category}/{slug}-{id}",
        ],
    },
    "cnn.com": {
        "list": [
            "/{section}/", "/{section}/{subsection}/",
            "/latest/", "/world/", "/politics/", "/business/", "/health/",
            "/climate/", "/sport/", "/culture/",
        ],
        "detail": [
            "/articles/{slug}-{id}", "/{section}/{slug}-{id}",
            "/show/{slug}", "/{slug}",
        ],
    },
    "reuters.com": {
        "list": [
            "/world/", "/business/", "/technology/", "/science/",
            "/health/", "/sports/", "/opinion/", "/analysis/",
            "/news/{topic}", "/{section}/",
        ],
        "detail": [
            "/article?id={id}", "/world/{slug}-{id}",
            "/business/{slug}/{id}", "/{section}/{slug}-{id}",
        ],
    },
    # 视频/电影
    "youtube.com": {
        "list": [
            "/feed/subscriptions", "/feed/history", "/results",
            "/channel/{channel_id}/videos", "/playlist",
            "/watch",   # watch with list context
        ],
        "detail": [
            "/watch?v={video_id}",
            "/shorts/{short_id}",
            "/channel/{channel_id}",
        ],
    },
    # 电商
    "amazon.com": {
        "list": [
            "/s?k={keyword}", "/gp/bestsellers/", "/gp/new-releases/",
            "/s/", "/gp/most-wished-for/",
        ],
        "detail": [
            "/dp/{asin}", "/product/{asin}",
        ],
    },
    "bestbuy.com": {
        "list": [
            "/site/{category}/", "/cp/{category_id}", "/category/{category}/",
            "/browse/", "/site/searchterm", "/{category}/",
        ],
        "detail": [
            "/site/{sku}.p?skuId={sku}",
            "/product.do?sku_id={sku}",
        ],
    },
    "ebay.com": {
        "list": [
            "/sch/{category}", "/b/{category}", "/itm/Search",
            "/b/{category}?{params}",
        ],
        "detail": [
            "/itm/{item_id}", "/itm/-/{item_id}",
        ],
    },
    # 视频/电影
    "youtube.com": {
        "list": [
            "/feed/subscriptions", "/feed/history", "/results",
            "/channel/{channel_id}/videos",
        ],
        "detail": [
            "/watch?v={video_id}",
        ],
    },
    "imdb.com": {
        "list": [
            "/chart/", "/search/", "/feature/", "/title/",
            "/user/{user}/lists", "/chart/{type}",
        ],
        "detail": [
            "/title/{title_id}/", "/name/{name_id}/",
        ],
    },
    # 旅游/点评
    "tripadvisor.com": {
        "list": [
            "/Hotels/", "/Restaurants/", "/Attractions/",
            "/Profile/{user}", "/Flights-",
            "/Hotels-{location}", "/Restaurants-{location}",
        ],
        "detail": [
            "/Hotel_Review-{location}-d{id}",
            "/Restaurant_Review-{location}-d{id}",
            "/Attraction_Review-{location}-d{id}",
        ],
    },
    "booking.com": {
        "list": [
            "/hotel/{country}", "/searchresults.html",
            "/saved_hotel/{id}", "/review/{topic}",
            "/domain/{topic}", "/{country}/{city}",
            "/countries/{topic}", "/browse",
        ],
        "detail": [
            "/hotel/{country}/{city}/{slug}.{id}.html",
            "/b/{slug}-{id}",
        ],
    },
    # 知识/学术
    "arxiv.org": {
        "list": [
            "/list/{topic}/recent", "/search/",
            "/group/{group}", "/{topic}",
        ],
        "detail": [
            "/abs/{id}", "/html/{id}", "/pdf/{id}",
        ],
    },
    "wikipedia.org": {
        "list": [
            "/wiki/Special:Categories", "/wiki/Category:{cat}",
            "/wiki/Portal:{portal}", "/wiki/{lang}",
        ],
        "detail": [
            "/wiki/{title}",
        ],
    },
    # 课程/学习
    "coursera.org": {
        "list": [
            "/browse", "/search", "/courses",
        ],
        "detail": [
            "/learn/{course}", "/specialization/{spec}",
            "/search?q={q}",
        ],
    },
    # 代码/PyPI
    "github.com": {
        "list": [
            "/search?q={q}", "/{user}?tab=repositories",
            "/{org}?tab=repositories",
            "/{user}?tab=stars", "/{org}",
        ],
        "detail": [
            "/{user}/{repo}", "/{user}/{repo}/tree/{branch}",
            "/{user}/{repo}/pull/{pr}", "/{user}/{repo}/issues/{issue}",
            "/{user}", "/{org}",
        ],
    },
    "pypi.org": {
        "list": [
            "/simple/", "/packages/", "/r/",
            "/search/", "/project/",
        ],
        "detail": [
            "/project/{name}", "/pypi/{name}/{version}",
            "/project/{name}/{version}",
        ],
    },
    # 体育
    "foxsports.com": {
        "list": [
            "/{league}/teams", "/{league}/{category}",
            "/players/{sport}", "/{sport}/teams",
            "/rankings", "/scores", "/standings",
        ],
        "detail": [
            "/{league}/teams/{sport}/{slug}",
            "/team/{sport}/{slug}",
            "/players/{sport}/{slug}",
            "/game/{game_id}",
        ],
    },
}

# 通用列表页标识符
LIST_PATH_KEYWORDS = [
    "search", "browse", "list", "category", "tag", "archive",
    "page", "pagen", "index", "trending", "popular", "latest",
    "recent", "top", "new-releases", "most-wished", "sitemap",
    "feed", "rss", "featured", "recommended", "explore",
    "discover", "community", "forum", "calendar",
    "hot", "rising", "best", "review", "report",
    "all", "countries", "saved_hotel", "review",
]

# 通用详情页标识符
DETAIL_PATH_KEYWORDS = [
    "article", "story", "post", "product", "item", "video",
    "review", "detail", "/d/", "/dp/", "/id/", "-d",
    "title", "name", ".html", ".htm",
]

# 全局兜底规则（domain 无关）
GLOBAL_LIST_PATTERNS = [
    r"/search[?/]", r"/\w+/\w+/\?.*=", r"/browse",
    r"/sitemap", r"/feed", r"/rss",
]

GLOBAL_DETAIL_PATTERNS = [
    r"/\d{6,}",  # 6位以上数字 = ID 类详情页
]


def expand_patterns(patterns: list, count: int = 20) -> list:
    """通过插值生成 URL 实例"""
    keywords = [
        "news", "sports", "tech", "world", "business", "culture",
        "health", "science", "politics", "food", "travel",
        "latest", "trending", "popular", "featured",
        "analysis", "report", "opinion", "breaking",
        "update", "today", "hot", "best", "top",
    ]
    slugs = [
        "what-you-need-to-know", "explainer", "analysis",
        "latest-updates", "deep-dive", "overview",
        "complete-guide", "breaking-news", "top-story",
        "in-depth-report", "weekly-roundup", "key-findings",
    ]
    results = []
    for p in patterns:
        for i in range(count):
            kw = random.choice(keywords)
            slug = random.choice(slugs)
            slug2 = random.choice(slugs)
            num = random.randint(1000000, 9999999999)
            expanded = (
                p.replace("{keyword}", kw)
                 .replace("{category}", random.choice(["tech", "world", "sports"]))
                 .replace("{category_id}", str(random.randint(1000, 999999)))
                 .replace("{id}", str(num))
                 .replace("{slug}", slug)
                 .replace("{slug2}", slug2)
                 .replace("{asin}", f"{random.randint(100,999)}{random.randint(100000000,999999999)}")
                 .replace("{sku}", str(random.randint(10000000, 99999999)))
                 .replace("{item_id}", str(random.randint(100000000, 9999999999)))
                 .replace("{video_id}", "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789_-", k=11)))
                 .replace("{title_id}", str(random.randint(1000, 99999999)))
                 .replace("{name_id}", f"nm{random.randint(1000000, 99999999)}")
                 .replace("{user}", f"user{random.randint(100, 9999)}")
                 .replace("{location}", random.choice(["London", "Paris", "Tokyo", "NewYork", "Berlin"]))
                 .replace("{country}", random.choice(["uk", "fr", "jp", "us", "de"]))
                 .replace("{city}", random.choice(["london", "paris", "tokyo", "newyork", "berlin"]))
                 .replace("{league}", random.choice(["nfl", "nba", "mlb", "nhl", "ncaaf", "ncaab"]))
                 .replace("{sport}", random.choice(["football", "basketball", "baseball", "soccer", "tennis"]))
                 .replace("{game_id}", str(random.randint(100000, 9999999)))
                 .replace("{group}", random.choice(["top", "hot", "new", "popular", "recommended"]))
                 .replace("{topic}", random.choice(keywords))
                 .replace("{section}", random.choice(["news", "business", "tech", "opinion"]))
                 .replace("{params}", f"param{random.randint(1,99)}={random.randint(1,999)}")
                 .replace("{branch}", "main")
                 .replace("{user2}", f"user{random.randint(100,999)}")
                 .replace("{repo}", f"repo{random.randint(1,999)}")
                 .replace("{q}", kw)
                 .replace("{name}", f"package{random.randint(1,999)}")
                 .replace("{version}", f"{random.randint(1,9)}.{random.randint(0,9)}.{random.randint(0,99)}")
                 .replace("{org}", f"org{random.randint(1,999)}")
                 .replace("{lang}", "en")
                 .replace("{cat}", kw)
                 .replace("{portal}", kw.title())
                 .replace("{channel_id}", f"UC{random.randint(10000000000, 99999999999)}")
            )
            results.append(expanded)
    return results


def build_synthetic_dataset() -> list:
    """生成大规模合成训练数据"""
    rows = []

    for domain, patterns in DOMAIN_LIST_PATTERNS.items():
        # 列表页
        list_paths = expand_patterns(patterns["list"], count=40)
        for path in list_paths:
            url = f"https://{domain}{path}"
            rows.append({"url": url, "label": "list", "domain": domain})

        # 详情页
        detail_paths = expand_patterns(patterns["detail"], count=40)
        for path in detail_paths:
            url = f"https://{domain}{path}"
            rows.append({"url": url, "label": "detail", "domain": domain})

    return rows


def normalize_label(label):
    """把各种 label 格式统一成 'list' / 'detail'"""
    if isinstance(label, str):
        l = label.lower()
        if l in ("list", "detail", "a"):
            return "list"
        if l in ("detail", "b"):
            return "detail"
        if l in ("0", "list"):
            return "list"
        if l in ("1", "detail"):
            return "detail"
    if isinstance(label, int):
        return "list" if label == 0 else "detail"
    return label


def load_json(path: str) -> list:
    """支持多种 JSON 格式，统一返回 [{"url": "...", "label": "list"|"detail", ...}]"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except UnicodeDecodeError:
        with open(path, "r", encoding="gbk", errors="ignore") as f:
            data = json.load(f)

    if isinstance(data, dict):
        if "urls" in data:
            data = data["urls"]
        else:
            # {"key": {...}} → 转 list
            data = [{"url": k, **v} for k, v in data.items()]

    if not isinstance(data, list):
        return []

    results = []
    for item in data:
        if not isinstance(item, dict):
            continue
        # 提取 url 字段（支持 "url" 或 "text"）
        url = item.get("url") or item.get("text") or ""
        if not url or not isinstance(url, str):
            continue
        # 去掉非 URL 内容（如 url_patterns_labeled 的 text 含多余文本）
        if "\n" in url or "<|" in url:
            url = url.split("\n")[0].strip()
        label = normalize_label(item.get("label") or item.get("label_str") or "")
        if label not in ("list", "detail"):
            continue
        results.append({
            "url": url,
            "label": label,
            "domain": item.get("domain", ""),
        })
    return results


def classify_path(url: str) -> str:
    """
    轻量规则兜底分类器 — 用于数据清洗和基准对比。
    实际生产环境只用 ML 模型，此函数作 fallback 和置信度参考。
    """
    url_lower = url.lower()

    # 数字 ID 密集出现 → 详情页
    import re
    path = url.split("?")[0]
    digit_ratio = sum(c.isdigit() for c in path) / max(len(path), 1)
    if digit_ratio > 0.3:
        return "detail"

    # 强列表标识
    for kw in LIST_PATH_KEYWORDS:
        if f"/{kw}" in url_lower or url_lower.endswith(f"/{kw}"):
            return "list"

    # 强详情标识
    for kw in DETAIL_PATH_KEYWORDS:
        if f"/{kw}" in url_lower:
            return "detail"

    # 数字结尾（/123456）
    if re.search(r'/\d{6,}$', path):
        return "detail"

    return "unknown"


def main():
    DATA_DIR = Path("C:/Users/windlx/Projects/url-classifier/data")

    # ── 1. 合并数据源 ──────────────────────────────
    print("Loading data sources...")

    # 合成数据
    synthetic = build_synthetic_dataset()
    print(f"  Synthetic: {len(synthetic)} rows")

    # 真实数据
    json_files = [
        "urls_enhanced.json",
        "real_urls/labeled_urls.json",
        "real_urls/url_patterns_labeled.json",
    ]
    real = []
    for f in json_files:
        p = DATA_DIR / f
        if p.exists():
            real.extend(load_json(str(p)))
    print(f"  Real data: {len(real)} rows")

    # 合并
    all_data = synthetic + real

    # 规则清洗：用分类器纠正标签错误
    print("Cleaning labels with rule classifier...")
    cleaned = []
    for item in all_data:
        rule_label = classify_path(item["url"])
        if rule_label == "unknown":
            # 分类器不确定，信任原始标签
            cleaned.append(item)
        else:
            # 规则与标签不符 → 修正
            item["label"] = rule_label
            cleaned.append(item)

    # 去重
    seen = set()
    deduped = []
    for item in cleaned:
        if item["url"] not in seen:
            seen.add(item["url"])
            deduped.append(item)

    print(f"  Total after dedup: {len(deduped)} rows")

    # 标签分布
    label_counts = {}
    for item in deduped:
        label_counts[item["label"]] = label_counts.get(item["label"], 0) + 1
    print(f"  Label distribution: {label_counts}")

    # ── 2. 分割训练/测试 ──────────────────────────
    X = [item["url"] for item in deduped]
    y = [item["label"] for item in deduped]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # ── 3. TF-IDF Vectorizer (char n-gram) ────────
    print("\nBuilding TF-IDF vectorizer (char 3-6 gram)...")
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 6),
        max_features=100_000,
        lowercase=True,
        sublinear_tf=True,  # log(1+tf) — 减少高频词影响
    )
    t0 = time.time()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"  Vectorized in {time.time()-t0:.1f}s — shape: {X_train_vec.shape}")

    # ── 4. 训练 LogisticRegression ───────────────
    print("\nTraining LogisticRegression...")
    t0 = time.time()
    clf = LogisticRegression(
        C=10,
        max_iter=1000,
        solver="lbfgs",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train_vec, y_train)
    print(f"  Trained in {time.time()-t0:.1f}s")

    # ── 5. 评估 ───────────────────────────────────
    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== Test Accuracy: {acc:.4f} ===")
    print(classification_report(y_test, y_pred))

    # 交叉验证
    print("Running 5-fold cross-validation...")
    cv_scores = cross_val_score(clf, X_train_vec, y_train, cv=5, n_jobs=-1)
    print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Per-fold: {[f'{s:.4f}' for s in cv_scores]}")

    # ── 6. 推理速度 benchmark ─────────────────────
    print("\nInference speed benchmark...")
    n_bench = 100_000
    t0 = time.time()
    for i in range(n_bench):
        vec = vectorizer.transform([X_test[i % len(X_test)]])
        clf.predict(vec)
    elapsed = time.time() - t0
    rate = n_bench / elapsed
    print(f"  {n_bench} inferences in {elapsed:.2f}s")
    print(f"  Speed: {rate:,.0f} URLs/second")

    # ── 7. 保存模型 ────────────────────────────────
    import pickle
    MODEL_DIR = DATA_DIR / "models"
    MODEL_DIR.mkdir(exist_ok=True)

    model_path = MODEL_DIR / "url_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump((vectorizer, clf), f)
    print(f"\nModel saved: {model_path}")

    # ── 8. 手动测试 ────────────────────────────────
    print("\n=== Manual Tests ===")
    test_urls = [
        "https://bbc.com/news/technology-3439437565",
        "https://bbc.com/news/technology",
        "https://amazon.com/dp/B09V3KXJPB",
        "https://amazon.com/s?k=laptop",
        "https://arxiv.org/abs/2301.00001",
        "https://arxiv.org/list/cs/recent",
        "https://youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/feed/subscriptions",
        "https://tripadvisor.com/Hotel_Review-g1-d123456",
        "https://tripadvisor.com/Hotels-g1",
        "https://github.com/facebook/react",
        "https://github.com/search?q=fasttext",
        "https://pypi.org/project/requests",
        "https://pypi.org/simple/",
    ]

    for url in test_urls:
        vec = vectorizer.transform([url])
        pred = clf.predict(vec)[0]
        proba = clf.predict_proba(vec)[0]
        conf = max(proba)
        print(f"  [{pred:6s}] {conf:.2f}  {url}")


if __name__ == "__main__":
    main()
