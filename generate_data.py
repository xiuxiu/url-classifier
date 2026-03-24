"""
Generate diverse URL training data for the classifier.
Expands URL templates with realistic path segments.

Usage:
    python generate_data.py                       # generate to data/urls_diverse.json
    python generate_data.py -o data/xxx.json    # custom output
    python generate_data.py --convert            # also convert to train.json format
"""

import os, json, random, argparse
from pathlib import Path

random.seed(42)

# ---------------------------------------------------------------------------
# URL Templates: (list_patterns, detail_patterns, base_url)
# ---------------------------------------------------------------------------

TEMPLATES = [
    # E-commerce
    ("amazon.com", "https://www.amazon.com", [
        ("list", "{base}/s?k={w}", "{base}/s?k={w}&page={p}", "{base}/gp/site-directory",
         "{base}/stores/{brand}/page/{p}", "{base}/deal/{did}"),
        ("detail", "{base}/dp/{asin}", "{base}/gp/product/{asin}"),
    ]),
    ("jd.com", "https://www.jd.com", [
        ("list", "{base}/category", "{base}/list/{cat}", "{base}/search?search={w}",
         "{base}/book/{cat}", "{base}/channel/{cat}"),
        ("detail", "{base}/item/{id}.html", "{base}/product/{id}", "{base}/{id}.html"),
    ]),
    ("taobao.com", "https://www.taobao.com", [
        ("list", "{base}/search?q={w}", "{base}/category", "{base}/home.htm", "{base}/market/{cat}"),
        ("detail", "{base}/item.htm?id={id}", "{base}/product/{id}.htm", "{base}/detail/{id}"),
    ]),
    ("天猫", "https://www.tmall.com", [
        ("list", "{base}/search.htm?q={w}", "{base}/list/{cat}.htm", "{base}/category/{cat}"),
        ("detail", "{base}/product/{id}.htm", "{base}/item/{id}.htm"),
    ]),
    ("拼多多", "https://www.pinduoduo.com", [
        ("list", "{base}/search.html?keyword={w}", "{base}/category.html"),
        ("detail", "{base}/home.html?id={id}", "{base}/goods/{id}"),
    ]),

    # Recruitment
    ("智联招聘", "https://www.zhaopin.com", [
        ("list", "{base}/jobs/list?KEYWORD={w}", "{base}/search/p{p}?KEYWORD={w}",
         "{base}/jobs/?key={w}", "{base}/s/{p}/?keyword={w}"),
        ("detail", "{base}/job/{jid}.html", "{base}/jobs/{cid}/{jid}.htm"),
    ]),
    ("BOSS直聘", "https://www.zhipin.com", [
        ("list", "{base}/search/job?query={w}", "{base}/zhaopin/{cat}", "{base}/job_list/?keyword={w}"),
        ("detail", "{base}/job/{jid}.html", "{base}/zhipin/{user}/{jid}.html"),
    ]),
    ("拉勾", "https://www.lagou.com", [
        ("list", "{base}/jobs-{city}-{w}-p{p}.html", "{base}/job/index?city={city}&kd={w}"),
        ("detail", "{base}/job/{jid}.html", "{base}/jobs/{user}/job_{jid}.html"),
    ]),

    # News
    ("sina.com.cn", "https://news.sina.com.cn", [
        ("list", "{base}/news/{cat}", "{base}/search?q={w}", "{base}/roll"),
        ("detail", "{base}/{y}/{m}/{id}.shtml", "{base}/news/{y}/{m}/{id}.htm"),
    ]),
    ("网易新闻", "https://news.163.com", [
        ("list", "{base}/news/{cat}", "{base}/search/search.htm?keyword={w}", "{base}/head/{cat}"),
        ("detail", "{base}/{y}/{m}/{id}.htm", "{base}/article/{id}.htm"),
    ]),
    ("腾讯新闻", "https://news.qq.com", [
        ("list", "{base}/news/{cat}", "{base}/search?q={w}"),
        ("detail", "{base}/detail/{y}/{m}/{id}.shtml", "{base}/news/{id}.htm"),
    ]),
    ("36kr.com", "https://36kr.com", [
        ("list", "{base}/news/{cat}", "{base}/search?keyword={w}", "{base}/information/{cat}"),
        ("detail", "{base}/news/{id}", "{base}/article/{id}"),
    ]),

    # Social / Q&A
    ("知乎", "https://www.zhihu.com", [
        ("list", "{base}/search?type=content&q={w}", "{base}/explore", "{base}/topics"),
        ("detail", "{base}/question/{id}", "{base}/article/{id}", "{base}/people/{user}/answers"),
    ]),
    ("豆瓣", "https://www.douban.com", [
        ("list", "{base}/search?q={w}", "{base}/tag/{cat}", "{base}/group/{cat}/board"),
        ("detail", "{base}/subject/{id}", "{base}/people/{user}", "{base}/group/topic/{id}"),
    ]),
    ("小红书", "https://www.xiaohongshu.com", [
        ("list", "{base}/search?keyword={w}", "{base}/discovery/search?s={w}"),
        ("detail", "{base}/explore/{id}", "{base}/note/{id}"),
    ]),
    ("Reddit", "https://www.reddit.com", [
        ("list", "{base}/r/{sub}", "{base}/r/{sub}/hot", "{base}/r/{sub}/new",
         "{base}/r/search?q={w}&type=sr_name"),
        ("detail", "{base}/r/{sub}/comments/{pid}", "{base}/r/{sub}/comments/{pid}/_/{cid}"),
    ]),

    # Video
    ("youtube.com", "https://www.youtube.com", [
        ("list", "{base}/results?search_query={w}", "{base}/@{user}/videos",
         "{base}/playlist?list={plid}"),
        ("detail", "{base}/watch?v={vid}", "{base}/shorts/{sid}"),
    ]),
    ("Bilibili", "https://www.bilibili.com", [
        ("list", "{base}/video/{bvid}?p={p}", "{base}/rank/{cat}", "{base}/popular/history",
         "{base}/search?keyword={w}"),
        ("detail", "{base}/video/{bvid}", "{base}/av{avid}"),
    ]),

    # Travel
    ("携程", "https://you.ctrip.com", [
        ("list", "{base}/list/{city}-page{p}.htm?k={w}", "{base}/tour/{cat}", "{base}/search?keyword={w}"),
        ("detail", "{base}/tours/{tid}", "{base}/hotel/{hid}.htm", "{base}/flight/{fid}"),
    ]),
    ("去哪儿", "https://www.qunar.com", [
        ("list", "{base}/list/{cat}?search={w}", "{base}/hotel/{city}"),
        ("detail", "{base}/hotel_{hid}.html", "{base}/flight_{fid}.html", "{base}/tour_{tid}.html"),
    ]),
    ("马蜂窝", "https://www.mafengwo.cn", [
        ("list", "{base}/travel-scenic/{city}", "{base}/search?keyword={w}", "{base}/notes/{cat}"),
        ("detail", "{base}/notes/{id}.html", "{base}/destination/{did}"),
    ]),

    # Education
    ("中国大学MOOC", "https://www.icourse163.org", [
        ("list", "{base}/course-list?page={p}&s={w}", "{base}/search/?q={w}",
         "{base}/learn/{cid}/"),
        ("detail", "{base}/course/{cid}", "{base}/info/{cid}"),
    ]),
    ("慕课网", "https://www.imooc.com", [
        ("list", "{base}/courses?sort=popular&page={p}", "{base}/search?q={w}",
         "{base}/course/list?tag={w}"),
        ("detail", "{base}/course/{cid}", "{base}/learn/{cid}"),
    ]),

    # Tech / Docs
    ("GitHub", "https://github.com", [
        ("list", "{base}/search?q={w}&type=repositories", "{base}/explore", "{base}/topics"),
        ("detail", "{base}/{user}/{repo}", "{base}/{user}/{repo}/tree/{branch}/{path}"),
    ]),
    ("ReadTheDocs", "https://docs.readthedocs.io", [
        ("list", "{base}/en/latest/", "{base}/en/latest/search.html?q={w}",
         "{base}/en/latest/{cat}/"),
        ("detail", "{base}/en/latest/{slug}.html", "{base}/en/latest/{path}/{page}.html"),
    ]),
    ("MDN", "https://developer.mozilla.org", [
        ("list", "{base}/en-US/search?q={w}", "{base}/en-US/docs/Web"),
        ("detail", "{base}/en-US/docs/{path}/{page}", "{base}/en-US/docs/Web/API/{name}"),
    ]),
]

# ---------------------------------------------------------------------------
# Value pools
# ---------------------------------------------------------------------------

WORDS = ["laptop","phone","book","shoes","headphone","camera","monitor","computer",
         "tablet","keyboard","mouse","watch","speaker","coffee","restaurant",
         "python","javascript","rust","golang","react","vue","docker","kubernetes",
         "machine-learning","ai","data-science","backend","frontend","cloud",
         "security","database","travel","hotel","flight","car","restaurant"]

CATS = ["electronics","books","clothing","home","kitchen","sports","toys",
        "beauty","health","automotive","food","music","games","travel"]

BRANDS = ["huawei","xiaomi","apple","samsung","sony","lenovo","dell"]
CITIES = ["beijing","shanghai","hangzhou","shenzhen","guangzhou","chengdu"]
YEARS = [2023,2024,2025]
MONTHS = ["01","02","03","04","05","06","07","08"]
USERS = ["john-dev","alice-code","bob-ml","carol-data","dave-ops"]
REPOS = ["awesome-project","python-scripts","ml-tools","data-pipeline","web-app","api-server","blog-src","docs-repo"]


def expand(url: str, base: str) -> str:
    """Fill in URL template with random values."""
    return url.format(
        base=base,
        w=random.choice(WORDS),
        cat=random.choice(CATS),
        p=random.randint(1,10),
        repo=random.choice(REPOS),
    did=f"deal-{random.randint(1000,9999)}",
        asin="".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",k=10)),
        id=random.randint(100000,999999),
        jid=random.randint(1000000,9999999),
        cid=f"c{random.randint(10000,99999)}",
        user=random.choice(USERS),
        city=random.choice(CITIES),
        brand=random.choice(BRANDS),
        y=random.choice(YEARS),
        m=random.choice(MONTHS),
        sub=random.choice(["python","programming","technology","science","worldnews","movies"]),
        pid=random.randint(10000000,99999999),
        cid2=random.randint(10000000,99999999),
        vid="".join(random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-",k=11)),
        sid=random.randint(10**8,10**9-1),
        bvid=f"BV{random.randint(10**9,10**10-1)}",
        avid=random.randint(10000000,99999999),
        plid=random.choice(["PLrMZB","PLkQuL","PLJgod","PLZfid"]),
        tid=f"tour-{random.randint(1000,9999)}",
        hid=f"hotel-{random.randint(100000,999999)}",
        fid=f"flight-{random.randint(100000,999999)}",
        did2=f"dest-{random.randint(1000,9999)}",
        did3=random.randint(1000,9999),
        cid2b=f"course-{random.randint(1000,9999)}",
        cid3=f"course-{random.randint(1000,9999)}",
        slug=random.choice(CATS),
        path=random.choice(CATS),
        page=random.choice(CATS),
        name=random.choice(WORDS),
        branch="main",
    )


def generate(count_per_type: int = 50) -> list[dict]:
    """Generate all URL-label pairs."""
    records = []

    for domain, base, patterns in TEMPLATES:
        list_patterns, detail_patterns = [], []
        for entry in patterns:
            ptype, *urls = entry
            if ptype == "list":
                list_patterns.extend(urls)
            else:
                detail_patterns.extend(urls)

        for _ in range(count_per_type):
            records.append({
                "text": expand(random.choice(list_patterns), base),
                "label": 0,   # list page
            })
            records.append({
                "text": expand(random.choice(detail_patterns), base),
                "label": 1,   # detail page
            })

    random.shuffle(records)
    return records


def to_conversation_format(records: list[dict]) -> list[dict]:
    """Convert to the conversation format that prepare.py expects."""
    conversation = []
    for r in records:
        label_str = "A" if r["label"] == 0 else "B"
        conversation.append({
            "text": f"user\nURL:\n\n{r['text']}\n<|im_end|>\n<|im_start|>assistant\n{label_str}\n<|im_end|>",
            "label": r["label"]
        })
    return conversation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="data/urls_diverse.json")
    ap.add_argument("-n", "--count", type=int, default=50,
                    help="URLs per type per domain (default: 50)")
    ap.add_argument("--convert", action="store_true",
                    help="Also save as train.json conversation format")
    args = ap.parse_args()

    print(f"=== Generating diverse URL training data ===")
    print(f"Domains: {len(TEMPLATES)}")
    print(f"URLs per type per domain: {args.count}")
    print()

    records = generate(args.count)

    # Save as raw URL JSON
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(records)} URLs → {args.output}")

    if args.convert:
        train_path = args.output.replace(".json", "_train.json")
        conv = to_conversation_format(records)
        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(conv, f, ensure_ascii=False, indent=2)
        print(f"Saved conversation format → {train_path}")

    # Stats
    a = sum(1 for r in records if r["label"] == 0)
    b = sum(1 for r in records if r["label"] == 1)
    print(f"\nTotal: {len(records)} | A(list)={a} | B(detail)={b}")
    print(f"Domains: {[d for d,_,_ in TEMPLATES]}")


if __name__ == "__main__":
    main()
