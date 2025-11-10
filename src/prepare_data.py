import os, re, requests, pandas as pd
from selectolax.parser import HTMLParser

os.makedirs("data", exist_ok=True)
df = pd.read_csv("data/labeled_train.csv")
urls = sorted(df["Assessment_url"].dropna().drop_duplicates().tolist())

def get(u):
    try:
        r = requests.get(u, headers={"User-Agent":"Mozilla/5.0"}, timeout=20)
        if r.status_code==200: return r.text
    except Exception: pass
    return ""

def clean(s):
    if not s: return ""
    return re.sub(r"\s+"," ",s).strip()

def meta(tree, k):
    n = tree.css_first(f"meta[name='{k}']") or tree.css_first(f"meta[property='{k}']")
    return n.attributes.get("content","") if n and "content" in n.attributes else ""

def guess_type(t):
    t=t.lower()
    if any(x in t for x in ["personality","behavior","behaviour","values","motivation","work style","situational judgement","sjt"]): return "P"
    if any(x in t for x in ["skill","knowledge","ability","aptitude","numerical","verbal","inductive","coding","programming","excel","sql","python","technical"]): return "K"
    return ""

rows=[]
for u in urls:
    html = get(u)
    if not html: 
        rows.append({"name":"","url":u,"description":"","test_type":"","text":u})
        continue
    tree = HTMLParser(html)
    h1 = tree.css_first("h1")
    title = clean(h1.text()) if h1 else ""
    if not title:
        title = clean(meta(tree,"og:title") or meta(tree,"twitter:title"))
    desc=""
    for sel in ["article",".entry-content",".content",".product-content",".elementor-widget-container","main","body"]:
        node = tree.css_first(sel)
        if node:
            desc = clean(node.text())
            if len(desc)>40: break
    if not desc:
        desc = clean(meta(tree,"description") or meta(tree,"og:description") or meta(tree,"twitter:description"))
    tt = guess_type(f"{title} {desc}")
    text = clean(f"{title}. {desc}") if title or desc else u
    rows.append({"name":title,"url":u,"description":desc,"test_type":tt,"text":text})

cat = pd.DataFrame(rows)
cat.drop_duplicates(subset=["url"], inplace=True)
cat.to_parquet("data/catalog.parquet", index=False)
print("wrote data/catalog.parquet", len(cat))
