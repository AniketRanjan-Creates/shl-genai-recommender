import re, urllib.parse

def canon_url(u:str)->str:
    if not u: return ""
    u = u.strip().replace("http://","https://").split("?")[0].rstrip("/")
    u = urllib.parse.unquote(u)
    u = u.replace("://www.shl.com/products/product-catalog/", "://www.shl.com/solutions/products/product-catalog/")
    return u

def extract_duration_req(q:str):
    q = (q or "").lower()
    m = re.search(r"(\d+)\s*-\s*(\d+)\s*min", q)
    if m: return int(m.group(2))
    m = re.search(r"(\d+)\s*min", q)
    if m: return int(m.group(1))
    m = re.search(r"(\d+)\s*-\s*(\d+)\s*hour", q)
    if m: return int(m.group(2))*60
    m = re.search(r"(\d+)\s*hour", q)
    if m: return int(m.group(1))*60
    return None

def page_duration_minutes(text:str):
    t = (text or "").lower()
    m = re.search(r"(\d+)\s*-\s*(\d+)\s*min", t)
    if m: return int(m.group(2))
    m = re.search(r"(\d+)\s*min", t)
    if m: return int(m.group(1))
    m = re.search(r"(\d+)\s*hour", t)
    if m: return int(m.group(1))*60
    return None

SOFT_SKILL_HINTS = [
    "collaborat","stakeholder","communication","interpersonal","behav","personality",
    "values","motivation","culture","leadership","team","opq","sjt","situational"
]

def query_implies_P(query:str)->bool:
    q=(query or "").lower()
    return any(h in q for h in SOFT_SKILL_HINTS)
