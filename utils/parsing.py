import json, re, ast

# def extract_json_balanced(raw: str) -> str | None:
#     """
#     Return the first balanced {...} block (ignoring braces inside strings).
#     """
#     in_str = False
#     esc = False
#     depth = 0
#     start = None
#     for i, ch in enumerate(raw):
#         if ch == '"' and not esc:
#             in_str = not in_str
#         esc = (ch == '\\' and not esc) if in_str else False
#         if in_str:
#             continue
#         if ch == '{':
#             if depth == 0:
#                 start = i
#             depth += 1
#         elif ch == '}':
#             if depth > 0:
#                 depth -= 1
#                 if depth == 0 and start is not None:
#                     return raw[start:i+1]
#     return None

# def repair_json_string(s: str) -> str:
#     """
#     Try to repair common issues:
#     - trailing commas before } or ]
#     - unbalanced [] or {}
#     - Python booleans/None
#     """
#     s = s.strip()
#     # remove trailing commas like {"a":1,}
#     s = re.sub(r',(\s*[}\]])', r'\1', s)
#     # normalize True/False/None
#     s = re.sub(r'\bTrue\b', 'true', s)
#     s = re.sub(r'\bFalse\b', 'false', s)
#     s = re.sub(r'\bNone\b', 'null', s)

#     # balance braces/brackets ignoring strings
#     def balance(text, open_ch, close_ch):
#         in_str = False
#         esc = False
#         count = 0
#         for ch in text:
#             if ch == '"' and not esc:
#                 in_str = not in_str
#             esc = (ch == '\\' and not esc) if in_str else False
#             if in_str:
#                 continue
#             if ch == open_ch:
#                 count += 1
#             elif ch == close_ch and count > 0:
#                 count -= 1
#         return text + (close_ch * count)

#     s = balance(s, '{', '}')
#     s = balance(s, '[', ']')
#     return s

# def parse_model_json(raw_output: str) -> dict | None:
#     """
#     Best-effort JSON parse:
#     1) balanced extraction
#     2) strict json.loads
#     3) repair and retry json.loads
#     4) last resort: ast.literal_eval after mapping true/false/null to Python
#     """
#     frag = extract_json_balanced(raw_output) or raw_output.strip()
#     fixed = repair_json_string(frag)
#     try:
#         return json.loads(fixed)
#     except Exception:
#         try:
#             pyish = (fixed
#                      .replace('null', 'None')
#                      .replace('true', 'True')
#                      .replace('false', 'False'))
#             return ast.literal_eval(pyish)
#         except Exception:
#             return None

# --- robust_json.py (or paste into runner.py) ---
import re, json, ast

def _strip_fences(s: str) -> str:
    """Prefer the fenced block with braces; otherwise remove lone fences and <eos>."""
    s = s.replace("<eos>", "").strip()

    # Prefer a fenced block that actually contains JSON braces
    best = None
    for m in re.finditer(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.S | re.I):
        chunk = m.group(1).strip()
        if "{" in chunk and "}" in chunk:
            best = chunk
            break
    if best is not None:
        return best

    # If no fenced block, strip lone fences and return whole string
    s = re.sub(r"```(?:json)?", "", s, flags=re.I)
    s = s.replace("```", "")
    return s.strip()

def _extract_balanced_json_candidates(text: str):
    """Yield every balanced {...} substring (strings-aware)."""
    in_str = False
    esc = False
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '"' and not esc:
            in_str = not in_str
        esc = (ch == '\\' and not esc) if in_str else False
        if in_str:
            continue
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start:i+1]

def _repair_common(s: str) -> str:
    # remove trailing commas like {"a":1,} or ["x",]
    s = re.sub(r',(\s*[}\]])', r'\1', s)
    # normalize Python-ish literals
    s = re.sub(r'\bTrue\b', 'true', s)
    s = re.sub(r'\bFalse\b', 'false', s)
    s = re.sub(r'\bNone\b', 'null', s)
    # balance ] if array brackets are short by a few
    def _counts(t):
        in_str = False; esc = False; ob=cb=osb=csb=0
        for ch in t:
            if ch == '"' and not esc: in_str = not in_str
            esc = (ch == '\\' and not esc) if in_str else False
            if in_str: continue
            if ch == '{': ob += 1
            elif ch == '}': cb += 1
            elif ch == '[': osb += 1
            elif ch == ']': csb += 1
        return ob, cb, osb, csb
    _, _, osb, csb = _counts(s)
    missing_rb = osb - csb
    if missing_rb > 0:
        # insert missing ']' before the final closing '}'
        j = len(s) - 1
        while j >= 0 and s[j].isspace(): j -= 1
        if j >= 0 and s[j] == '}':
            s = s[:j] + (']' * missing_rb) + s[j:]
        else:
            s = s + (']' * missing_rb)
    return s

def parse_model_json(raw_output: str) -> dict | None:
    """
    Try: strip fences -> extract first balanced {...} -> repair -> json.loads
    Fallback: ast.literal_eval with Python tokens.
    """
    body = _strip_fences(raw_output)
    candidates = list(_extract_balanced_json_candidates(body))
    if not candidates and "{" in body and "}" in body:
        candidates = [body[body.find("{"): body.rfind("}")+1]]

    for frag in candidates or [body]:
        s = _repair_common(frag.strip())
        try:
            val = json.loads(s)
            if isinstance(val, dict):
                return val
        except Exception:
            pyish = (s.replace('null', 'None')
                       .replace('true', 'True')
                       .replace('false', 'False'))
            try:
                val = ast.literal_eval(pyish)
                if isinstance(val, dict):
                    return val
            except Exception:
                continue
    return None

