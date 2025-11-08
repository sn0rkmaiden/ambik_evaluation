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

def _strip_fences(s: str) -> str:
    """Prefer content inside a single fenced block; otherwise remove stray fences."""
    s = s.replace("<eos>", "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    # remove lone fences if present
    s = re.sub(r"```(?:json)?", "", s, flags=re.I)
    s = s.replace("```", "")
    return s.strip()

def _extract_json_braces(raw: str) -> str | None:
    """Return the substring from first '{' to its matching '}' (brace-balanced, strings-aware)."""
    s = raw
    in_str = False
    esc = False
    depth = 0
    start = None
    for i, ch in enumerate(s):
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
                    return s[start:i+1]
    return None

def parse_model_json(raw_output: str) -> dict | None:
    """
    Robust JSON parse:
      - strip code fences / <eos>
      - extract the outermost {...}
      - repair: trailing commas, Python literals, and *insert missing ] before the final }*
    """
    raw_output = _strip_fences(raw_output)
    frag = _extract_json_braces(raw_output) or raw_output

    # 1) clean trailing commas & python literals
    s = frag.strip()
    s = re.sub(r',(\s*[}\]])', r'\1', s)   # remove trailing commas
    s = re.sub(r'\bTrue\b', 'true', s)
    s = re.sub(r'\bFalse\b', 'false', s)
    s = re.sub(r'\bNone\b', 'null', s)

    # 2) if we have more '[' than ']', insert the missing ones *before* the final closing brace
    def _counts(text):
        in_str = False; esc = False; ob=cb=0; osb=csb=0
        for ch in text:
            if ch == '"' and not esc: in_str = not in_str
            esc = (ch == '\\' and not esc) if in_str else False
            if in_str: continue
            if ch == '{': ob += 1
            elif ch == '}': cb += 1
            elif ch == '[': osb += 1
            elif ch == ']': csb += 1
        return ob, cb, osb, csb

    ob, cb, osb, csb = _counts(s)
    missing_rb = osb - csb
    if missing_rb > 0:
        # insert the missing ] right before the last non-space '}' so we end with ..."]} not ..."}]
        j = len(s) - 1
        while j >= 0 and s[j].isspace():
            j -= 1
        if j >= 0 and s[j] == '}':
            s = s[:j] + (']' * missing_rb) + s[j:]
        else:
            s = s + (']' * missing_rb)

    # 3) try strict JSON
    try:
        return json.loads(s)
    except Exception:
        # 4) last resort: python-literal eval
        pyish = (s.replace('null', 'None')
                   .replace('true', 'True')
                   .replace('false', 'False'))
        try:
            val = ast.literal_eval(pyish)
            # ensure it's a dict
            return val if isinstance(val, dict) else None
        except Exception:
            return None
