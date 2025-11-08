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

import re, json, ast

# -------------------- helpers to sanitize the raw text --------------------

def _strip_fences_and_eos(s: str) -> str:
    """Remove ``` fences, optional ```json, and <eos>. Prefer fenced JSON if present."""
    s = s.replace("<eos>", "").strip()

    # Prefer a fenced block that actually contains braces
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", s, flags=re.I)
    if m:
        return m.group(1).strip()

    # Otherwise remove any stray fences and return whole
    s = re.sub(r"```(?:json)?", "", s, flags=re.I)
    s = s.replace("```", "")
    return s.strip()

def _first_curly_to_end(s: str) -> str | None:
    """Return text starting from first '{' (inclusive)."""
    i = s.find("{")
    return s[i:] if i != -1 else None

# -------------------- main tolerant parser --------------------

def parse_model_json(raw_output: str) -> dict | None:
    """
    Robust, schema-aware parse for:
      {
        "ambiguous": true|false,
        "question": [ "...", "...", ... ]
      }

    Tolerates:
    - pre/post text (instructions), code fences, <eos>
    - trailing commas, Python literals (True/False/None)
    - missing closing ']' and/or '}' (truncated output)
    - last list item missing a closing quote
    """
    # 0) strip noise and get from first brace onward
    body = _strip_fences_and_eos(raw_output)
    body = _first_curly_to_end(body) or body

    # 1) Try strict JSON first
    try:
        obj = json.loads(body)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # 2) Quick repairs: trailing commas, python literals
    repaired = re.sub(r',(\s*[}\]])', r'\1', body)  # trailing commas
    repaired = re.sub(r'\bTrue\b', 'true', repaired)
    repaired = re.sub(r'\bFalse\b', 'false', repaired)
    repaired = re.sub(r'\bNone\b', 'null', repaired)

    # 3) If braces/brackets are unbalanced, try to heal by appending the missing closers.
    repaired = _balance_closers(repaired)

    # 4) Try JSON again
    try:
        obj = json.loads(repaired)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 5) Schema-guided extraction (works even if truncated mid-string)
    return _schema_parse_fallback(repaired)

# -------------------- balancing + fallback --------------------

def _balance_closers(s: str) -> str:
    """Append missing ] or } at the end if counts don't match (string-aware)."""
    in_str = False
    esc = False
    braces = brackets = 0
    for ch in s:
        if ch == '"' and not esc:
            in_str = not in_str
        esc = (ch == '\\' and not esc) if in_str else False
        if in_str:
            continue
        if ch == '{': braces += 1
        elif ch == '}': braces -= 1 if braces > 0 else 0
        elif ch == '[': brackets += 1
        elif ch == ']': brackets -= 1 if brackets > 0 else 0

    s = s.rstrip()
    if brackets > 0:
        s += ']' * brackets
    if braces > 0:
        s += '}' * braces
    return s

def _schema_parse_fallback(text: str) -> dict | None:
    """
    Last-resort: pull "ambiguous" and "question" with a tolerant scanner.
    Handles truncated last string in the list.
    """
    # ambiguous: true/false
    m = re.search(r'"ambiguous"\s*:\s*(true|false)', text, flags=re.I)
    if m:
        ambiguous = (m.group(1).lower() == "true")
    else:
        # default to False if missing
        ambiguous = False

    # question: [ ... ]
    q_start = re.search(r'"question"\s*:\s*\[', text, flags=re.I)
    questions = []
    if q_start:
        i = q_start.end()
        questions, _ = _scan_string_list(text, i)

    return {"ambiguous": ambiguous, "question": questions}

def _scan_string_list(s: str, i: int):
    """
    Parse a JSON-like list of strings starting at index i (first char AFTER '[').
    Tolerates missing closing quote or ']' and basic escapes.
    Returns (list, end_index).
    """
    items = []
    n = len(s)
    cur = []
    in_str = False
    esc = False

    while i < n:
        ch = s[i]

        if in_str:
            if esc:
                cur.append(ch)
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
                items.append(''.join(cur))
                cur = []
            else:
                cur.append(ch)
            i += 1
            continue

        # not in string
        if ch == '"':
            in_str = True
            esc = False
            i += 1
            continue
        if ch == ',':
            # separator between items; if we had a truncated string w/o closing quote, flush it
            if cur:
                items.append(''.join(cur))
                cur = []
            i += 1
            continue
        if ch == ']':
            # end of list
            if cur:
                items.append(''.join(cur))
                cur = []
            i += 1
            break
        if ch == '{' or ch == '}':
            # list ended abruptly (truncated before ']')
            if cur:
                items.append(''.join(cur))
            break

        # skip whitespace/other
        i += 1

    # strip whitespace around items
    items = [x.strip() for x in items if x is not None]
    # Remove empty strings created by truncation noise
    items = [x for x in items if x != ""]
    return items, i


