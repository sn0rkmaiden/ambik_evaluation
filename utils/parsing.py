import json, re, ast

def extract_json_balanced(raw: str) -> str | None:
    """
    Return the first balanced {...} block (ignoring braces inside strings).
    """
    in_str = False
    esc = False
    depth = 0
    start = None
    for i, ch in enumerate(raw):
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
                    return raw[start:i+1]
    return None

def repair_json_string(s: str) -> str:
    """
    Try to repair common issues:
    - trailing commas before } or ]
    - unbalanced [] or {}
    - Python booleans/None
    """
    s = s.strip()
    # remove trailing commas like {"a":1,}
    s = re.sub(r',(\s*[}\]])', r'\1', s)
    # normalize True/False/None
    s = re.sub(r'\bTrue\b', 'true', s)
    s = re.sub(r'\bFalse\b', 'false', s)
    s = re.sub(r'\bNone\b', 'null', s)

    # balance braces/brackets ignoring strings
    def balance(text, open_ch, close_ch):
        in_str = False
        esc = False
        count = 0
        for ch in text:
            if ch == '"' and not esc:
                in_str = not in_str
            esc = (ch == '\\' and not esc) if in_str else False
            if in_str:
                continue
            if ch == open_ch:
                count += 1
            elif ch == close_ch and count > 0:
                count -= 1
        return text + (close_ch * count)

    s = balance(s, '{', '}')
    s = balance(s, '[', ']')
    return s

def parse_model_json(raw_output: str) -> dict | None:
    """
    Best-effort JSON parse:
    1) balanced extraction
    2) strict json.loads
    3) repair and retry json.loads
    4) last resort: ast.literal_eval after mapping true/false/null to Python
    """
    frag = extract_json_balanced(raw_output) or raw_output.strip()
    fixed = repair_json_string(frag)
    try:
        return json.loads(fixed)
    except Exception:
        try:
            pyish = (fixed
                     .replace('null', 'None')
                     .replace('true', 'True')
                     .replace('false', 'False'))
            return ast.literal_eval(pyish)
        except Exception:
            return None
