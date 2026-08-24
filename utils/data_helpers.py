import ast

def load_id2name_from_txt(path):
    """
    Expects a Python dict literal in the file, e.g.:
    {0: 'tench, Tinca tinca', 1: 'goldfish, Carassius auratus', ...}
    Returns a list `id2name` where id2name[i] is the raw alias string.
    """
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    data = ast.literal_eval(txt)  # safe parse for Python literals

    if not isinstance(data, dict):
        raise ValueError("id2name txt must contain a dict literal of {int: str}")

    max_id = max(int(k) for k in data.keys())
    id2name = [None] * (max_id + 1)
    for k, v in data.items():
        i = int(k)
        if not isinstance(v, str):
            raise ValueError(f"id2name[{i}] must be a string, got {type(v)}")
        id2name[i] = v

    # sanity: no holes
    if any(x is None for x in id2name):
        missing = [i for i, x in enumerate(id2name) if x is None]
        raise ValueError(f"id2name mapping has holes at indices: {missing[:10]}...")

    return id2name

def canonicalize_alias(alias_str, policy="first"):
    """
    alias_str like: 'great white shark, white shark, man-eater, ...'
    - policy='first': return the first alias (before a comma), trimmed.
    - policy='full': return the full string (as-is).
    """
    s = alias_str.strip()
    if policy == "full":
        return s
    # 'first'
    # split on comma and take the first human-friendly alias
    first = s.split(",")[0].strip()
    return first