import re

# Approximate fractional positions in a standard lumbar-spine MRI volume.
# 0.0 = superior (top of stack), 1.0 = inferior (bottom).
# Coverage assumed: roughly T12 to S1.
_LEVEL_FRACTIONS = {
    "T12":    0.05,
    "T12-L1": 0.12,
    "L1":     0.20,
    "L1-L2":  0.28,
    "L2":     0.35,
    "L2-L3":  0.43,
    "L3":     0.50,
    "L3-L4":  0.57,
    "L4":     0.63,
    "L4-L5":  0.70,
    "L5":     0.77,
    "L5-S1":  0.85,
    "S1":     0.91,
}

_LEVEL_RE = re.compile(
    r'\b(T12[-–]L1|L[1-5][-–][LS][1-5]|L[1-5]|T12|S1)\b',
    re.IGNORECASE,
)


def extract_spinal_levels(text):
    """Return unique spinal-level labels found in text, in order of appearance."""
    seen = set()
    result = []
    for match in _LEVEL_RE.finditer(text):
        key = match.group().upper().replace('–', '-')
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


def level_to_slice_index(level, total_slices):
    """Convert a spinal-level label to the nearest axial Z-index in the volume."""
    key = level.upper().replace('–', '-')
    fraction = _LEVEL_FRACTIONS.get(key, 0.5)
    return max(0, min(total_slices - 1, int(fraction * total_slices)))


def build_level_finding_map(section_texts):
    """
    Parse the LLM-generated report sections and return:
        {level_label: [short_finding_string, ...]}
    Only includes levels with an explicit mention in the clinical sections.
    """
    relevant = [
        "DISC ASSESSMENT",
        "FORAMINAL ASSESSMENT",
        "SPINAL CANAL & THECAL SAC",
        "FACET JOINTS",
        "IMPRESSION",
    ]
    combined = "\n".join(section_texts.get(k, "") for k in relevant)

    level_map = {}
    for sent in re.split(r'[.;\n]', combined):
        sent = sent.strip()
        if len(sent) < 10:
            continue
        for lev in extract_spinal_levels(sent):
            level_map.setdefault(lev, [])
            if len(level_map[lev]) < 3:
                level_map[lev].append(sent[:100])

    return level_map


LEVEL_FRACTIONS = _LEVEL_FRACTIONS


def get_levels_from_sir(sir_json):
    """Extract spinal-level keys from a SIR JSON dict (e.g. {'L1-L2': {...}, 'L4-L5': {...}})."""
    levels = []
    for key in sir_json:
        clean = key.upper().replace('–', '-')
        if _LEVEL_RE.match(clean):
            levels.append(clean)
    return sorted(set(levels), key=lambda l: _LEVEL_FRACTIONS.get(l, 0.5))
