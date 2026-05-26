"""
PaperForge backend — Rule-Based Formatter edition (no auto-crop)
=================================================================
DESIGN: AI extracts structured data + picks from FIXED MENUS only.
        Deterministic code owns 100% of LaTeX formatting.
        Figures are PLACEHOLDERS only — the AI marks where a figure goes;
        the user supplies the cropped images (1.png, 2.png, ... global order).

POST /convert  ->  image(s) + questionNumber  =>  { latex, totalMarks, structured, figureCount }
POST /export   ->  full paper state            =>  zip of paper.tex (+ any uploaded images)
GET  /         ->  serves index.html
GET  /health   ->  status
"""

import os, io, json, zipfile, re
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from PIL import Image
import google.generativeai as genai

MODEL = "gemini-2.5-flash"
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

app = FastAPI(title="PaperForge")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
HERE = os.path.dirname(os.path.abspath(__file__))

@app.get("/")
def home():
    index = os.path.join(HERE, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"ok": True, "model": MODEL}

# stores any images the user uploads (optional — they can also supply files at compile time)
FIGURE_STORE = {}

# ============================================================================
# EXTRACTION PROMPT — model extracts data + picks from FIXED MENUS only.
# Figures: model only marks WHERE a figure goes (no cropping, no coordinates).
# ============================================================================
SYSTEM_PROMPT = r"""
You read a screenshot of ONE exam question and extract its STRUCTURE as JSON.
You do NOT write any LaTeX layout. You only extract content and PICK from fixed menus.

Return STRICT VALID JSON:
{
  "intro": "<text before the first sub-part, or '' >",
  "intro_breaks": [],
  "parts": [
    {
      "level": 0,
      "text": "<full text of this part>",
      "breaks": [{"after_sentence": <int index>, "type": "tight|para|double"}],
      "marks": <integer or null>,
      "lines_visible": false,
      "answer_rows": [
        {"label": "<text label before the line, or '' >",
         "has_equals": false,
         "unit": "<unit after the line as printed, e.g. 'µm','m/s', or '' >",
         "unit_siunitx": "<the SAME unit written with siunitx macros, or '' >",
         "width": "standard|narrow|wide",
         "indent": false}
      ],
      "answer_kind": "none|coordinates",
      "figure_here": false,
      "figure_position": "before|after",
      "figure_size": "small|medium|large",
      "is_table": false,
      "table": {"has_header": true, "headers": ["..."], "rows": [["..."]]},
      "is_mcq": false,
      "mcq_options": ["..."],
      "bullets": []
    }
  ],
  "totalMarks": <integer or null>
}

=== EXTRACTION RULES ===

CONTENT:
- Copy ALL text EXACTLY. Preserve wording, numbers, punctuation, symbols.
- Do NOT include the question's own number (like "2").
- Write maths inline using \( \): e.g. \( x^2+3x \), \( 105^\circ \), \( \dfrac{a}{b} \).
- Keep units as written ("220 m", "0.70 s") — the system handles spacing.

UNITS (siunitx vocabulary — for "unit_siunitx" fields):
- Whenever you report a unit, ALSO give its siunitx form. siunitx is a LaTeX package
  with a named macro for every SI prefix and unit; the system wraps your string in
  \unit{...} and it renders perfectly. You do NOT write \unit{} yourself — only its body.
- Prefix macros: \quecto \ronto \yocto \zepto \atto \femto \pico \nano \micro \milli
  \centi \deci \deca \hecto \kilo \mega \giga \tera \peta \exa \zetta \yotta
- Unit macros: \metre \gram \second \ampere \kelvin \mole \candela \hertz \newton
  \pascal \joule \watt \coulomb \volt \farad \ohm \siemens \weber \tesla \henry
  \celsius \lumen \lux \becquerel \gray \sievert \katal \litre \electronvolt \degree
  \radian \steradian \tonne \angstrom \hour \minute \day \bar \percent
- Division: use \per before the denominator unit. Powers: \squared \cubed, or
  \tothe{n}. Prefix the denominator too if needed.
- EXAMPLES (printed -> unit_siunitx):
  "µm" -> "\micro\metre"          "m/s" -> "\metre\per\second"
  "m/s²" -> "\metre\per\second\squared"   "cm" -> "\centi\metre"
  "kg" -> "\kilogram"             "g" -> "\gram"     "°C" -> "\celsius"
  "MΩ" -> "\mega\ohm"             "µF" -> "\micro\farad"   "kW" -> "\kilo\watt"
  "N m" -> "\newton\metre"        "Hz" -> "\hertz"   "kJ" -> "\kilo\joule"
  "kg/m³" -> "\kilogram\per\cubic\metre"
  "J/(kg °C)" -> "\joule\per\kilogram\per\celsius"
  "mol/dm³" -> "\mole\per\cubic\deci\metre"
- If you are unsure of the exact siunitx form, still fill "unit" with the printed text
  and set "unit_siunitx" to "" — the system will format the plain text as a fallback.
- ONLY use the macros listed above. Do NOT invent macros or use any other LaTeX commands.

HIERARCHY (level):
- level 0 = a top-level part (becomes (a),(b),(c)).
- level 1 = a nested part (becomes (i),(ii),(iii)).
- A question with NO sub-parts = ONE part with level 0.

MARKS:
- marks = the integer in (N) for that part, or null if none shown.

ANSWER LINES (answer_rows):
- CRITICAL: Only add answer_rows if the screenshot LITERALLY SHOWS printed answer lines:
  solid lines, dotted lines, an "Answer:" prompt, or a printed template like "x = ____ unit".
  When it does, set "lines_visible": true and list ONE object per printed answer line.
  If you are not certain printed answer lines/blanks are visible, use "answer_rows": []
  and "lines_visible": false. The MAJORITY of questions have NO printed lines — just
  empty working space — so default to an empty list.
- Each row object describes ONE printed answer line, top to bottom:
  - "label": any printed text that appears BEFORE the line on that row. Examples:
    "Object", "Reason", "wavelength at maximum intensity for object L", "x".
    Use '' if the line has no leading label.
  - "has_equals": true if an "=" sign is printed between the label and the line
    (e.g. "x = ____", "L = ____"). false otherwise (e.g. "Object ........").
  - "unit": any unit printed AFTER the line on that row, copied EXACTLY as printed
    (e.g. "µm", "m/s", "cm", "kg/m³"). '' if no unit.
  - "unit_siunitx": the SAME unit rewritten using siunitx macros (see UNITS below),
    e.g. "\micro\metre", "\metre\per\second", "\kilogram\per\cubic\metre". '' if no unit.
  - "width": "narrow" (short blank, e.g. a single value or coordinate),
    "standard" (normal), "wide" (long line spanning the page). Default "standard".
  - "indent": true if the label/line is visibly indented/centered on the page
    (common for "quantity = ____ unit" numeric answers). false if it starts at the
    left margin (common for "Object ...", "Reason ..." descriptive answers).
- EXAMPLES:
  - "Object ........" then "Reason ........"  =>
      [{"label":"Object","has_equals":false,"unit":"","width":"standard","indent":false},
       {"label":"Reason","has_equals":false,"unit":"","width":"standard","indent":false}]
  - "wavelength ... object L = ........ µm" then "... object M = ........ µm" =>
      [{"label":"wavelength at maximum intensity for object L","has_equals":true,"unit":"µm","width":"standard","indent":true},
       {"label":"wavelength at maximum intensity for object M","has_equals":true,"unit":"µm","width":"standard","indent":true}]
  - a single plain answer line => [{"label":"","has_equals":false,"unit":"","width":"standard","indent":false}]
  - "x = ........" => [{"label":"x","has_equals":true,"unit":"","width":"standard","indent":false}]

ANSWER KIND (special printed templates):
- "answer_kind": "coordinates" ONLY if the image prints a "( ____ , ____ )" template.
  In that case answer_rows may be []. Otherwise "none".

FIGURE vs TABLE — IMPORTANT, these are mutually exclusive:
- A TABLE is a grid of rows and columns of data/text with ruled lines. Set is_table=true,
  fill "table", and set figure_here=false. NEVER mark a table as a figure.
- A FIGURE is a diagram, drawing, photo, graph, chart, or geometric illustration (NOT a grid of data).
  Set figure_here=true and is_table=false.
- If something has both a table and a separate diagram, use separate parts.

FIGURE (PLACEHOLDER ONLY — you do NOT crop or give coordinates):
- figure_here: true ONLY for a diagram/drawing/photo/graph/chart (never for a data table).
- figure_position: "before" (figure before the question instruction, usual) or "after".
- figure_size: "small" (simple shape), "medium" (standard diagram), "large" (graph/grid/wide).
- That is ALL for figures — just mark that one exists, where, and its size.
- Do NOT transcribe text that is inside the figure artwork.

TABLE:
- is_table=true + fill "table" for any grid of data with rows and columns.
- has_header: true if the first row is column headers; false for label-style tables.
- Keep inline maths in cells. Empty cells = "".
- A table is rendered directly as LaTeX — it does NOT need a figure/image.

MCQ: is_mcq + mcq_options as a list (A,B,C,D order).
BULLETS: bulleted items as a list of texts.

LINE BREAKS (breaks array):
- Mark where the ORIGINAL visibly starts a new line within this part's text.
- type: "tight" (stacked lines), "para" (paragraph gap), "double" (larger gap).
- If unsure, omit breaks.

Output ONLY the JSON. No LaTeX layout commands anywhere.
"""

# ============================================================================
# RULE-BASED FORMATTER — owns 100% of LaTeX. Fixed rules, no decisions.
# ============================================================================

def space_for_marks(marks):
    if marks is None:
        return "2cm"
    table = {1: "2cm", 2: "3cm", 3: "4cm", 4: "5cm", 5: "6cm"}
    if marks <= 0:
        return "2cm"
    return table.get(marks, "7cm")

FIG_WIDTH = {"small": "0.45", "medium": "0.6", "large": "0.8"}
def fig_width(size):
    return FIG_WIDTH.get(size, "0.6")

def ans_width(width):
    return WIDTHS.get(width, "4.5cm")

# ---------------------------------------------------------------------------
# UNIT ENGINE — AI-first, with deterministic fallback. NEVER breaks the compile.
# Tier 1: AI-supplied siunitx (validated against a whitelist) -> \unit{...}
# Tier 2: deterministic mapping of the plain printed text (gensymb macros)
# Tier 3: upright literal text
# Requires \usepackage{siunitx}, \usepackage{gensymb}, \usepackage{textcomp}.
# ---------------------------------------------------------------------------
_SI_PREFIXES = ["quecto","ronto","yocto","zepto","atto","femto","pico","nano","micro",
                "milli","centi","deci","deca","deka","hecto","kilo","mega","giga","tera",
                "peta","exa","zetta","yotta","ronna","quetta"]
_SI_UNITS = ["metre","meter","gram","kilogram","second","ampere","kelvin","mole",
             "candela","hertz","newton","pascal","joule","watt","coulomb","volt","farad",
             "ohm","siemens","weber","tesla","henry","celsius","degreeCelsius","lumen",
             "lux","becquerel","gray","sievert","katal","litre","liter","electronvolt",
             "dalton","bar","bel","decibel","neper","minute","hour","day","degree",
             "arcminute","arcsecond","radian","steradian","percent","tonne","angstrom",
             "astronomicalunit","gon"]
_SI_ABBR = ["fg","pg","ng","ug","mg","kg","pm","nm","um","mm","cm","dm","km","fs","ps",
            "ns","us","ms","fmol","pmol","nmol","umol","mmol","kmol","pA","nA","uA","mA",
            "kA","mV","kV","mW","kW","MW","mF","uF","nF","pF","kohm","Mohm","kHz","MHz",
            "GHz","kPa","MPa","kJ","MJ","kN","Hz","Pa","Wb","Sv","Gy","Bq","mol","cd",
            "lm","lx"]
_SI_OPS = ["per","squared","cubed","tothe","raisetothe","of","square","cubic","quartic",
           "power","ang","highlight"]
_ALLOWED_SI = set("\\" + t for t in (_SI_PREFIXES + _SI_UNITS + _SI_ABBR + _SI_OPS))

def validate_siunitx(s):
    """True only if s is a safe \\unit{...} body: whitelisted macros, digits, braces,
    spaces; balanced braces; no other backslash macros. Blocks LaTeX injection."""
    if not s or len(s) > 200:
        return False
    depth = 0
    for ch in s:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
    if depth != 0:
        return False
    for m in re.finditer(r"\\[a-zA-Z]+", s):
        if m.group(0) not in _ALLOWED_SI:
            return False
    if not re.fullmatch(r"[0-9{}\s]*", re.sub(r"\\[a-zA-Z]+", "", s)):
        return False
    return True

# Longer / multi-char source tokens first so 'm/s' beats 'm', '°C' beats '°'.
_UNIT_TOKENS = [
    ("°C", r"\celsius"), ("degC", r"\celsius"), ("celsius", r"\celsius"),
    ("°", r"\degree"), ("deg", r"\degree"),
    ("µ", r"\micro "), ("μ", r"\micro "), ("micro", r"\micro "),  # U+00B5 and U+03BC
    ("Ω", r"\ohm"), ("ohms", r"\ohm"), ("ohm", r"\ohm"),
]

def _fallback_plain(unit):
    r"""Tier 2: map plain printed text -> LaTeX using gensymb macros + \mathrm.
    Also normalises Unicode superscripts (²³) and middots so it never crashes."""
    if not unit:
        return ""
    u = unit.strip()
    # normalise common Unicode that inputenc(utf8) chokes on
    sup = {"\u00b2": "^{2}", "\u00b3": "^{3}", "\u2070": "^{0}", "\u00b9": "^{1}",
           "\u2074": "^{4}", "\u2075": "^{5}", "\u2076": "^{6}", "\u2077": "^{7}",
           "\u2078": "^{8}", "\u2079": "^{9}", "\u207b": "^{-}",
           "\u00b7": r"\cdot ", "\u22c5": r"\cdot ", "\u2009": " ", "\u00a0": " "}
    for k, v in sup.items():
        u = u.replace(k, v)
    special = {}
    for i, (src, tex) in enumerate(_UNIT_TOKENS):
        if src in u:
            key = f"\x00{i}\x00"
            u = u.replace(src, key)
            special[key] = tex
    def wrap_chunk(chunk):
        return re.sub(r"[A-Za-z]+", lambda m: r"\mathrm{%s}" % m.group(0), chunk)
    out = []
    for p in re.split(r"(\x00\d+\x00)", u):
        out.append(special.get(p, wrap_chunk(p)))
    return "".join(out).strip()

def render_unit(unit_plain="", unit_siunitx=""):
    """Return (latex, is_math_mode). Never raises; never breaks the compile."""
    si = (unit_siunitx or "").strip()
    if si and validate_siunitx(si):
        return (r"\unit{%s}" % si, False)        # \unit is valid in text mode
    plain = (unit_plain or "").strip()
    tex = _fallback_plain(plain)
    if tex:
        return (tex, _is_mathy(tex))
    if plain:
        safe = (plain.replace("&", r"\&").replace("%", r"\%")
                     .replace("#", r"\#").replace("_", r"\_"))
        # strip any non-ASCII so inputenc never fails on the literal path
        safe = safe.encode("ascii", "ignore").decode("ascii").strip()
        if safe:
            return (r"\mathrm{%s}" % safe, True)
    return ("", False)

def _is_mathy(tex):
    return ("\\" in tex) or ("^" in tex) or ("_" in tex)

def _esc_label(s):
    return s.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")

# fixed-width dotted answer lines (matches Edexcel printed dotted leaders)
WIDTHS = {"narrow": "2.5cm", "standard": "4.5cm", "wide": "7cm"}

def _dotted(width_cm):
    return r"\makebox[%s]{\dotfill}" % width_cm


def render_answer_row(row):
    label = (row.get("label") or "").strip()
    unit = (row.get("unit") or "").strip()
    unit_si = (row.get("unit_siunitx") or "").strip()
    has_eq = bool(row.get("has_equals"))
    indent = bool(row.get("indent"))
    width = row.get("width", "standard")
    unit_tex, unit_math = render_unit(unit, unit_si)

    if has_eq or unit:
        # numeric style:  label = ....(fixed dotted box).... unit   (kept on one line)
        eqp = " = " if (has_eq or label) else ""
        u = ""
        if unit_tex:
            u = r"\,$%s$" % unit_tex if unit_math else r"\,%s" % unit_tex
        box = _dotted(WIDTHS.get(width, "4.5cm"))
        line = r"%s%s\mbox{%s%s}" % (_esc_label(label), eqp, box, u)
        if indent:
            return r"\noindent \hspace*{1.5cm}%s" % line
        return r"\noindent %s" % line
    # descriptive style:  Object .............................. (leader to margin)
    if label:
        return r"\noindent %s~\dotfill" % _esc_label(label)
    return r"\noindent %s" % _dotted(WIDTHS.get(width, "4.5cm"))

def _legacy_rows(part):
    """Back-compat: convert an old answer_type into answer_rows so older saved
    state / older model output still renders."""
    at = part.get("answer_type", "none")
    label = (part.get("answer_label") or "").strip()
    unit = (part.get("answer_unit") or "").strip()
    width = part.get("answer_width", "standard")
    if at == "line":
        return [{"label": "", "width": width}]
    if at == "line_unit":
        return [{"label": label, "has_equals": bool(label), "unit": unit, "width": width}]
    if at == "equation":
        return [{"label": label or "x", "has_equals": True, "width": width}]
    if at == "two_values":
        return [{"label": "", "width": "narrow"}, {"label": "", "width": "narrow"}]
    if at == "answer_label":
        return [{"label": "Answer", "width": width}]
    if at == "coordinates":
        part["answer_kind"] = "coordinates"
    return []

def answer_block(part):
    """100% rule-based answer template. Renders only when printed lines are visible."""
    if not part.get("lines_visible", False):
        return ""
    if part.get("answer_kind") == "coordinates":
        return ("\n\n\\noindent \\( ( \\makebox[2.5cm]{\\dotfill} , "
                "\\makebox[2.5cm]{\\dotfill} ) \\)\n\n\\vspace{0.5cm}")
    rows = part.get("answer_rows")
    if rows is None or (not rows and "answer_type" in part):
        rows = _legacy_rows(part)
        if part.get("answer_kind") == "coordinates":
            return ("\n\n\\noindent \\( ( \\makebox[2.5cm]{\\dotfill} , "
                    "\\makebox[2.5cm]{\\dotfill} ) \\)\n\n\\vspace{0.5cm}")
    if not rows:
        return ""
    chunks = []
    for r in rows:
        chunks.append(render_answer_row(r))
        chunks.append(r"\vspace{0.8cm}")
    return "\n\n" + "\n\n".join(chunks)

def apply_breaks(text, breaks):
    if not breaks:
        return text
    sentences = re.split(r'(?<=\.)\s+', text.strip())
    out = []
    bmap = {b["after_sentence"]: b["type"] for b in breaks if "after_sentence" in b}
    for idx, sent in enumerate(sentences):
        out.append(sent)
        if idx in bmap:
            t = bmap[idx]
            if t == "tight":
                out.append("\\\\\n")
            elif t == "double":
                out.append("\n\n\\vspace{0.3cm}\n")
            else:
                out.append("\n\n")
        else:
            out.append(" ")
    return "".join(out).strip()

def clean_text(text):
    if not text:
        return ""
    s = text.strip()
    if re.fullmatch(r'\d+\s*[.)]?', s):
        return ""
    s = re.sub(r'^\s*\d+\s*[.)]?\s+', '', s)
    unit_alt = (r'(?:°C|µm|μm|µs|μs|m/s|km/h|cm|mm|nm|km|kg|kHz|MHz|Hz|µ|μ|Ω|°'
                r'|mol|rad|Pa|m|s|g|N|J|W|V|A|K)')
    def _sp(m):
        tex, mathy = render_unit(m.group(1), "")
        if not tex:
            return m.group(0)
        return ('\\,$' + tex + '$') if mathy else ('\\,' + tex)
    s = re.sub(r'(?<=\d)\s+(' + unit_alt + r')\b', _sp, s)
    return s

def render_table(tbl):
    headers = tbl.get("headers", [])
    rows = tbl.get("rows", [])
    has_header = tbl.get("has_header", True)
    ncol = max(len(headers), max((len(r) for r in rows), default=0)) or 2
    spec = "|" + "c|" * ncol
    out = ["\\begin{center}", f"\\begin{{tabular}}{{{spec}}}", "\\hline"]
    if headers and has_header:
        out.append(" & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\ \\hline")
    elif headers and not has_header:
        out.append(" & ".join(str(h) for h in headers) + " \\\\ \\hline")
    for r in rows:
        cells = list(r) + [""] * (ncol - len(r))
        out.append(" & ".join(str(c) for c in cells) + " \\\\ \\hline")
    out += ["\\end{tabular}", "\\end{center}"]
    return "\n".join(out)

def render_mcq(options):
    out = ["\\begin{center}", "\\begin{tabular}{|c|l|}", "\\hline"]
    letters = ["A", "B", "C", "D", "E", "F"]
    for i, opt in enumerate(options[:6]):
        out.append(f"{letters[i]} & {opt} \\\\ \\hline")
    out += ["\\end{tabular}", "\\end{center}"]
    return "\n".join(out)

def render_bullets(bullets):
    out = ["\\begin{itemize}"]
    for b in bullets:
        out.append(f"    \\item {b}")
    out.append("\\end{itemize}")
    return "\n".join(out)

def figure_placeholder(n, size):
    return ("\\begin{center}\n"
            f"\\includegraphics[width={fig_width(size)}\\textwidth]{{{n}.png}}\n"
            "\\end{center}")

def render_part(part, fig_counter):
    lines = []
    text = clean_text(part.get("text", ""))
    text = apply_breaks(text, part.get("breaks", []))
    fig_pos = part.get("figure_position", "before")
    fig_size = part.get("figure_size", "medium")

    # A table is rendered as LaTeX tabular, NEVER as a figure placeholder.
    is_table = bool(part.get("is_table") and part.get("table"))

    fig_num = None
    # only allocate a figure if there's a real figure AND it's not a table
    if part.get("figure_here") and not is_table:
        fig_counter[0] += 1
        fig_num = fig_counter[0]

    if is_table:
        if text:
            lines.append(text); text = ""
        lines.append(render_table(part["table"]))

    if fig_num is not None and fig_pos == "before":
        lines.append(figure_placeholder(fig_num, fig_size))

    if text:
        lines.append(text)

    if part.get("bullets"):
        lines.append(render_bullets(part["bullets"]))

    if fig_num is not None and fig_pos == "after":
        lines.append(figure_placeholder(fig_num, fig_size))

    if part.get("is_mcq") and part.get("mcq_options"):
        lines.append(render_mcq(part["mcq_options"]))

    body = "\n".join(lines)

    marks = part.get("marks")
    if marks is not None:
        body += f"\n\n\\hfill ({marks})"
    body += f"\n\n\\vspace{{{space_for_marks(marks)}}}"
    body += answer_block(part)
    return body, fig_counter

def item_fmt(body, indent="  "):
    if body.lstrip().startswith("\\begin"):
        return f"{indent}\\item \\hfill\n{body}"
    return f"{indent}\\item {body}"

def generate_latex(structured, fig_counter=None):
    if fig_counter is None:
        fig_counter = [0]
    blocks = []

    intro = clean_text(structured.get("intro") or "")
    intro = apply_breaks(intro, structured.get("intro_breaks", []))
    if intro:
        blocks.append(intro)

    parts = structured.get("parts", [])
    real_multi = len(parts) > 1

    if not real_multi and len(parts) == 1 and parts[0].get("level", 0) == 0:
        body, fig_counter = render_part(parts[0], fig_counter)
        blocks.append(body)
    elif parts:
        out = ["\\begin{enumerate}"]
        i = 0
        while i < len(parts):
            p = parts[i]
            if p.get("level", 0) == 0:
                body, fig_counter = render_part(p, fig_counter)
                nested = []
                j = i + 1
                while j < len(parts) and parts[j].get("level", 0) == 1:
                    nested.append(parts[j]); j += 1
                if nested:
                    out.append(item_fmt(body))
                    out.append("  \\begin{enumerate}")
                    for np in nested:
                        nbody, fig_counter = render_part(np, fig_counter)
                        out.append(item_fmt(nbody, indent="    "))
                    out.append("  \\end{enumerate}")
                    i = j
                else:
                    out.append(item_fmt(body)); i += 1
            else:
                body, fig_counter = render_part(p, fig_counter)
                out.append(item_fmt(body)); i += 1
        out.append("\\end{enumerate}")
        blocks.append("\n".join(out))

    return "\n\n".join(blocks), fig_counter[0]


# ============================================================================
# /convert  — text extraction only, no cropping
# ============================================================================
@app.post("/convert")
async def convert(images: List[UploadFile] = File(...), questionNumber: int = Form(...)):
    raw_list = []
    for upload in images:
        raw = await upload.read()
        try:
            Image.open(io.BytesIO(raw)).verify()
        except Exception:
            raise HTTPException(400, f"Could not read image: {upload.filename}")
        raw_list.append((raw, upload.content_type or "image/png"))

    if not raw_list:
        raise HTTPException(400, "No images provided")
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(500, "GEMINI_API_KEY not set on the server")

    content_parts = []
    for i, (raw, mime) in enumerate(raw_list, start=1):
        content_parts.append({"mime_type": mime, "data": raw})
        if len(raw_list) > 1:
            content_parts.append(f"(Screenshot {i} of {len(raw_list)} for the same question.)")
    content_parts.append("Extract the structure of this exam question. Return valid JSON only.")

    model = genai.GenerativeModel(MODEL, system_instruction=SYSTEM_PROMPT)
    try:
        resp = model.generate_content(
            content_parts,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.1,
                "max_output_tokens": 16384,
            },
        )
        text = resp.text.strip()
    except Exception as e:
        raise HTTPException(502, f"Gemini call failed: {e}")

    structured = _parse_json(text)
    if structured is None:
        raise HTTPException(502, f"Model did not return valid JSON:\n{text[:500]}")

    try:
        latex, fig_count = generate_latex(structured)
    except Exception as e:
        raise HTTPException(500, f"LaTeX generation failed: {e}")

    total = structured.get("totalMarks")
    return {"latex": latex, "totalMarks": total, "marksFound": total is not None,
            "structured": structured, "figureCount": fig_count}


def _parse_json(text):
    text = re.sub(r"^```(json)?\s*|```\s*$", "", text.strip()).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(re.sub(r'(?<!\\)\n', r'\\n', text))
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(re.sub(r'(?<!\\)\n', r'\\n', m.group()))
        except json.JSONDecodeError:
            pass
    return None


# ============================================================================
# /export  — renumber placeholders globally, bundle any uploaded images
# ============================================================================
class QIn(BaseModel):
    body: str
    marks: Optional[int] = None
    figureCount: int = 0

class FigIn(BaseModel):
    name: str
    dataUrl: str

class PaperIn(BaseModel):
    title: str; author: str; cred: str; inst: str; contact: str; date: str
    questions: List[QIn]
    figures: List[FigIn] = []

@app.post("/export")
def export(paper: PaperIn):
    global_idx = 0
    items = []
    for qi, q in enumerate(paper.questions, start=1):
        body = (q.body or "").strip()
        # renumber this question's local figure placeholders (1.png,2.png within the body)
        # to global numbers across the whole paper.
        # The body uses {k.png} per-question; remap to running globals.
        n_here = int(q.figureCount or 0)
        if n_here > 0:
            # replace from highest to lowest to avoid collisions
            mapping = {}
            for local_k in range(1, n_here + 1):
                global_idx += 1
                mapping[local_k] = global_idx
            for local_k in sorted(mapping.keys(), reverse=True):
                body = body.replace(f"{{{local_k}.png}}", f"{{__G{mapping[local_k]}__}}")
            body = re.sub(r"__G(\d+)__", r"\1.png", body)
        total = ""
        if q.marks is not None:
            total = f"\n\n\\hfill \\textbf{{(Total for Question {qi} is {q.marks} marks)}}"
        if body.lstrip().startswith("\\begin"):
            items.append(f"\\item \\hfill\n{body}{total}\n\\hrule")
        else:
            items.append(f"\\item\n{body}{total}\n\\hrule")

    tex = _build_doc(paper, "\n\n".join(items))

    # decode the user's cropped figures (base64 data URLs) and bundle them
    import base64 as _b64
    packaged = {}
    for fig in paper.figures:
        m = re.match(r'data:image/\w+;base64,(.*)$', fig.dataUrl, re.DOTALL)
        if not m:
            continue
        try:
            packaged[fig.name] = _b64.b64decode(m.group(1))
        except Exception:
            continue

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("paper.tex", tex)
        for name, data in packaged.items():
            z.writestr(name, data)
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="paper_export.zip"'})


def _build_doc(p, items):
    return (
        "\\documentclass[a4paper,12pt]{article}\n"
        "\\usepackage{amsmath}\n\\usepackage{amssymb}\n\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{geometry}\n\\usepackage{array}\n\\usepackage{graphicx}\n"
        "\\usepackage{textcomp}\n\\usepackage{gensymb}\n\\usepackage{siunitx}\n"
        # Safety net: render stray Unicode the AI may type as plain text (in MCQ
        # options, table cells, bullets, prose) instead of crashing inputenc.
        "\\DeclareUnicodeCharacter{00B5}{\\textmu}\n"
        "\\DeclareUnicodeCharacter{03BC}{\\textmu}\n"
        "\\DeclareUnicodeCharacter{03A9}{\\ensuremath{\\Omega}}\n"
        "\\DeclareUnicodeCharacter{2126}{\\ensuremath{\\Omega}}\n"
        "\\DeclareUnicodeCharacter{00B0}{\\textdegree}\n"
        "\\DeclareUnicodeCharacter{00B2}{\\textsuperscript{2}}\n"
        "\\DeclareUnicodeCharacter{00B3}{\\textsuperscript{3}}\n"
        "\\DeclareUnicodeCharacter{00B7}{\\textperiodcentered}\n"
        "\\DeclareUnicodeCharacter{2212}{\\ensuremath{-}}\n"
        "\\DeclareUnicodeCharacter{00D7}{\\ensuremath{\\times}}\n"
        "\\geometry{margin=1in}\n\n\\begin{document}\n"
        f"\\title{{\\LARGE \\textbf{{{p.title}}}}}\n"
        f"\\author{{\\large {p.author} \\\\ \\text{{{p.cred}}} \\\\ {p.inst} \\\\ \\textbf{{Contact: {p.contact}}}}}\n"
        f"\\date{{{p.date}}}\n\\maketitle\n\\hrule\n\\begin{{enumerate}}\n\n"
        f"{items}\n\n\\end{{enumerate}}\n\n\\end{{document}}\n"
    )


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL, "key_set": bool(os.environ.get("GEMINI_API_KEY"))}
