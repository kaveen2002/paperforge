"""
PaperForge backend — Rule-Based Formatter edition
===================================================
DESIGN: AI extracts structured data + picks from FIXED MENUS only.
        Deterministic code owns 100% of LaTeX formatting.

POST /convert  ->  image(s) + questionNumber  =>  { latex, totalMarks, structured, crops }
POST /export   ->  full paper state            =>  zip of paper.tex + N.png figures
GET  /         ->  serves index.html
GET  /health   ->  status
"""

import os, io, json, base64, zipfile, re
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

FIGURE_STORE = {}

# ============================================================================
# EXTRACTION PROMPT — model extracts data + picks from FIXED MENUS only.
# It NEVER writes layout commands (\item, \vspace, \hfill, \begin, etc).
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
      "answer_type": "none|line|line_unit|equation|two_values|coordinates|answer_label",
      "lines_visible": false,
      "answer_label": "<e.g. 'wave speed' or 'x' or '' >",
      "answer_unit": "<e.g. 'm/s' or 'cm' or '' >",
      "answer_width": "standard|narrow",
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
  "figures": [{"box_2d":[ymin,xmin,ymax,xmax],"image_index":0}],
  "totalMarks": <integer or null>
}

=== EXTRACTION RULES ===

CONTENT:
- Copy ALL text EXACTLY. Preserve wording, numbers, punctuation, symbols.
- Do NOT include the question's own number (like "2").
- Write maths inline using \( \): e.g. \( x^2+3x \), \( 105^\circ \), \( \dfrac{a}{b} \).
- Keep units as written ("220 m", "0.70 s") — the system handles spacing.

HIERARCHY (level):
- level 0 = a top-level part (becomes (a),(b),(c)).
- level 1 = a nested part (becomes (i),(ii),(iii)).
- A question with NO sub-parts = ONE part with level 0.

MARKS:
- marks = the integer in (N) for that part, or null if none shown.

ANSWER TYPE (pick the ONE that matches how the answer is collected):
- CRITICAL: Only use a type OTHER than "none" if the screenshot LITERALLY SHOWS printed
  answer lines, dotted lines, an "Answer:" prompt, or a printed template like "x = ____".
  You must also set "lines_visible": true when you do. If you are not certain that printed
  answer lines/blanks are visible in the image, use "none" and "lines_visible": false.
- The MAJORITY of questions are "none" — they just have empty working space, NOT printed lines.
- "none" = working space only, NO printed answer line. (proofs, "show that", "describe",
  "explain", "calculate", "work out" — unless a printed blank is clearly visible).
- "line" = the image shows a single printed answer line.
- "line_unit" = the image shows an answer line followed by a unit.
- "equation" = the image shows a printed "x = ____".
- "two_values" = the image shows two printed answer lines.
- "coordinates" = the image shows a printed ( ____ , ____ ) template.
- "answer_label" = the image shows a printed "Answer: ____".
- When in any doubt: "none" with "lines_visible": false.

ANSWER WIDTH (pick by how long the expected answer is):
- "standard" = normal width (single values, expressions, units).
- "narrow" = short (each coordinate/value in two_values or coordinates).

FIGURE:
- figure_here: true if a diagram/photo belongs in this part.
- figure_position: "before" (figure before the question instruction, usual) or "after".
- figure_size: "small" (simple shape), "medium" (standard diagram), "large" (graph/grid/wide).

TABLE:
- is_table + table if a data table is present.
- has_header: true if the first row is column headers; false for label-style tables.
- Keep inline maths in cells (e.g. "\( 0 < m \le 100 \)"). Empty cells = "".

MCQ: is_mcq + mcq_options as a list of the option texts (A,B,C,D order).

BULLETS: if the part lists bulleted items, put their texts in bullets [].

LINE BREAKS (breaks array):
- Mark where the ORIGINAL visibly starts a new line within this part's text.
- For each break: "after_sentence" = the 0-based index of the sentence it follows,
  "type" = "tight" (lines directly stacked, e.g. listed conditions),
           "para" (normal paragraph gap between statements),
           "double" (a larger visual gap in the original).
- If unsure, omit breaks (the system uses paragraph spacing by default).

FIGURE BOXES:
- box_2d: [ymin, xmin, ymax, xmax] normalized 0-1000, TIGHT around artwork only.
- Detect diagrams/photos ONLY (not tables/text/equations).
- image_index: 0-based. Empty [] if none.

Output ONLY the JSON. No LaTeX layout commands anywhere.
"""

# ============================================================================
# RULE-BASED FORMATTER — owns 100% of LaTeX. Fixed rules, no decisions.
# ============================================================================

# --- spacing: code-derived from marks (FIXED) ---
def space_for_marks(marks):
    if marks is None:
        return "2cm"
    table = {1: "2cm", 2: "3cm", 3: "4cm", 4: "5cm", 5: "6cm"}
    if marks <= 0:
        return "2cm"
    return table.get(marks, "7cm")  # 6+ -> 7cm

# --- figure size menu (FIXED) ---
FIG_WIDTH = {"small": "0.45", "medium": "0.6", "large": "0.8"}
def fig_width(size):
    return FIG_WIDTH.get(size, "0.6")

# --- answer width menu (FIXED) ---
def ans_width(width):
    return "2cm" if width == "narrow" else "5cm"

# --- answer block layouts (FIXED, one per type) ---
def answer_block(part):
    at = part.get("answer_type", "none")
    # GUARD: only emit an answer line if the model confirmed printed lines are visible.
    if not part.get("lines_visible", False):
        at = "none"
    label = (part.get("answer_label") or "").strip()
    unit = (part.get("answer_unit") or "").strip()
    w = ans_width(part.get("answer_width", "standard"))
    if at == "line":
        return f"\n\n\\noindent \\underline{{\\hspace{{{w}}}}}\n\n\\vspace{{0.5cm}}"
    if at == "line_unit":
        lab = f"{label} = " if label else ""
        return f"\n\n\\noindent {lab}\\underline{{\\hspace{{{w}}}}} {unit}\n\n\\vspace{{0.5cm}}"
    if at == "equation":
        lab = label if label else "x"
        return f"\n\n\\[ {lab} = \\dotfill \\]\n\n\\vspace{{0.3cm}}"
    if at == "two_values":
        nw = "2cm"
        return (f"\n\n\\noindent \\underline{{\\hspace{{{nw}}}}}\n\n\\vspace{{0.3cm}}"
                f"\n\n\\noindent \\underline{{\\hspace{{{nw}}}}}\n\n\\vspace{{0.5cm}}")
    if at == "coordinates":
        return ("\n\n\\noindent \\( ( \\underline{\\hspace{2cm}} , \\underline{\\hspace{2cm}} ) \\)"
                "\n\n\\vspace{0.5cm}")
    if at == "answer_label":
        return f"\n\n\\noindent \\textbf{{Answer:}} \\underline{{\\hspace{{{w}}}}}\n\n\\vspace{{0.5cm}}"
    return ""  # none

# --- line break application (FIXED) ---
def apply_breaks(text, breaks):
    """Insert \\ / paragraph / double-gap at marked sentence boundaries."""
    if not breaks:
        return text
    # split text into sentences on '. ' boundaries (keep the period)
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
            else:  # para
                out.append("\n\n")
        else:
            out.append(" ")
    return "".join(out).strip()

# --- text cleanup (FIXED) ---
def clean_text(text):
    if not text:
        return ""
    s = text.strip()
    if re.fullmatch(r'\d+\s*[.)]?', s):
        return ""
    s = re.sub(r'^\s*\d+\s*[.)]?\s+', '', s)
    s = re.sub(r'(?<=\d)\s+(m/s|km/h|cm|mm|km|kg|Hz|m|s|g|N|J|W|V|A)\b',
               lambda m: '\\,' + m.group(1), s)
    return s

# --- table render (FIXED) ---
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
            f"\\includegraphics[width={fig_width(size)}\\textwidth]{{__FIGURE_{n}__}}\n"
            "\\end{center}")

def render_part(part, fig_counter):
    lines = []
    text = clean_text(part.get("text", ""))
    text = apply_breaks(text, part.get("breaks", []))
    fig_pos = part.get("figure_position", "before")
    fig_size = part.get("figure_size", "medium")

    fig_num = None
    if part.get("figure_here"):
        fig_counter[0] += 1
        fig_num = fig_counter[0]

    # table (with lead-in text)
    if part.get("is_table") and part.get("table"):
        if text:
            lines.append(text); text = ""
        lines.append(render_table(part["table"]))

    # figure before text
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
    # ORDER: question text -> (blank) marks -> (blank) working space -> (blank) answer
    if marks is not None:
        body += f"\n\n\\hfill ({marks})"
    body += f"\n\n\\vspace{{{space_for_marks(marks)}}}"
    body += answer_block(part)
    return body, fig_counter

def item_fmt(body, indent="  "):
    # if the part begins with a block element (figure/table), use \item \hfill
    # so the item label line is pushed and the figure sits cleanly below
    if body.lstrip().startswith("\\begin"):
        return f"{indent}\\item \\hfill\n{body}"
    return f"{indent}\\item {body}"

def generate_latex(structured):
    fig_counter = [0]
    blocks = []

    intro = clean_text(structured.get("intro") or "")
    intro = apply_breaks(intro, structured.get("intro_breaks", []))
    if intro:
        blocks.append(intro)

    parts = structured.get("parts", [])
    has_subparts = len(parts) > 1 or (len(parts) == 1 and parts[0].get("level", 0) != 0)
    # also treat single level-0 with no siblings as a plain question
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

    return "\n\n".join(blocks)


# ============================================================================
# /convert
# ============================================================================
@app.post("/convert")
async def convert(images: List[UploadFile] = File(...), questionNumber: int = Form(...)):
    raw_list, img_list = [], []
    for upload in images:
        raw = await upload.read()
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            raise HTTPException(400, f"Could not read image: {upload.filename}")
        raw_list.append((raw, upload.content_type or "image/png"))
        img_list.append(img)

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
                "max_output_tokens": 8192,
            },
        )
        text = resp.text.strip()
    except Exception as e:
        raise HTTPException(502, f"Gemini call failed: {e}")

    structured = _parse_json(text)
    if structured is None:
        raise HTTPException(502, f"Model did not return valid JSON:\n{text[:500]}")

    try:
        latex = generate_latex(structured)
    except Exception as e:
        raise HTTPException(500, f"LaTeX generation failed: {e}")

    total = structured.get("totalMarks")
    figures = structured.get("figures", []) or []

    crops = []
    for i, fig in enumerate(figures, start=1):
        img_idx = min(fig.get("image_index", 0), len(img_list) - 1)
        src_img = img_list[max(0, img_idx)]
        W, H = src_img.size
        box = fig.get("box_2d", [])
        if not box or len(box) != 4:
            continue
        rect = _box2d_to_px(box, W, H)
        if rect is None:
            continue
        try:
            crop = src_img.crop(rect)
            if crop.width < 10 or crop.height < 10:
                continue
            buf = io.BytesIO(); crop.save(buf, format="PNG"); png = buf.getvalue()
        except Exception:
            continue
        tmp_name = f"q{questionNumber}_fig{i}.png"
        FIGURE_STORE[tmp_name] = png
        preview = "data:image/png;base64," + base64.standard_b64encode(png).decode()
        crops.append({"placeholder": f"__FIGURE_{i}__", "tempName": tmp_name,
                      "box_2d": box, "rect": rect, "dataUrl": preview})

    return {"latex": latex, "totalMarks": total, "marksFound": total is not None,
            "structured": structured, "crops": crops}


def _parse_json(text):
    text = re.sub(r"^```(json)?\s*|```\s*$", "", text.strip()).strip()
    # attempt 1: direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # attempt 2: fix unescaped newlines in strings
    try:
        return json.loads(re.sub(r'(?<!\\)\n', r'\\n', text))
    except json.JSONDecodeError:
        pass
    # attempt 3: extract the JSON object
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(re.sub(r'(?<!\\)\n', r'\\n', m.group()))
        except json.JSONDecodeError:
            pass
    # attempt 4: response was TRUNCATED mid-JSON — try to repair by closing it
    repaired = _repair_truncated(text)
    if repaired is not None:
        return repaired
    return None


def _repair_truncated(text):
    """Best-effort repair of a JSON object cut off mid-stream.
    Strategy: keep only up to the last COMPLETE top-level structure we can close.
    Truncation recovery is lossy; this salvages what parsed cleanly."""
    s = re.sub(r'(?<!\\)\n', r'\\n', text)
    start = s.find('{')
    if start == -1:
        return None
    s = s[start:]
    # progressively trim from the end, trying to close open braces/brackets, until valid
    # first, try closing as-is with bracket balancing
    def try_close(fragment):
        stack, in_str, esc = [], False, False
        for ch in fragment:
            if esc: esc = False; continue
            if ch == '\\': esc = True; continue
            if ch == '"': in_str = not in_str; continue
            if in_str: continue
            if ch in '{[': stack.append(ch)
            elif ch == '}' and stack and stack[-1] == '{': stack.pop()
            elif ch == ']' and stack and stack[-1] == '[': stack.pop()
        suffix = '"' if in_str else ''
        for opener in reversed(stack):
            suffix += '}' if opener == '{' else ']'
        return fragment + suffix

    # try the full fragment, then trim back to each preceding '}' (end of a complete part)
    candidates = [try_close(s)]
    # cut at successive last '}' positions to drop an incomplete trailing object
    idx = len(s)
    for _ in range(6):
        cut = s.rfind('}', 0, idx)
        if cut == -1:
            break
        frag = s[:cut+1]
        # strip a trailing comma if present
        frag = re.sub(r',\s*$', '', frag)
        candidates.append(try_close(frag))
        idx = cut
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None


def _box2d_to_px(box_2d, W, H, pad_frac=0.02):
    ymin, xmin, ymax, xmax = box_2d
    for v in (ymin, xmin, ymax, xmax):
        if not (0 <= v <= 1000):
            return None
    if ymin >= ymax or xmin >= xmax:
        return None
    left = int(xmin/1000*W); top = int(ymin/1000*H)
    right = int(xmax/1000*W); bottom = int(ymax/1000*H)
    px, py = int(pad_frac*W), int(pad_frac*H)
    return (max(0,left-px), max(0,top-py), min(W,right+px), min(H,bottom+py))


# ============================================================================
# /export
# ============================================================================
class QIn(BaseModel):
    body: str
    marks: Optional[int] = None
    tempImageNames: List[str] = []

class PaperIn(BaseModel):
    title: str; author: str; cred: str; inst: str; contact: str; date: str
    questions: List[QIn]

@app.post("/export")
def export(paper: PaperIn):
    global_idx = 0
    items, packaged = [], {}
    for qi, q in enumerate(paper.questions, start=1):
        body = (q.body or "").strip()
        for local_i, tmp in enumerate(q.tempImageNames, start=1):
            global_idx += 1
            final = f"{global_idx}.png"
            body = body.replace(f"__FIGURE_{local_i}__", final)
            if tmp in FIGURE_STORE:
                packaged[final] = FIGURE_STORE[tmp]
        total = ""
        if q.marks is not None:
            total = f"\n\n\\hfill \\textbf{{(Total for Question {qi} is {q.marks} marks)}}"
        # if the question body begins with a block (figure/table), use \item \hfill
        if body.lstrip().startswith("\\begin"):
            items.append(f"\\item \\hfill\n{body}{total}\n\\hrule")
        else:
            items.append(f"\\item\n{body}{total}\n\\hrule")

    tex = _build_doc(paper, "\n\n".join(items))
    refs = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex)
    missing = [r for r in refs if r not in packaged]
    if missing:
        raise HTTPException(409, f"Missing figures: {missing}")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("paper.tex", tex)
        for name, png in packaged.items():
            z.writestr(name, png)
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="paper_export.zip"'})


def _build_doc(p, items):
    return (
        "\\documentclass[a4paper,12pt]{article}\n"
        "\\usepackage{amsmath}\n\\usepackage{amssymb}\n\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{geometry}\n\\usepackage{array}\n\\usepackage{graphicx}\n"
        "\\geometry{margin=1in}\n\n\\begin{document}\n"
        f"\\title{{\\LARGE \\textbf{{{p.title}}}}}\n"
        f"\\author{{\\large {p.author} \\\\ \\text{{{p.cred}}} \\\\ {p.inst} \\\\ \\textbf{{Contact: {p.contact}}}}}\n"
        f"\\date{{{p.date}}}\n\\maketitle\n\\hrule\n\\begin{{enumerate}}\n\n"
        f"{items}\n\n\\end{{enumerate}}\n\n\\end{{document}}\n"
    )


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL, "key_set": bool(os.environ.get("GEMINI_API_KEY")),
            "figures_cached": len(FIGURE_STORE)}
