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
- "none" = working space only, NO printed answer line.
- "line" = the image shows a single printed answer line.
- "line_unit" = the image shows an answer line followed by a unit (set answer_unit).
- "equation" = the image shows a printed "x = ____" (set answer_label).
- "two_values" = the image shows two printed answer lines.
- "coordinates" = the image shows a printed ( ____ , ____ ) template.
- "answer_label" = the image shows a printed "Answer: ____".
- When in any doubt: "none" with "lines_visible": false.

ANSWER WIDTH:
- "standard" = normal width. "narrow" = short (coordinates/two_values).

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
    return "2cm" if width == "narrow" else "5cm"

def answer_block(part):
    at = part.get("answer_type", "none")
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
    return ""

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
    s = re.sub(r'(?<=\d)\s+(m/s|km/h|cm|mm|km|kg|Hz|m|s|g|N|J|W|V|A)\b',
               lambda m: '\\,' + m.group(1), s)
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

    # call Gemini with detect-and-retry: distinguishes rate-limit from truncation,
    # retries truncation once with a higher token ceiling, returns plain messages.
    structured = _call_and_parse(content_parts)
    # _call_and_parse raises HTTPException with a clean message on hard failure

    try:
        latex, fig_count = generate_latex(structured)
    except Exception as e:
        raise HTTPException(500, f"LaTeX generation failed: {e}")

    total = structured.get("totalMarks")
    return {"latex": latex, "totalMarks": total, "marksFound": total is not None,
            "structured": structured, "figureCount": fig_count}


def _classify_error(err_text):
    """Return 'rate', 'safety', or 'other' from an exception/string."""
    t = (err_text or "").lower()
    if "429" in t or "quota" in t or "rate" in t or "resource_exhausted" in t:
        return "rate"
    if "safety" in t or "blocked" in t or "recitation" in t:
        return "safety"
    return "other"


def _finish_truncated(resp):
    """True if the response was cut off by the token limit."""
    try:
        for cand in (resp.candidates or []):
            fr = getattr(cand, "finish_reason", None)
            # MAX_TOKENS finish reason indicates truncation
            if fr is not None and ("MAX_TOKENS" in str(fr) or str(fr) == "2"):
                return True
    except Exception:
        pass
    return False


def _one_call(content_parts, max_tokens):
    """One Gemini call. Returns (text, resp). Raises on API error."""
    model = genai.GenerativeModel(MODEL, system_instruction=SYSTEM_PROMPT)
    resp = model.generate_content(
        content_parts,
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.1,
            "max_output_tokens": max_tokens,
        },
    )
    text = ""
    try:
        text = (resp.text or "").strip()
    except Exception:
        # resp.text can raise on truncation/block; pull partial text from parts
        try:
            for cand in (resp.candidates or []):
                for p in (cand.content.parts or []):
                    if getattr(p, "text", None):
                        text += p.text
        except Exception:
            text = ""
        text = text.strip()
    return text, resp


def _call_and_parse(content_parts):
    """Call Gemini, detect failure type, retry truncation once, parse JSON.
    Raises HTTPException with a plain-English message on hard failure."""
    # --- attempt 1 ---
    try:
        text, resp = _one_call(content_parts, 16384)
    except Exception as e:
        kind = _classify_error(str(e))
        if kind == "rate":
            raise HTTPException(429, "Gemini is rate-limited right now. Wait a few seconds and press Convert again.")
        if kind == "safety":
            raise HTTPException(422, "Gemini blocked this image. Try a clearer or differently-cropped screenshot.")
        raise HTTPException(502, "Could not reach Gemini. Check your connection and try again.")

    data = _parse_json(text)
    if data is not None:
        return data

    # Parse failed for ANY reason — retry once with a much higher token ceiling.
    # (Covers real truncation AND cases where finish-reason detection missed it.)
    try:
        text2, resp2 = _one_call(content_parts, 32768)
    except Exception as e:
        kind = _classify_error(str(e))
        if kind == "rate":
            raise HTTPException(429, "Gemini is rate-limited right now. Wait a few seconds and press Convert again.")
        raise HTTPException(502, "Could not process this question. Try splitting it into two smaller screenshots.")
    data2 = _parse_json(text2)
    if data2 is not None:
        return data2

    # Still failing — final attempt: ask the model for a more compact response
    # (shorter field values) so the JSON fits and stays well-formed.
    concise = list(content_parts)
    concise.append("IMPORTANT: Return COMPACT valid JSON. Keep text fields complete but do not "
                   "add any commentary. Ensure the JSON is fully closed and valid.")
    try:
        text3, resp3 = _one_call(concise, 32768)
    except Exception:
        text3 = ""
    data3 = _parse_json(text3) if text3 else None
    if data3 is not None:
        return data3

    raise HTTPException(502, "This question is too complex to process in one go. Try splitting it into two "
                             "screenshots (e.g. parts (a)-(b) in one, the rest in another) and convert each separately.")


def _fix_invalid_escapes(s):
    r"""Double any backslash that is NOT a valid JSON escape.
    Valid JSON escapes: \" \\ \/ \b \f \n \r \t \uXXXX.
    LaTeX like \( \, \text becomes \\( \\, \\text so the JSON parses."""
    return re.sub(r'\\(?![\\"/bfnrtu])', r'\\\\', s)

def _parse_json(text):
    if not text:
        return None
    text = re.sub(r"^```(json)?\s*|```\s*$", "", text.strip()).strip()
    # 1: direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2: escape raw newlines
    try:
        return json.loads(re.sub(r'(?<!\\)\n', r'\\n', text))
    except json.JSONDecodeError:
        pass
    # 3: extract outermost object
    m = re.search(r'\{.*\}', text, re.DOTALL)
    candidate = m.group() if m else text
    # 4: progressive cleanup, INCLUDING invalid LaTeX escape sequences
    variants = []
    base = candidate.replace('\t', '\\t').replace('\r', '\\r')
    base = re.sub(r'(?<!\\)\n', r'\\n', base)
    base = re.sub(r',\s*([}\]])', r'\1', base)            # trailing commas
    variants.append(base)
    variants.append(_fix_invalid_escapes(base))           # fix \( \, \text etc.
    variants.append(_fix_invalid_escapes(candidate))
    variants.append(candidate)
    for attempt in variants:
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue
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
        "\\geometry{margin=1in}\n\n\\begin{document}\n"
        f"\\title{{\\LARGE \\textbf{{{p.title}}}}}\n"
        f"\\author{{\\large {p.author} \\\\ \\text{{{p.cred}}} \\\\ {p.inst} \\\\ \\textbf{{Contact: {p.contact}}}}}\n"
        f"\\date{{{p.date}}}\n\\maketitle\n\\hrule\n\\begin{{enumerate}}\n\n"
        f"{items}\n\n\\end{{enumerate}}\n\n\\end{{document}}\n"
    )


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL, "key_set": bool(os.environ.get("GEMINI_API_KEY"))}
