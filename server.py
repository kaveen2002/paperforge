"""
PaperForge backend — Structured Extraction edition
====================================================
The model EXTRACTS structured data (no LaTeX). Deterministic code GENERATES LaTeX
from fixed templates. This guarantees 100% consistent formatting.

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
# EXTRACTION PROMPT — model returns STRUCTURED DATA only, never LaTeX formatting
# ============================================================================
SYSTEM_PROMPT = r"""
You read a screenshot of ONE exam question and extract its STRUCTURE as JSON.
You do NOT write LaTeX layout. You only identify the content and label its parts.

Return STRICT VALID JSON:
{
  "intro": "<any text/equations that appear BEFORE the first sub-part, or '' if none>",
  "parts": [
    {
      "label": "a",                      // a, b, c... or "i","ii" for nested; "" if question has no parts
      "level": 0,                        // 0 = top-level part (a,b,c), 1 = nested (i,ii)
      "text": "<the full text of this part/question>",
      "marks": <integer or null>,
      "answer_type": "none|line|line_unit|equation_box|two_lines",
      "answer_label": "<e.g. 'wave speed' or 'x' or '' >",
      "answer_unit": "<e.g. 'm/s' or 'cm' or '' >",
      "figure_here": <true if a figure appears within this part, else false>,
      "is_table": <true if this part contains a data table>,
      "table": {"headers": ["..."], "rows": [["..."],["..."]]},  // only if is_table
      "is_mcq": <true if multiple choice>,
      "mcq_options": ["option A text","option B text",...],       // only if is_mcq
      "bullets": ["bullet 1","bullet 2"]                          // [] if none
    }
  ],
  "figures": [{"box_2d":[ymin,xmin,ymax,xmax],"image_index":0}],
  "totalMarks": <integer or null>
}

EXTRACTION RULES:
- Copy ALL text EXACTLY as shown. Preserve wording, numbers, punctuation, symbols.
- For maths in text, write it in LaTeX inline form using \( \) — e.g. \( x^2 + 3x \), \( 105^\circ \), \( \dfrac{a}{b} \).
- For a question with NO sub-parts: use ONE part with label "" and level 0.
- For sub-parts (a),(b),(c): each is a part with its label and level 0.
- For nested (i),(ii) under a part: separate parts with level 1, labels "i","ii".
- marks: the number in (N) for that part, or null if none shown.
- answer_type:
    "none" = no answer blank needed (just working space)
    "line" = a single answer blank line
    "line_unit" = answer blank with a unit (set answer_unit)
    "equation_box" = answer of form "x = ___" (set answer_label)
    "two_lines" = two answer blanks (e.g. two values)
- figure_here: true if a diagram/photo belongs in this part's flow.
- bullets: if the part lists items as bullets, put them here (text only, no markers).
- is_table/table: extract data tables with headers and rows.
- is_mcq/mcq_options: extract A/B/C/D choices as a list.

FIGURE BOXES:
- box_2d: [ymin, xmin, ymax, xmax] normalized 0-1000, TIGHT around artwork only.
- Detect diagrams/photos ONLY (not tables, not text, not equations).
- image_index: 0-based which screenshot. Empty [] if no figures.

Do NOT output any LaTeX layout commands (no \item, \vspace, \hfill, \begin, etc).
Only the content text (with inline maths) and the structural labels above.
"""

# ============================================================================
# DETERMINISTIC LATEX GENERATOR — fixed templates, 100% consistent
# ============================================================================

# fixed spacing by mark count — SAME every time
SPACE_BY_MARKS = {0: "2cm", 1: "2cm", 2: "3cm", 3: "4cm", 4: "5cm", 5: "6cm", 6: "7cm"}
def _space_for(marks):
    if marks is None:
        return "2cm"
    return SPACE_BY_MARKS.get(marks, "8cm" if marks > 6 else "2cm")

def _answer_block(part):
    """Generate a consistent answer blank based on answer_type."""
    at = part.get("answer_type", "none")
    label = part.get("answer_label", "") or ""
    unit = part.get("answer_unit", "") or ""
    if at == "line":
        return "\n\\hfill \\underline{\\hspace{5cm}}\n\\vspace{0.5cm}"
    if at == "line_unit":
        lab = f"{label} = " if label else ""
        return f"\n\\hfill {lab}\\underline{{\\hspace{{5cm}}}} {unit}\n\\vspace{{0.5cm}}"
    if at == "equation_box":
        lab = label if label else "x"
        return f"\n\\[ {lab} = \\dotfill \\]\n\\vspace{{0.3cm}}"
    if at == "two_lines":
        return ("\n\\hfill \\underline{\\hspace{4cm}}\n\\vspace{0.3cm}"
                "\n\\hfill \\underline{\\hspace{4cm}}\n\\vspace{0.5cm}")
    return ""

def _render_table(tbl):
    headers = tbl.get("headers", [])
    rows = tbl.get("rows", [])
    ncol = max(len(headers), max((len(r) for r in rows), default=0)) or 2
    spec = "|" + "c|" * ncol
    out = ["\\begin{center}", f"\\begin{{tabular}}{{{spec}}}", "\\hline"]
    if headers:
        out.append(" & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\ \\hline")
    for r in rows:
        cells = list(r) + [""] * (ncol - len(r))
        out.append(" & ".join(str(c) for c in cells) + " \\\\ \\hline")
    out += ["\\end{tabular}", "\\end{center}"]
    return "\n".join(out)

def _render_mcq(options):
    out = ["\\begin{center}", "\\begin{tabular}{|c|l|}", "\\hline"]
    letters = ["A", "B", "C", "D", "E", "F"]
    for i, opt in enumerate(options):
        out.append(f"{letters[i]} & {opt} \\\\ \\hline")
    out += ["\\end{tabular}", "\\end{center}"]
    return "\n".join(out)

def _render_bullets(bullets):
    out = ["\\begin{itemize}"]
    for b in bullets:
        out.append(f"    \\item {b}")
    out.append("\\end{itemize}")
    return "\n".join(out)

def _figure_placeholder(n):
    return ("\\begin{center}\n"
            f"\\includegraphics[width=0.6\\textwidth]{{__FIGURE_{n}__}}\n"
            "\\end{center}")

def _render_part(part, fig_counter):
    """Render one part into LaTeX. Returns (latex, new_fig_counter)."""
    lines = []
    text = part.get("text", "").strip()

    # figure (if it belongs in this part)
    if part.get("figure_here"):
        fig_counter[0] += 1
        # figure goes before the question text typically
    # table
    if part.get("is_table") and part.get("table"):
        if text:
            lines.append(text)
        lines.append(_render_table(part["table"]))
        text = ""  # consumed
    # text
    if text:
        lines.append(text)
    # bullets
    if part.get("bullets"):
        lines.append(_render_bullets(part["bullets"]))
    # figure placeholder
    if part.get("figure_here"):
        lines.append(_figure_placeholder(fig_counter[0]))
    # mcq
    if part.get("is_mcq") and part.get("mcq_options"):
        lines.append(_render_mcq(part["mcq_options"]))

    body = "\n".join(lines)

    # marks
    marks = part.get("marks")
    if marks is not None:
        body += f" \\hfill ({marks})"

    # working space
    body += f"\n\\vspace{{{_space_for(marks)}}}"

    # answer block
    body += _answer_block(part)

    return body, fig_counter

def generate_latex(structured):
    """Deterministically build the question body LaTeX from structured data."""
    fig_counter = [0]
    blocks = []

    intro = (structured.get("intro") or "").strip()
    if intro:
        blocks.append(intro)

    parts = structured.get("parts", [])
    # detect if there are real sub-parts (more than one, or labeled)
    has_subparts = len(parts) > 1 or (len(parts) == 1 and parts[0].get("label"))

    if not has_subparts and len(parts) == 1:
        # single question, no enumerate
        body, fig_counter = _render_part(parts[0], fig_counter)
        blocks.append(body)
    else:
        # group by nesting: build enumerate with possible nested enumerate
        out = ["\\begin{enumerate}"]
        i = 0
        while i < len(parts):
            p = parts[i]
            if p.get("level", 0) == 0:
                body, fig_counter = _render_part(p, fig_counter)
                # check for nested level-1 parts following
                nested = []
                j = i + 1
                while j < len(parts) and parts[j].get("level", 0) == 1:
                    nested.append(parts[j])
                    j += 1
                if nested:
                    out.append(f"  \\item {body}")
                    out.append("  \\begin{enumerate}")
                    for np in nested:
                        nbody, fig_counter = _render_part(np, fig_counter)
                        out.append(f"    \\item {nbody}")
                    out.append("  \\end{enumerate}")
                    i = j
                else:
                    out.append(f"  \\item {body}")
                    i += 1
            else:
                # stray nested part without parent — treat as item
                body, fig_counter = _render_part(p, fig_counter)
                out.append(f"  \\item {body}")
                i += 1
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
    content_parts.append(f"Extract the structure of this exam question. Return valid JSON only.")

    model = genai.GenerativeModel(MODEL, system_instruction=SYSTEM_PROMPT)
    try:
        resp = model.generate_content(
            content_parts,
            generation_config={"response_mime_type": "application/json", "temperature": 0.1},
        )
        text = resp.text.strip()
    except Exception as e:
        raise HTTPException(502, f"Gemini call failed: {e}")

    structured = _parse_json(text)
    if structured is None:
        raise HTTPException(502, f"Model did not return valid JSON:\n{text[:500]}")

    # GENERATE LATEX DETERMINISTICALLY from the structured data
    try:
        latex = generate_latex(structured)
    except Exception as e:
        raise HTTPException(500, f"LaTeX generation failed: {e}")

    total = structured.get("totalMarks")
    figures = structured.get("figures", []) or []

    # crop figures
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
            png_buf = io.BytesIO(); crop.save(png_buf, format="PNG"); png = png_buf.getvalue()
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
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(re.sub(r'(?<!\\)\n', r'\\n', text))
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(re.sub(r'(?<!\\)\n', r'\\n', match.group()))
        except json.JSONDecodeError:
            pass
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
        items.append(f"\\item\n{body}{total}\n\\hline")

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
        f"\\date{{{p.date}}}\n\\maketitle\n\\hline\n\\begin{{enumerate}}\n\n"
        f"{items}\n\n\\end{{enumerate}}\n\n\\end{{document}}\n"
    )


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL, "key_set": bool(os.environ.get("GEMINI_API_KEY")),
            "figures_cached": len(FIGURE_STORE)}
