"""
PaperForge backend — Gemini edition (production-tuned)
======================================================
POST /convert  ->  image(s) + questionNumber  =>  { latex, totalMarks, figures, crops }
POST /export   ->  full paper state            =>  zip of paper.tex + all N.png figures
GET  /         ->  serves index.html
GET  /health   ->  status check
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
# THE PROMPT — tuned against 3 real hand-written papers, every pattern matched
# ============================================================================
SYSTEM_PROMPT = r"""
You convert a screenshot of ONE exam question into Edexcel-style LaTeX.

Return STRICT VALID JSON (properly escaped, no markdown fences):
{
  "latex": "<question body only>",
  "totalMarks": <integer or null if not visible>,
  "figures": [{"box_2d":[ymin,xmin,ymax,xmax],"label":"desc","image_index":0}]
}

RULES:

1. COPY wording EXACTLY. Never rephrase, reorder, or skip anything.

2. "latex" = question body only. The server adds \item, the total-marks line, and \hline.
   Do NOT include \item. Do NOT include "Total for Question" line. Do NOT include \hline.

3. SUB-PARTS: use \begin{enumerate}...\end{enumerate}. Auto-labels only.
   NEVER type (a), (b), (i), (ii) manually at start of \item.

4. MARKS: \hfill (N) after question text, then \vspace on next line:
   Calculate the speed. \hfill (2)
   \vspace{3cm}

5. SPACING by marks (be generous for calculations):
   1-mark state/name: \vspace{2cm}
   2-mark calculate: \vspace{3cm}
   3-mark explain/describe: \vspace{4cm}
   4-mark: \vspace{5cm}
   5+ mark: \vspace{6cm} to \vspace{8cm}
   Tiny gaps between conditions: \vspace{0.3cm}
   After answer blanks: \vspace{0.5cm}

6. MATHS: inline \( \), display \[ \]. \dfrac inline, \frac display.
   Units: thin-space then text (220\,m, 0.70\,s). Degrees: ^\circ.
   Vectors: \mathbf{a}, \overrightarrow{OA}. Matrices: \begin{pmatrix}.
   Aligned: \begin{aligned}...\end{aligned} inside \[ \].

7. TABLES:
   \begin{center}
   \begin{tabular}{|c|c|}
   \hline
   \textbf{Header} & \textbf{Header} \\ \hline
   data & data \\ \hline
   \end{tabular}
   \end{center}

8. MCQ: tabular |c|l| with A/B/C/D rows.

9. WORD BOXES: single-row tabular |c|c|c|c|.

10. ANSWER BLANKS (right-aligned with unit):
    \hfill wave speed = \underline{\hspace{5cm}} m/s
    \vspace{0.5cm}
    Or: \hfill \textbf{Answer:} \underline{\hspace{5cm}}
    Or in display: \[ x = \dotfill \]

11. BULLET LISTS: \begin{itemize}\item...\end{itemize}.

12. FIGURES at correct position:
    \begin{center}
    \includegraphics[width=0.6\textwidth]{__FIGURE_n__}
    \end{center}
    Width: 0.45-0.85\textwidth. NEVER add captions or write filename.

13. LINE BREAKS: \\ only for genuine breaks. \noindent as needed.

14. PACKAGES: only amsmath, amssymb, inputenc, geometry, array, graphicx, xcolor.
    NEVER use tikz, siunitx, booktabs, enumitem, cancel, boxed environment.

EXAMPLES:

EX1 — Simple:
The 3rd term of an arithmetic series is 25.

The sum of the first 10 terms is 350.

Find the 12th term. \hfill (5)

\vspace{6cm}

EX2 — Physics calculate + answer blank:
\begin{enumerate}
  \item A sound wave in air travels a distance of 220\,m in a time of 0.70\,s.
  \begin{enumerate}
    \item State the equation linking speed, distance and time. \hfill (1)
    \vspace{2cm}

    \item Calculate the speed of the sound wave in air. \hfill (2)
    \vspace{3cm}

    \hfill wave speed = \underline{\hspace{5cm}} m/s
    \vspace{0.5cm}
  \end{enumerate}

  \item Sound waves are longitudinal waves.
  Water waves are transverse waves.

  Describe the difference between longitudinal waves and transverse waves. \hfill (3)
  \vspace{4cm}
\end{enumerate}

EX3 — Image + answer line:
The diagram shows a shape made up of three semicircles.

\begin{center}
\includegraphics[width=0.6\textwidth]{__FIGURE_1__}
\end{center}

\( BC = CA = 6\,\text{cm} \)

Work out the perimeter of the shape.
Give your answer correct to one decimal place.

\vspace{4cm}

\hfill \underline{\hspace{3cm}} cm \hfill (4)

EX4 — Bearings + bullets:
\begin{center}
\includegraphics[width=0.7\textwidth]{__FIGURE_1__}
\end{center}

\begin{itemize}
    \item The bearing of \( B \) from \( A \) is \( 054^\circ \)
    \item The bearing of \( C \) from \( B \) is \( 132^\circ \)
\end{itemize}

Work out the total time Melur takes.
Give your answer in hours and minutes.

\vspace{3.5cm}

\hfill \underline{\hspace{2.5cm}} hours \quad \underline{\hspace{2.5cm}} minutes \hfill (5)

FIGURE BOXES:
- box_2d: [ymin, xmin, ymax, xmax] normalized 0-1000.
- TIGHT around artwork only (not text/tables/equations).
- image_index: 0-based. Empty [] if no figures.
"""

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

    # build content: all images + instruction
    content_parts = []
    for i, (raw, mime) in enumerate(raw_list, start=1):
        content_parts.append({"mime_type": mime, "data": raw})
        if len(raw_list) > 1:
            content_parts.append(f"(Screenshot {i} of {len(raw_list)} for the same question.)")
    content_parts.append(f"Convert this as Question {questionNumber}. Return valid JSON only.")

    model = genai.GenerativeModel(MODEL, system_instruction=SYSTEM_PROMPT)
    try:
        resp = model.generate_content(
            content_parts,
            generation_config={"response_mime_type": "application/json", "temperature": 0.1},
        )
        text = resp.text.strip()
    except Exception as e:
        raise HTTPException(502, f"Gemini call failed: {e}")

    # parse JSON with repair
    data = _parse_json(text)
    if data is None:
        raise HTTPException(502, f"Model did not return valid JSON:\n{text[:500]}")

    latex = data.get("latex", "")
    # SERVER CLEANUP: strip anything the model shouldn't have added
    latex = _clean_latex(latex)

    total = data.get("totalMarks")
    figures = data.get("figures", []) or []

    # crop figures using Gemini's box_2d format
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
            png_buf = io.BytesIO()
            crop.save(png_buf, format="PNG")
            png = png_buf.getvalue()
        except Exception:
            continue
        tmp_name = f"q{questionNumber}_fig{i}.png"
        FIGURE_STORE[tmp_name] = png
        preview = "data:image/png;base64," + base64.standard_b64encode(png).decode()
        crops.append({"placeholder": f"__FIGURE_{i}__", "tempName": tmp_name,
                      "box_2d": box, "rect": rect, "dataUrl": preview})

    return {"latex": latex, "totalMarks": total, "marksFound": total is not None,
            "figures": figures, "crops": crops}


def _parse_json(text):
    """Try to parse JSON, with repair for common model output issues."""
    text = re.sub(r"^```(json)?\s*|```\s*$", "", text.strip()).strip()
    # attempt 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # attempt 2: fix unescaped newlines in string values
    try:
        return json.loads(re.sub(r'(?<!\\)\n', r'\\n', text))
    except json.JSONDecodeError:
        pass
    # attempt 3: extract JSON object
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(re.sub(r'(?<!\\)\n', r'\\n', match.group()))
        except json.JSONDecodeError:
            pass
    return None


def _clean_latex(latex):
    """Strip things the model shouldn't have added — server handles these."""
    s = latex.strip()
    # remove trailing \hline (possibly multiple)
    s = re.sub(r'(\\hline\s*)+$', '', s).strip()
    # remove any Total for Question line
    s = re.sub(r'\\hfill\s*\\textbf\{[^}]*Total for Question[^}]*\}\s*', '', s).strip()
    # remove leading \item if model added it
    s = re.sub(r'^\\item\s*', '', s).strip()
    # remove trailing \hline again after other removals
    s = re.sub(r'(\\hline\s*)+$', '', s).strip()
    return s


def _box2d_to_px(box_2d, W, H, pad_frac=0.02):
    """Convert Gemini box_2d [ymin,xmin,ymax,xmax] (0-1000) to pixel rect."""
    ymin, xmin, ymax, xmax = box_2d
    for v in (ymin, xmin, ymax, xmax):
        if not (0 <= v <= 1000):
            return None
    if ymin >= ymax or xmin >= xmax:
        return None
    left   = int(xmin / 1000 * W)
    top    = int(ymin / 1000 * H)
    right  = int(xmax / 1000 * W)
    bottom = int(ymax / 1000 * H)
    px, py = int(pad_frac * W), int(pad_frac * H)
    return (max(0, left - px), max(0, top - py), min(W, right + px), min(H, bottom + py))


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
        body = _clean_latex(q.body)
        for local_i, tmp in enumerate(q.tempImageNames, start=1):
            global_idx += 1
            final = f"{global_idx}.png"
            body = body.replace(f"__FIGURE_{local_i}__", final)
            if tmp in FIGURE_STORE:
                packaged[final] = FIGURE_STORE[tmp]
        # assemble: \item + body + total marks + \hline
        total = ""
        if q.marks is not None:
            total = f"\n\n\\hfill \\textbf{{(Total for Question {qi} is {q.marks} marks)}}"
        items.append(f"\\item\n{body}{total}\n\\hline")

    tex = _build_doc(paper, "\n\n".join(items))

    # verify all referenced images are present
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
        "\\usepackage{amsmath}\n"
        "\\usepackage{amssymb}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{geometry}\n"
        "\\usepackage{array}\n"
        "\\usepackage{graphicx}\n"
        "\\geometry{margin=1in}\n\n"
        "\\begin{document}\n"
        f"\\title{{\\LARGE \\textbf{{{p.title}}}}}\n"
        f"\\author{{\\large {p.author} \\\\ \\text{{{p.cred}}} \\\\ {p.inst} \\\\ \\textbf{{Contact: {p.contact}}}}}\n"
        f"\\date{{{p.date}}}\n"
        "\\maketitle\n"
        "\\hline\n"
        "\\begin{enumerate}\n\n"
        f"{items}\n\n"
        "\\end{enumerate}\n\n"
        "\\end{document}\n"
    )


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL, "key_set": bool(os.environ.get("GEMINI_API_KEY")),
            "figures_cached": len(FIGURE_STORE)}
