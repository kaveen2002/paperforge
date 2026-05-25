"""
PaperForge backend — Gemini (free-tier) edition
================================================
Same app, but the conversion call uses Google's Gemini Flash, which has a
generous FREE tier through Google AI Studio. Everything else — cropping,
numbering, export/zip — is identical and model-independent.

  POST /convert  ->  image + question number  =>  { latex, totalMarks, figures, crops }
  POST /export   ->  full paper state          =>  zip of paper.tex + all N.png figures
  GET  /         ->  serves the web UI (index.html beside this file)

Run locally:
  pip install -r requirements.txt
  export GEMINI_API_KEY=your-google-ai-studio-key      # free key, stays server-side
  uvicorn server:app --reload --port 8000
  # open http://localhost:8000

Get a free key: https://aistudio.google.com/apikey
"""

import os, io, json, base64, zipfile, re
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from PIL import Image
import google.generativeai as genai

# ---- model config ----------------------------------------------------------
MODEL = "gemini-2.5-flash"
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

app = FastAPI(title="PaperForge (Gemini)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

HERE = os.path.dirname(os.path.abspath(__file__))

@app.get("/")
def home():
    index = os.path.join(HERE, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"ok": True, "model": MODEL, "note": "index.html not found beside server.py"}

FIGURE_STORE = {}   # tempName -> PNG bytes

SYSTEM_PROMPT = r"""
You convert a screenshot of ONE exam question into Edexcel-style LaTeX. Match the house style EXACTLY.

OUTPUT: STRICT VALID JSON ONLY (no prose, no fences). Escape all special chars in strings properly.
{
  "latex": "<question body>",
  "totalMarks": <integer or null>,
  "figures": [ {"box_2d": [ymin, xmin, ymax, xmax], "label": "desc", "image_index": 0} ]
}

=== CORE RULES ===

1. WORDING: Copy EXACTLY as shown. Never change wording, punctuation, symbols. Never rephrase or skip.

2. STRUCTURE:
   - "latex" = question body only.
   - Do NOT write \item at the start (server adds it).
   - Do NOT write a "Total for Question" line (server adds it).
   - Sub-parts: nested \begin{enumerate}...\end{enumerate}, let enumerate auto-label.
     NEVER manually type (a), (b), (i), (ii) at start of \item.
   - End each question with \hline as separator.

3. MARKS: Each part ends with \hfill (N) AFTER question text, BEFORE \vspace:
   Work out the value of x. \hfill (3)
   \vspace{4cm}

4. MATHS:
   - Inline: \( ... \), use \dfrac for inline fractions.
   - Display: \[ ... \], use \frac in display.
   - Units: plain text (m/s, km, cm, Hz, g/cm^3). Greek: \Omega, \pi.
   - Vectors: \mathbf{a}, \vec{a}, \overrightarrow{OA}.
   - Degrees: ^\circ. Currency: pounds directly or \pounds.
   - Aligned equations: \begin{aligned}...\end{aligned} inside \[ \].

5. WORKING SPACE - match marks to space (be generous for calculations):
   - 1 mark (state/name/write down): \vspace{2cm}
   - 2 marks (calculate/work out): \vspace{3cm}
   - 3 marks (explain/describe/calculate): \vspace{4cm}
   - 4 marks: \vspace{5cm}
   - 5+ marks: \vspace{6cm} to \vspace{8cm}
   NEVER exceed 10cm. Use \vspace{0.3cm} for small visual gaps between conditions/statements.
   Add \vspace{0.5cm} after answer blank lines.

6. TABLES:
   \begin{center}
   \begin{tabular}{|c|c|}
   \hline
   \textbf{Header} & \textbf{Header} \\ \hline
   data & data \\ \hline
   \end{tabular}
   \end{center}

7. MULTIPLE CHOICE (A/B/C/D):
   \begin{center}
   \begin{tabular}{|c|l|}
   \hline
   A & option \\ \hline
   B & option \\ \hline
   C & option \\ \hline
   D & option \\ \hline
   \end{tabular}
   \end{center}

8. WORD BOXES:
   \begin{center}
   \begin{tabular}{|c|c|c|c|}
   \hline
   word1 & word2 & word3 & word4 \\ \hline
   \end{tabular}
   \end{center}

9. ANSWER LINES: Place answer blanks on their own line, right-aligned with \hfill:
   \hfill wave speed = \underline{\hspace{5cm}} m/s
   or:
   \hfill \textbf{Answer:} \underline{\hspace{5cm}}
   or for labelled values:
   \hfill \( x = \) \underline{\hspace{4cm}}
   Use \dotfill for fill-in-blank in display maths: \[ x = \dotfill \]
   ALWAYS add \vspace{0.5cm} AFTER an answer blank line before the next part.

10. BULLET LISTS: \begin{itemize} \item ... \end{itemize} for bullet points.

11. FIGURES: Insert at correct position:
    \begin{center}
    \includegraphics[width=0.6\textwidth]{__FIGURE_n__}
    \end{center}
    n=1,2,3 in order. Server replaces __FIGURE_n__.
    NEVER add captions. Width 0.45-0.85\textwidth by content.

12. LINE BREAKS: \\ only for genuine line breaks. \vspace{0.3cm} for small gaps. \noindent as needed.

13. PACKAGES: ONLY amsmath, amssymb, inputenc, geometry, array, graphicx, xcolor.
    NEVER use tikz, siunitx, booktabs, enumitem, cancel.
    MAY use base LaTeX: itemize, tabbing, minipage, flushright, quote, medskip.

14. FORBIDDEN: \begin{boxed}, manual (a)/(b) labels, explanations, excessive spacing.

=== EXAMPLES ===

EX1 — Simple:

The 3rd term of an arithmetic series is 25.

The sum of the first 10 terms is 350.

Find the 12th term. \hfill (5)

\vspace{6cm}

\hline

EX2 — Physics with calculate + answer blank:

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
\hline

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

\hline

EX4 — Bearings with bullets:

The diagram shows the positions of three villages, \( A, B \) and \( C \).

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

\hline

=== FIGURE BOUNDING BOXES ===

Use Gemini's native detection. For EVERY diagram/photo (NOT tables/text/equations):
- "box_2d": [ymin, xmin, ymax, xmax] normalized 0-1000.
- TIGHT box around artwork only, exclude surrounding text/captions.
- Same order as __FIGURE_n__ placeholders.
- "image_index": 0-based (which screenshot). [] if no figures.
"""

# ============================ /convert =====================================
@app.post("/convert")
async def convert(images: List[UploadFile] = File(...), questionNumber: int = Form(...)):
    # read all uploaded images
    raw_list = []
    img_list = []
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

    # build the Gemini content: all images + the instruction
    content_parts = []
    for i, (raw, mime) in enumerate(raw_list, start=1):
        content_parts.append({"mime_type": mime, "data": raw})
        if len(raw_list) > 1:
            content_parts.append(f"(This is screenshot {i} of {len(raw_list)} for the same question.)")
    content_parts.append(f"Convert this as Question {questionNumber}. JSON only.")

    model = genai.GenerativeModel(MODEL, system_instruction=SYSTEM_PROMPT)
    try:
        resp = model.generate_content(
            content_parts,
            generation_config={"response_mime_type": "application/json", "temperature": 0.2},
        )
        text = resp.text.strip()
    except Exception as e:
        raise HTTPException(502, f"Gemini call failed: {e}")

    text = re.sub(r"^```(json)?|```$", "", text.strip()).strip()
    
    # attempt to fix common JSON issues from the model
    # fix unescaped newlines inside string values
    def repair_json(s):
        # try parsing as-is first
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        # try fixing unescaped newlines in string values
        try:
            fixed = re.sub(r'(?<!\\)\n', r'\\n', s)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        # try extracting just the JSON object
        match = re.search(r'\{.*\}', s, re.DOTALL)
        if match:
            try:
                fixed = re.sub(r'(?<!\\)\n', r'\\n', match.group())
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass
        return None
    
    data = repair_json(text)
    if data is None:
        raise HTTPException(502, f"Model did not return valid JSON:\n{text[:400]}")

    latex = data.get("latex", "")
    total = data.get("totalMarks")
    figures = data.get("figures", []) or []

    # figures now use Gemini's native box_2d format: [ymin, xmin, ymax, xmax] scaled 0-1000
    crops = []
    for i, fig in enumerate(figures, start=1):
        img_idx = fig.get("image_index", 0)
        if img_idx >= len(img_list):
            img_idx = 0
        src_img = img_list[img_idx]
        W, H = src_img.size
        try:
            box = fig.get("box_2d", [])
            if not box or len(box) != 4:
                continue
            rect = _box2d_to_px(box, W, H, pad_frac=0.02)
            if rect is None:
                continue
            crop = src_img.crop(rect)
            # skip tiny or degenerate crops
            if crop.width < 10 or crop.height < 10:
                continue
            png = io.BytesIO(); crop.save(png, format="PNG"); png = png.getvalue()
        except Exception:
            continue
        tmp_name = f"q{questionNumber}_fig{i}.png"
        FIGURE_STORE[tmp_name] = png
        preview = "data:image/png;base64," + base64.standard_b64encode(png).decode()
        crops.append({"placeholder": f"__FIGURE_{i}__", "tempName": tmp_name,
                      "box_2d": box, "rect": rect, "dataUrl": preview})

    return {"latex": latex, "totalMarks": total, "marksFound": total is not None,
            "figures": figures, "crops": crops}


def _box2d_to_px(box_2d, W, H, pad_frac=0.02):
    """Convert Gemini box_2d [ymin, xmin, ymax, xmax] (0-1000) to pixel rect (left, top, right, bottom).
    
    Gemini returns coordinates on a virtual 1000x1000 grid.
    ymin/ymax are vertical (top/bottom), xmin/xmax are horizontal (left/right).
    We descale to actual image pixels, add padding, and clamp to image bounds.
    """
    ymin, xmin, ymax, xmax = box_2d
    
    # validate: all coords should be 0-1000 range
    for v in (ymin, xmin, ymax, xmax):
        if not (0 <= v <= 1000):
            return None
    # validate: min < max
    if ymin >= ymax or xmin >= xmax:
        return None
    
    # descale from 1000-grid to actual pixels
    left   = int(xmin / 1000 * W)
    top    = int(ymin / 1000 * H)
    right  = int(xmax / 1000 * W)
    bottom = int(ymax / 1000 * H)
    
    # add padding (percentage of image dimensions)
    pad_x = int(pad_frac * W)
    pad_y = int(pad_frac * H)
    left   = max(0, left - pad_x)
    top    = max(0, top - pad_y)
    right  = min(W, right + pad_x)
    bottom = min(H, bottom + pad_y)
    
    return (left, top, right, bottom)


# ============================ /export ======================================
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
        body = q.body
        for local_i, tmp in enumerate(q.tempImageNames, start=1):
            global_idx += 1
            final = f"{global_idx}.png"
            body = body.replace(f"__FIGURE_{local_i}__", final)
            if tmp in FIGURE_STORE:
                packaged[final] = FIGURE_STORE[tmp]
        total = (f"\n\\hfill \\textbf{{(Total for Question {qi} is {q.marks} marks)}}"
                 if q.marks is not None else "")
        items.append(f"\\item\n{body}\n{total}\n\\hline")

    tex = _doc(paper, "\n\n".join(items))

    refs = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex)
    missing = [r for r in refs if r not in packaged]
    if missing:
        raise HTTPException(409, f"Missing figures for: {missing}")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("paper.tex", tex)
        for name, png in packaged.items():
            z.writestr(name, png)
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="paper_export.zip"'})


def _doc(p: PaperIn, items: str) -> str:
    return (
        "\\documentclass[a4paper,12pt]{article}\n"
        "\\usepackage{amsmath}\\usepackage{amssymb}\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{geometry}\\usepackage{array}\\usepackage{graphicx}\\usepackage{xcolor}\n"
        "\\geometry{margin=1in}\n\\begin{document}\n"
        f"\\title{{\\LARGE \\textbf{{{p.title}}}}}\n"
        f"\\author{{\\large {p.author} \\\\ \\text{{{p.cred}}} \\\\ {p.inst} \\\\ \\textbf{{Contact: {p.contact}}}}}\n"
        f"\\date{{{p.date}}}\n\\maketitle\n\\hline\n\\begin{{enumerate}}\n\n"
        f"{items}\n\n\\end{{enumerate}}\n\\end{{document}}"
    )


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL, "key_set": bool(os.environ.get("GEMINI_API_KEY")),
            "figures_cached": len(FIGURE_STORE)}
