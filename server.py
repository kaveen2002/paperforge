"""
PaperForge backend
==================
Two endpoints that turn the prototype from Demo mode into a working product:

  POST /convert   ->  image + question number  =>  { latex, totalMarks, figures, crops }
                      (calls Claude once: gets house-style LaTeX, the total mark,
                       AND figure bounding boxes; then crops each figure to N.png)

  POST /export    ->  full paper state         =>  a zip of paper.tex + all N.png figures

Run:
  pip install fastapi uvicorn anthropic pillow python-multipart
  export ANTHROPIC_API_KEY=sk-ant-...        # key stays server-side, never in the browser
  uvicorn server:app --reload --port 8000

Then in the web app: Live mode, endpoint = http://localhost:8000/convert
"""

import os, io, json, base64, zipfile, re
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from PIL import Image
import anthropic

app = FastAPI(title="PaperForge")

# allow the web/phone client to call us
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"],
)

# serve the front-end page (index.html sits next to this file) so the whole
# app is ONE deployable unit: opening the site root loads the UI, and the UI
# calls /convert and /export on this same server.
HERE = os.path.dirname(os.path.abspath(__file__))

@app.get("/")
def home():
    index = os.path.join(HERE, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"ok": True, "model": MODEL, "note": "index.html not found beside server.py"}

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
MODEL = "claude-opus-4-7"

# in-memory figure store for this demo; production -> object storage (S3/GCS) keyed by paper id
FIGURE_STORE = {}   # filename -> PNG bytes

# ---------------------------------------------------------------------------
# The house-style rules. This is the heart of the conversion: it encodes
# exactly the formatting the teacher wants, and asks for a STRICT JSON reply
# so the server can parse LaTeX, marks, and crop boxes in one call.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = r"""
You convert a screenshot of ONE exam question into Edexcel-style LaTeX.

Output STRICT JSON only (no prose, no markdown fences), with this shape:
{
  "latex": "<the \\item body, WITHOUT the \\item line and WITHOUT the total-marks line>",
  "totalMarks": <integer or null if no total is printed in the image>,
  "figures": [ {"x":<0-1>,"y":<0-1>,"w":<0-1>,"h":<0-1>}, ... ]   // empty if none
}

LaTeX rules:
- Preserve wording, punctuation, symbols, bold, italics, fractions, surds, powers EXACTLY.
- Sub-parts use nested \begin{enumerate} ... \end{enumerate}.
- Inline maths \( \); display maths \[ \].
- Marks per part as:  \hfill (2)
- Leave working space with \vspace: 2-3cm short, 4-5cm medium, 6-8cm long.
- For each figure in the image, insert, at the correct position in the flow:
    \begin{center}
    \includegraphics[width=0.7\textwidth]{__FIGURE_n__}
    \end{center}
  where n is 1,2,3... in order of appearance. The server replaces __FIGURE_n__
  with the real filename.
- Do NOT add the \item line; do NOT add the (Total for Question ...) line; the server adds those.

figures:
- Give a bounding box for EVERY diagram/photo/figure (NOT for tables, NOT for text).
- Coordinates are fractions of the image (x,y = top-left corner; w,h = size).
- Box only the figure artwork, excluding caption text and surrounding question text.
- Order boxes the same as the __FIGURE_n__ placeholders.
"""


# ============================ /convert =====================================
@app.post("/convert")
async def convert(image: UploadFile = File(...), questionNumber: int = Form(...)):
    raw = await image.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read image")
    W, H = img.size

    b64 = base64.standard_b64encode(raw).decode()
    media = image.content_type or "image/png"

    # --- single Claude call: LaTeX + marks + figure boxes ---
    msg = client.messages.create(
        model=MODEL, max_tokens=2000, system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}},
                {"type": "text", "text": f"Convert this as Question {questionNumber}. JSON only."},
            ],
        }],
    )
    text = "".join(b.text for b in msg.content if b.type == "text").strip()
    text = re.sub(r"^```(json)?|```$", "", text.strip()).strip()  # strip any fences
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise HTTPException(502, f"Model did not return valid JSON:\n{text[:400]}")

    latex = data.get("latex", "")
    total = data.get("totalMarks")
    figures = data.get("figures", []) or []

    # --- crop each figure box -> store + return a preview so the UI shows it ---
    crops = []
    for i, box in enumerate(figures, start=1):
        rect = _frac_to_px(box, W, H, pad=0.01)
        crop = img.crop(rect)
        png = io.BytesIO(); crop.save(png, format="PNG"); png = png.getvalue()
        # temp name keyed to this question; /export reassigns the GLOBAL number
        tmp_name = f"q{questionNumber}_fig{i}.png"
        FIGURE_STORE[tmp_name] = png
        preview = "data:image/png;base64," + base64.standard_b64encode(png).decode()
        crops.append({"placeholder": f"__FIGURE_{i}__", "tempName": tmp_name,
                      "box": box, "rect": rect, "dataUrl": preview})

    return {
        "latex": latex,
        "totalMarks": total,
        "marksFound": total is not None,
        "figures": figures,
        "crops": crops,   # client shows these for optional manual nudge before locking
    }


def _frac_to_px(box, W, H, pad=0.0):
    x = int(box["x"] * W); y = int(box["y"] * H)
    w = int(box["w"] * W); h = int(box["h"] * H)
    px = int(pad * W); py = int(pad * H)
    return (max(0, x - px), max(0, y - py), min(W, x + w + px), min(H, y + h + py))


# ============================ /export ======================================
class QIn(BaseModel):
    body: str                 # latex with __FIGURE_n__ placeholders
    marks: Optional[int] = None
    tempImageNames: List[str] = []   # the q#_fig# names from /convert, in order

class PaperIn(BaseModel):
    title: str; author: str; cred: str; inst: str; contact: str; date: str
    questions: List[QIn]

@app.post("/export")
def export(paper: PaperIn):
    # global numbering pass: assign 1.png,2.png... across the whole paper
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

    # verify every referenced image is packaged
    refs = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex)
    missing = [r for r in refs if r not in packaged]
    if missing:
        raise HTTPException(409, f"Missing figures for: {missing}")

    # zip .tex + figures
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
    return {"ok": True, "model": MODEL, "figures_cached": len(FIGURE_STORE)}
