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
MODEL = "gemini-2.5-flash-lite"
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
You convert a screenshot of ONE exam question into Edexcel-style LaTeX, matching a STRICT house style.

Reply with STRICT JSON ONLY (no prose, no markdown fences), shape:
{
  "latex": "<the body only: NO \\item line, NO (Total for Question...) line>",
  "totalMarks": <integer, or null if no total is printed in the image>,
  "figures": [ {"x":<0-1>,"y":<0-1>,"w":<0-1>,"h":<0-1>} ]
}

HOUSE STYLE RULES (follow exactly):
- Preserve wording, punctuation, symbols, bold, italics, fractions, surds, powers EXACTLY as in the image.
- Inline maths \( \); display maths \[ \]. Fractions \frac, surds \sqrt, units as plain text (m/s, Hz, \Omega).
- Sub-parts use a nested \begin{enumerate} ... \end{enumerate}. Let enumerate produce the (a),(b),(i),(ii)
  labels itself: DO NOT type "(a)", "(b)", "(i)" etc. at the start of an item. Just write the text.
- Marks for a part go at the END of that part as:  \hfill (2)
- After each part that needs working room, add \vspace: 2-3cm short, 4-5cm medium, 6-8cm long/algebra/proof.
- FIGURES: for each diagram/photo, insert at the right spot:
    \begin{center}
    \includegraphics[width=0.7\textwidth]{__FIGURE_n__}
    \end{center}
  n = 1,2,3... in order of appearance. NEVER add a caption, NEVER write the filename, and DO NOT
  write loose label text like "Figure 1" unless it is part of the actual question wording.
- MULTIPLE-CHOICE options: render as a tabular, one option per row, NOT as \begin{boxed} (that is invalid),
  NOT as a nested enumerate. Use exactly:
    \begin{center}
    \begin{tabular}{|c|l|}
    \hline
    A & first option \\
    \hline
    B & second option \\
    \hline
    C & third option \\
    \hline
    D & fourth option \\
    \hline
    \end{tabular}
    \end{center}
- WORD/ANSWER BOXES (choose-from-the-box): a single-row tabular with the words separated by columns:
    \begin{center}
    \begin{tabular}{|c|c|c|c|}
    \hline
    density & mass & volume & weight \\
    \hline
    \end{tabular}
    \end{center}
- Fill-in answer lines use \dotfill, e.g.  \[ x = \dotfill \]
- NEVER invent the \begin{boxed} environment. NEVER use \boxed except inside maths mode for a real boxed value.
- Do NOT add the \item line; do NOT add the (Total for Question ...) line; the server adds both.
- PACKAGE LIMIT: the document preamble loads ONLY these packages:
  amsmath, amssymb, inputenc, geometry, array, graphicx, xcolor.
  Use ONLY commands available from these (and base LaTeX). NEVER use a command that needs another
  package. In particular: do NOT use tikz, pgfplots, siunitx (\\SI, \\si), cancel, multirow, multicolumn
  beyond base, booktabs (\\toprule etc.), enumitem options, or chemfig. Write units as plain text
  (e.g. m/s, Hz, \\Omega from amssymb), vectors with \\vec or \\mathbf, and degrees as ^\\circ.

WORKED EXAMPLE (image: a part (a) multiple choice about walking speed, then part (b) a spring-balance
figure with a word box). Correct "latex" value:

\begin{enumerate}
\item Which of these speeds would be normal for a person walking? \hfill (1)

\begin{center}
\begin{tabular}{|c|l|}
\hline
A & 0.1 m/s \\
\hline
B & 1.0 m/s \\
\hline
C & 10 m/s \\
\hline
D & 100 m/s \\
\hline
\end{tabular}
\end{center}
\vspace{1cm}

\item Figure 1 shows a block hanging from a spring balance.

\begin{center}
\includegraphics[width=0.7\textwidth]{__FIGURE_1__}
\end{center}

Use a word from the box to complete the sentence below.

\begin{center}
\begin{tabular}{|c|c|c|c|}
\hline
density & mass & volume & weight \\
\hline
\end{tabular}
\end{center}

The quantity measured by the spring balance in Figure 1 is \dotfill \hfill (1)
\vspace{2cm}
\end{enumerate}

figures:
- A bounding box for EVERY diagram/photo/figure (NOT tables, NOT text, NOT word boxes).
- Coordinates are fractions of that specific image (x,y = top-left corner; w,h = size).
- Box only the figure artwork, excluding caption and surrounding text.
- Same order as the __FIGURE_n__ placeholders.
- If MULTIPLE screenshots are provided for the same question, add "image_index" (0-based) to each
  figure box indicating which screenshot it belongs to. E.g. {"x":0.2,"y":0.3,"w":0.5,"h":0.4,"image_index":1}
  means the figure is in the second screenshot. If only one screenshot, omit image_index or set to 0.
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
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise HTTPException(502, f"Model did not return valid JSON:\n{text[:400]}")

    latex = data.get("latex", "")
    total = data.get("totalMarks")
    figures = data.get("figures", []) or []

    # figures now include an "image_index" (0-based) indicating which screenshot
    # the figure is in. Default to 0 if not provided (single-image case).
    crops = []
    for i, box in enumerate(figures, start=1):
        img_idx = box.get("image_index", 0)
        if img_idx >= len(img_list):
            img_idx = 0
        src_img = img_list[img_idx]
        W, H = src_img.size
        try:
            rect = _frac_to_px(box, W, H, pad=0.01)
            crop = src_img.crop(rect)
            png = io.BytesIO(); crop.save(png, format="PNG"); png = png.getvalue()
        except Exception:
            continue
        tmp_name = f"q{questionNumber}_fig{i}.png"
        FIGURE_STORE[tmp_name] = png
        preview = "data:image/png;base64," + base64.standard_b64encode(png).decode()
        crops.append({"placeholder": f"__FIGURE_{i}__", "tempName": tmp_name,
                      "box": box, "rect": rect, "dataUrl": preview})

    return {"latex": latex, "totalMarks": total, "marksFound": total is not None,
            "figures": figures, "crops": crops}


def _frac_to_px(box, W, H, pad=0.0):
    x = int(box["x"] * W); y = int(box["y"] * H)
    w = int(box["w"] * W); h = int(box["h"] * H)
    px = int(pad * W); py = int(pad * H)
    return (max(0, x - px), max(0, y - py), min(W, x + w + px), min(H, y + h + py))


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
