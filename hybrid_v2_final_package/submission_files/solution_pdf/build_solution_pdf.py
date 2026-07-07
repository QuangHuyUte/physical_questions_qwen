from __future__ import annotations

from pathlib import Path
from textwrap import wrap


OUT_DIR = Path(__file__).resolve().parent
PDF_PATH = OUT_DIR / "EXACT2026_HybridV2_Solution.pdf"

W, H = 595, 842
M = 46


def safe(text: str) -> str:
    replacements = {
        "\\": "\\\\",
        "(": "\\(",
        ")": "\\)",
        "–": "-",
        "—": "-",
        "×": "x",
        "²": "^2",
        "μ": "u",
        "Ω": "ohm",
        "Δ": "Delta",
    }
    out = str(text)
    for a, b in replacements.items():
        out = out.replace(a, b)
    return out.encode("latin-1", errors="replace").decode("latin-1")


class PDF:
    def __init__(self) -> None:
        self.objects: list[bytes] = []
        self.pages: list[int] = []
        self.f1 = self.obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        self.f2 = self.obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
        self.f3 = self.obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")

    def obj(self, data: bytes) -> int:
        self.objects.append(data)
        return len(self.objects)

    def page(self, stream_text: str) -> None:
        stream = stream_text.encode("latin-1", errors="replace")
        content = self.obj(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
        page = self.obj(
            (
                f"<< /Type /Page /Parent 0 0 R /MediaBox [0 0 {W} {H}] "
                f"/Resources << /Font << /F1 {self.f1} 0 R /F2 {self.f2} 0 R /F3 {self.f3} 0 R >> >> "
                f"/Contents {content} 0 R >>"
            ).encode()
        )
        self.pages.append(page)

    def save(self, path: Path) -> None:
        kids = " ".join(f"{p} 0 R" for p in self.pages)
        pages = self.obj(f"<< /Type /Pages /Kids [{kids}] /Count {len(self.pages)} >>".encode())
        catalog = self.obj(f"<< /Type /Catalog /Pages {pages} 0 R >>".encode())
        patched = []
        for idx, data in enumerate(self.objects, 1):
            if idx in self.pages:
                data = data.replace(b"/Parent 0 0 R", f"/Parent {pages} 0 R".encode())
            patched.append(data)

        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for idx, data in enumerate(patched, 1):
            offsets.append(len(out))
            out.extend(f"{idx} 0 obj\n".encode())
            out.extend(data)
            out.extend(b"\nendobj\n")
        xref = len(out)
        out.extend(f"xref\n0 {len(patched) + 1}\n".encode())
        out.extend(b"0000000000 65535 f \n")
        for off in offsets[1:]:
            out.extend(f"{off:010d} 00000 n \n".encode())
        out.extend(f"trailer << /Size {len(patched) + 1} /Root {catalog} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
        path.write_bytes(out)


class Canvas:
    def __init__(self, page_no: int, title: str) -> None:
        self.ops: list[str] = []
        self.page_no = page_no
        self.title = title
        self.header()

    def rgb(self, r: float, g: float, b: float, stroke: bool = False) -> None:
        self.ops.append(f"{r:.3f} {g:.3f} {b:.3f} {'RG' if stroke else 'rg'}")

    def rect(self, x, y, w, h, fill=None, stroke=None, sw=1) -> None:
        if fill:
            self.rgb(*fill)
            self.ops.append(f"{x:.1f} {y:.1f} {w:.1f} {h:.1f} re f")
        if stroke:
            self.rgb(*stroke, stroke=True)
            self.ops.append(f"{sw:.1f} w {x:.1f} {y:.1f} {w:.1f} {h:.1f} re S")

    def line(self, x1, y1, x2, y2, color=(0.78, 0.82, 0.86), sw=1) -> None:
        self.rgb(*color, stroke=True)
        self.ops.append(f"{sw:.1f} w {x1:.1f} {y1:.1f} m {x2:.1f} {y2:.1f} l S")

    def text(self, x, y, text, size=9.5, font="F1", color=(0.12, 0.14, 0.18)) -> None:
        self.rgb(*color)
        self.ops.append(f"BT /{font} {size:.1f} Tf {x:.1f} {y:.1f} Td ({safe(text)}) Tj ET")

    def para(self, x, y, width, text, size=9.2, leading=12.5, font="F1", color=(0.20, 0.22, 0.27)) -> float:
        for line in wrap(str(text), max(18, int(width / (size * 0.53)))):
            self.text(x, y, line, size, font, color)
            y -= leading
        return y

    def bullet(self, x, y, text, width, size=9.0) -> float:
        self.text(x, y, "-", size, "F2", (0.06, 0.31, 0.50))
        return self.para(x + 13, y, width - 13, text, size=size, leading=12)

    def header(self) -> None:
        self.text(M, H - 36, "EXACT 2026 Hybrid V2", 8.5, "F2", (0.18, 0.34, 0.48))
        self.text(M, H - 58, self.title, 16.5, "F2", (0.06, 0.09, 0.14))
        self.line(M, H - 70, W - M, H - 70)

    def footer(self) -> None:
        self.line(M, 34, W - M, 34, (0.87, 0.89, 0.92), 0.6)
        self.text(M, 20, "Calculator-first API: LLMs parse/explain; deterministic physics computes numeric answers.", 7.7, "F1", (0.42, 0.45, 0.50))
        self.text(W - M - 22, 20, str(self.page_no), 8, "F2", (0.42, 0.45, 0.50))

    def card(self, x, y, w, h, title, body, accent=(0.08, 0.35, 0.55)) -> None:
        self.rect(x, y, w, h, fill=(0.985, 0.99, 1.0), stroke=(0.80, 0.85, 0.90), sw=0.8)
        self.rect(x, y + h - 7, w, 7, fill=accent)
        self.text(x + 11, y + h - 25, title, 10.8, "F2", (0.07, 0.10, 0.15))
        self.para(x + 11, y + h - 43, w - 22, body, 8.3, 11)

    def flow(self, x, y, w, h, title, body, fill=(0.96, 0.98, 1.0)) -> None:
        self.rect(x, y, w, h, fill=fill, stroke=(0.72, 0.80, 0.88), sw=0.7)
        self.text(x + 9, y + h - 18, title, 9.3, "F2", (0.05, 0.22, 0.36))
        self.para(x + 9, y + h - 33, w - 18, body, 7.4, 9.5)

    def out(self) -> str:
        self.footer()
        return "\n".join(self.ops)


def table(c: Canvas, x: float, y: float, cols: list[int], headers: list[str], rows: list[list[str]], row_h=49) -> float:
    total = sum(cols)
    c.rect(x, y - 24, total, 24, fill=(0.07, 0.20, 0.34))
    cx = x
    for i, h in enumerate(headers):
        c.text(cx + 7, y - 16, h, 8.3, "F2", (1, 1, 1))
        cx += cols[i]
    y -= 24
    for r, row in enumerate(rows):
        fill = (0.985, 0.99, 1.0) if r % 2 == 0 else (0.95, 0.97, 0.985)
        c.rect(x, y - row_h, total, row_h, fill=fill, stroke=(0.82, 0.86, 0.90), sw=0.5)
        cx = x
        for i, cell in enumerate(row):
            c.para(cx + 7, y - 14, cols[i] - 14, cell, 7.4, 9.3)
            cx += cols[i]
        y -= row_h
    return y


def build() -> None:
    pdf = PDF()

    # Cover
    c = Canvas(1, "Solution Overview")
    c.rect(0, H - 210, W, 210, fill=(0.06, 0.16, 0.28))
    c.rect(0, H - 210, W, 38, fill=(0.06, 0.34, 0.54))
    c.text(M, H - 105, "Hybrid V2 Physics API", 25, "F2", (1, 1, 1))
    c.para(M, H - 136, 470, "Calculator-first inference with Qwen2.5-3B LoRA adapters for numeric parsing, locked-trace explanation, and conceptual Type 1 / CHLT reasoning.", 11, 15, color=(0.88, 0.94, 0.98))
    c.card(M, 540, 240, 110, "Main principle", "Use LLMs for language-heavy work, but keep arithmetic inside deterministic physics formulas.", (0.06, 0.34, 0.54))
    c.card(M + 262, 540, 240, 110, "Numeric authority", "Type 2 answers are recomputed by physics_calculator_v2 after payload validation.", (0.17, 0.48, 0.34))
    c.card(M, 395, 240, 110, "CHLT clarification", "CHLT is Yes / No / Uncertain, but it may still require numeric calculation before choosing the label.", (0.78, 0.42, 0.14))
    c.card(M + 262, 395, 240, 110, "Final package", "The runtime uses one 3B base model and three LoRA adapters: numeric parser, trace explainer, and conceptual reasoner.", (0.48, 0.34, 0.70))
    c.text(M, 320, "Why this design", 14, "F2")
    y = 296
    for item in [
        "Fast deterministic paths solve common numeric physics without waiting for vLLM.",
        "Unseen wording is handled by a parser adapter, then verified by calculator code.",
        "Explanation polish is optional and cannot change locked values.",
        "Conceptual questions are isolated from numeric solving.",
    ]:
        y = c.bullet(M, y, item, 500) - 3
    pdf.page(c.out())

    # Architecture
    c = Canvas(2, "Runtime Architecture")
    c.flow(55, 710, 105, 58, "Input", "Single query or batch at /predict.")
    c.flow(205, 710, 130, 58, "Router", "Separates Type 2 numeric from Type 1 conceptual.")
    c.flow(380, 710, 145, 58, "JSON response", "Answer, unit, explanation, reasoning and debug.")
    c.line(160, 739, 205, 739, sw=1.2)
    c.line(335, 739, 380, 739, sw=1.2)
    c.flow(70, 590, 175, 75, "Type 1 / CHLT branch", "Quantitative yes/no claims are calculated first. Pure conceptual CHLT uses the reasoner.", (0.965, 0.985, 0.955))
    c.flow(325, 590, 175, 75, "Type 2 numeric branch", "V1 deterministic solver and guardrails run first.", (0.96, 0.98, 1.0))
    c.line(270, 710, 157, 665, sw=1.0)
    c.line(270, 710, 412, 665, sw=1.0)
    c.flow(325, 480, 175, 75, "Parser fallback", "numeric_parser_final extracts givens, target, constraints and formula candidates.", (0.98, 0.965, 1.0))
    c.flow(325, 370, 175, 75, "Validator + calculator", "Payload is checked. Registered formulas compute the locked answer.", (1.0, 0.98, 0.94))
    c.flow(325, 260, 175, 75, "Trace explanation", "Deterministic explanation by default. Optional polish only explains locked traces.", (0.97, 0.98, 0.99))
    c.line(412, 590, 412, 555, sw=1.1)
    c.line(412, 480, 412, 445, sw=1.1)
    c.line(412, 370, 412, 335, sw=1.1)
    c.text(M, 215, "Routing guardrail", 14, "F2")
    c.para(M, 190, 505, "A numeric yes/no request should not be solved directly by the conceptual adapter. The API first computes the relevant quantity, compares it with the claim, and only then returns Yes, No, or Uncertain.", 9.3, 13)
    pdf.page(c.out())

    # Adapters
    c = Canvas(3, "Adapters and Contracts")
    c.para(M, 735, 505, "Each LoRA adapter has a narrow role and a JSON contract. This keeps the system auditable and reduces hallucinated outputs.", 9.5, 13)
    table(c, M, 680, [132, 145, 80, 148], ["Adapter", "Role", "Rows", "Contract"], [
        ["numeric_parser_final", "Turns natural Type 2 wording into structured quantities and formulas.", "1443 / 220", "No answer, no Python, no final_result, no CoT."],
        ["trace_explainer_final", "Explains a locked calculator trace in readable language.", "1196 / 163", "Explanation only; answer, unit and formula are immutable."],
        ["chlt_reasoner_final", "Conceptual Type 1 reasoner. Pure CHLT subset is Yes / No / Uncertain.", "334 / 73", "Used after quantitative claim verification fails or when the prompt is purely conceptual."],
    ], row_h=60)
    c.text(M, 390, "Why no planner-code adapter", 14, "F2")
    y = 365
    for item in [
        "Generated code can be syntactically invalid or use the wrong equation.",
        "Generated answer fields can disagree with the code result.",
        "The final design asks the model to parse only, then lets deterministic formulas calculate.",
    ]:
        y = c.bullet(M, y, item, 500) - 3
    c.text(M, 245, "Numeric parser allowed output", 12, "F2")
    c.para(M, 222, 235, "topic, question_kind, target, givens, constraints, formula_candidates, confidence", 8.8, 12)
    c.text(M + 270, 245, "Numeric parser forbidden output", 12, "F2")
    c.para(M + 270, 222, 235, "answer, final_result, unit_answer, python_code, golden_code, chain-of-thought", 8.8, 12)
    pdf.page(c.out())

    # Verification
    c = Canvas(4, "Calculator and Verification")
    c.card(M, 665, 158, 78, "69 formulas", "Registered branches in formula_bank.py and calculator.py.", (0.06, 0.34, 0.54))
    c.card(M + 174, 665, 158, 78, "Payload validator", "Checks JSON, spans, roles and formula IDs before calculation.", (0.17, 0.48, 0.34))
    c.card(M + 348, 665, 158, 78, "Locked result", "Final numeric answer is computed by deterministic code.", (0.78, 0.42, 0.14))
    c.text(M, 610, "Covered formula families", 14, "F2")
    y = 585
    for name, desc in [
        ("Circuits", "Ohm law, power, wire resistance, series/parallel networks."),
        ("Capacitors", "Charge, voltage, capacitance, stored energy and equivalent capacitance."),
        ("LC / RLC", "LC energy and frequency, resonance, reactance and impedance."),
        ("Electrostatics", "Coulomb force, point-charge fields and vector resultants."),
        ("Induction", "Magnetic flux, Faraday emf, solenoid field and inductor energy."),
        ("Measurement", "Relative error, absolute error and propagated uncertainty."),
    ]:
        c.text(M, y, name, 9.4, "F2", (0.05, 0.25, 0.40))
        c.para(M + 110, y, 390, desc, 8.5, 11)
        y -= 39
    c.text(M, 315, "Validation ladder", 14, "F2")
    y = 290
    for item in [
        "1. Parse JSON and required fields.",
        "2. Reject forbidden generated fields in numeric parser output.",
        "3. Check raw spans against the question.",
        "4. Check formula candidates against the registered formula bank.",
        "5. Compute and format the final answer with deterministic unit conversion.",
    ]:
        y = c.bullet(M, y, item, 500) - 3
    pdf.page(c.out())

    # Deployment
    c = Canvas(5, "Deployment and API Contract")
    table(c, M, 725, [135, 370], ["Item", "Final setting"], [
        ["Endpoint", "/predict returns one JSON result per query."],
        ["Kaggle package", "kaggle_api_package_hybrid_v2_clean.zip"],
        ["Notebook", "result/Notebook/deploy_physics_api_qwen3b_vllm_kaggle.ipynb"],
        ["Runtime default", "Numeric parser and conceptual reasoner enabled. Polish disabled by default for speed."],
        ["Health checks", "/health and /v1/models expose pipeline and model status."],
    ], row_h=45)
    c.text(M, 430, "Response fields", 14, "F2")
    y = 405
    for item in [
        "query_id: copied from the request.",
        "answer and unit: final formatted result.",
        "explanation: deterministic or locked-trace explanation.",
        "reasoning: concise steps/evidence for auditability.",
        "debug: optional timing, route, confidence and model status.",
    ]:
        y = c.bullet(M, y, item, 500) - 3
    c.rect(M, 105, 505, 72, fill=(0.94, 0.98, 0.95), stroke=(0.72, 0.84, 0.74), sw=0.8)
    c.text(M + 12, 150, "Final design claim", 11, "F2", (0.12, 0.36, 0.20))
    c.para(M + 12, 130, 480, "The system combines V1 deterministic speed with V2 parser-plus-calculator flexibility. The model helps understand and explain; the calculator owns numeric truth.", 9, 12, color=(0.18, 0.28, 0.22))
    pdf.page(c.out())

    pdf.save(PDF_PATH)
    print(PDF_PATH)
    print(PDF_PATH.stat().st_size)


if __name__ == "__main__":
    build()
