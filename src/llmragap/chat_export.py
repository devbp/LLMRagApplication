"""Export the chat history as a PDF (used by the "Export chat as PDF" button in chat_gui.py).

Answers are Markdown with Streamlit extras (:red[...], :material/icon:). The PDF shows them as
plain text: headings and **bold** spans become bold lines/text without the markers, and list
items get a bullet.
"""

import os
import re
from datetime import datetime

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from llmragap.session_cost import money

# A Unicode TrueType font is needed for characters like §, ä, ß, – and “”.
# The first pair (regular, bold) that exists is used; otherwise Helvetica (Latin-1 only).
_WINDOWS_FONTS = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
FONT_CANDIDATES = [
    (os.path.join(_WINDOWS_FONTS, "DejaVuSans.ttf"), os.path.join(_WINDOWS_FONTS, "DejaVuSans-Bold.ttf")),
    (os.path.join(_WINDOWS_FONTS, "arial.ttf"), os.path.join(_WINDOWS_FONTS, "arialbd.ttf")),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
]

ROLE_STYLE = {  # label and RGB color of each message's header
    "user": ("You", (25, 90, 190)),
    "assistant": ("Assistant", (30, 130, 70)),
}
GRAY = (110, 110, 110)
BLACK = (0, 0, 0)


def clean_markdown_line(line: str) -> tuple[str, bool]:
    """Return (plain text, is_heading) for one Markdown line."""
    line = re.sub(r":material/[\w_]+:", "", line)  # Streamlit icons
    line = re.sub(r":(?:red|orange|yellow|green|blue|violet|gray|grey|rainbow|primary)\[(.*?)\]", r"\1", line)
    line = re.sub(r"(\*\*|__)(.+?)\1", r"\2", line)  # bold markers
    line = line.replace("`", "")  # inline code
    heading = re.match(r"^\s*#{1,6}\s+(.*)", line)
    if heading:
        return heading.group(1), True
    bullet = re.match(r"^(\s*)[-*+]\s+(.*)", line)
    if bullet:
        return f"{bullet.group(1)}•  {bullet.group(2)}", False
    return line, False


class ChatPdf(FPDF):
    """A4 PDF with a Unicode font (when one is found) and page numbers in the footer."""

    def __init__(self, title: str):
        super().__init__(format="A4")
        self.doc_title = title
        self.font_name = "Helvetica"
        for regular, bold in FONT_CANDIDATES:
            if os.path.exists(regular) and os.path.exists(bold):
                self.add_font("ChatFont", "", regular)
                self.add_font("ChatFont", "B", bold)
                self.font_name = "ChatFont"
                break
        self.set_title(title)
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(18, 18, 18)

    def text_for_font(self, text: str) -> str:
        """Helvetica only knows Latin-1, so replace other characters with '?' when it is used."""
        if self.font_name == "Helvetica":
            return text.replace("•", "-").encode("latin-1", "replace").decode("latin-1")
        return text

    def style(self, size: float, bold: bool = False, color: tuple[int, int, int] = BLACK) -> None:
        self.set_font(self.font_name, "B" if bold else "", size)
        self.set_text_color(*color)

    def write_line(self, text: str, height: float = 5.5) -> None:
        self.multi_cell(0, height, self.text_for_font(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def footer(self) -> None:
        self.set_y(-12)
        self.style(8, color=GRAY)
        self.cell(0, 6, self.text_for_font(f"{self.doc_title} · page {self.page_no()}/{{nb}}"), align="C")


def build_chat_pdf(messages: list[dict], title: str) -> bytes:
    """Render chat messages ({"role", "content", "steps"?, "cost"?} dicts) as PDF bytes."""
    pdf = ChatPdf(title)
    pdf.add_page()

    pdf.style(16, bold=True)
    pdf.write_line(f"{title}: chat history", height=8)
    questions = sum(1 for m in messages if m["role"] == "user")
    pdf.style(9, color=GRAY)
    pdf.write_line(f"Exported {datetime.now():%Y-%m-%d %H:%M} · {questions} question(s)")
    pdf.ln(4)

    for msg in messages:
        label, color = ROLE_STYLE.get(msg["role"], (msg["role"].title(), BLACK))
        pdf.style(11, bold=True, color=color)
        pdf.write_line(label, height=7)

        for raw_line in msg["content"].splitlines() or [""]:
            text, is_heading = clean_markdown_line(raw_line)
            pdf.style(11 if is_heading else 10, bold=is_heading)
            pdf.write_line(text)

        if msg["role"] == "assistant":
            steps = msg.get("steps") or []
            if steps:
                pdf.ln(1)
                pdf.style(8.5, bold=True, color=GRAY)
                pdf.write_line("Tools used", height=4.5)
                pdf.style(8.5, color=GRAY)
                for step in steps:
                    args = step.get("input") or {}
                    query = args.get("query") if isinstance(args, dict) else None
                    pdf.write_line(f"•  {step['tool']}: {query if query else args}", height=4.5)
            cost = msg.get("cost")
            if cost:
                total = cost["cost_usd"] + cost.get("search_embedding_usd", 0.0)
                pdf.style(8.5, color=GRAY)
                pdf.write_line(
                    f"Cost: {money(total)} · {cost['input_tokens']:,} in / {cost['output_tokens']:,} out tokens"
                    f" · {cost['calls']} model call(s)",
                    height=4.5,
                )

        # thin divider between messages
        pdf.ln(2)
        pdf.set_draw_color(220, 220, 220)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(3)

    return bytes(pdf.output())
