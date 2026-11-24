"""Собрать самостоятельную версию отчёта с картами и ссылками.

Поточная вёрстка: разделы и абзацы не привязаны к одной странице,
текст переносится по измеренной ширине, страницы нумеруются после
завершения вёрстки, когда общее число страниц уже известно.
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties

from .maptools import TITLE_FONT

SOURCE = Path("reports/standalone.md")
OUTPUT = Path("output/pdf/Наследие_Пскова.pdf")
TEXT_FONT = ["Liberation Sans", "DejaVu Sans"]
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
PAGE_SIZE = (8.27, 11.69)
plt.rcParams["pdf.fonttype"] = 42

X0, X1 = 0.065, 0.935
TOP, BOTTOM = 0.895, 0.075
TEXT_SIZE, TEXT_LEAD = 11, 0.007
LINK_SIZE, LINK_LEAD = 8.5, 0.005
HEAD_SIZE, HEAD_LEAD = 23, 0.03
PARAGRAPH_GAP = 0.014
IMAGE_GAP = 0.02

_PROBE = plt.figure(figsize=PAGE_SIZE)
_RENDERER = _PROBE.canvas.get_renderer()


def measure(text, size, family):
    """Доля ширины и высоты страницы, занимаемая строкой."""
    font = FontProperties(size=size, family=family)
    width, height, descent = _RENDERER.get_text_width_height_descent(text, font, False)
    return width / _PROBE.bbox.width, (height + descent) / _PROBE.bbox.height


def wrap(content, size, family, width=X1 - X0):
    line = ""
    for word in content.split():
        candidate = f"{line} {word}".strip()
        if line and measure(candidate, size, family)[0] > width:
            yield line
            line = word
        else:
            line = candidate
    if line:
        yield line


class Flow:
    """Накапливает draw-команды по страницам, начиная новую при нехватке места."""

    def __init__(self):
        self.pages = []
        self.items = []
        self.section = ""
        self.y = TOP

    def _flush(self):
        self.pages.append({"section": self.section, "items": self.items})
        self.items = []
        self.y = TOP

    def _ensure(self, height):
        if self.y - height < BOTTOM:
            self._flush()

    def heading(self, text):
        self._ensure(HEAD_LEAD + measure(text, HEAD_SIZE, TITLE_FONT)[1])
        self.items.append({
            "kind": "text", "x": X0, "y": self.y, "text": text,
            "size": HEAD_SIZE, "family": TITLE_FONT, "bold": True, "color": "#202127",
        })
        self.y -= HEAD_LEAD + measure(text, HEAD_SIZE, TITLE_FONT)[1]

    def paragraph(self, content, links):
        lines = list(wrap(content, TEXT_SIZE, TITLE_FONT))
        # резерв под подписи ссылок: последняя строка не должна оставаться
        # на странице одна, если подписи переходят на новую
        reserve = PARAGRAPH_GAP + sum(
            measure(label, LINK_SIZE, TEXT_FONT)[1] + LINK_LEAD for label, _ in links
        )
        for index, line in enumerate(lines):
            _, line_height = measure(line, TEXT_SIZE, TITLE_FONT)
            tail = reserve if index == len(lines) - 1 else 0.0
            self._ensure(line_height + TEXT_LEAD + tail)
            self.items.append({
                "kind": "text", "x": X0, "y": self.y, "text": line,
                "size": TEXT_SIZE, "family": TITLE_FONT, "bold": False, "color": "#202127",
            })
            self.y -= line_height + TEXT_LEAD
        self.y -= PARAGRAPH_GAP
        for label, url in links:
            _, line_height = measure(label, LINK_SIZE, TEXT_FONT)
            self._ensure(line_height + LINK_LEAD)
            self.items.append({
                "kind": "text", "x": X0, "y": self.y, "text": label,
                "size": LINK_SIZE, "family": TEXT_FONT, "bold": False,
                "color": "#31566e", "url": url,
            })
            self.y -= line_height + LINK_LEAD
        if links:
            self.y -= PARAGRAPH_GAP

    def image(self, path, width=X1 - X0):
        image_height, image_width = plt.imread(path).shape[:2]
        height = width * image_height / image_width
        self._ensure(height + IMAGE_GAP)
        self.items.append({
            "kind": "image", "x": X0, "y": self.y, "w": width, "h": height, "path": path,
        })
        self.y -= height + IMAGE_GAP

    def finish(self):
        if self.items or not self.pages:
            self._flush()
        return self.pages


def parse_sections(source):
    sections = re.split(r"(?m)^## ", source.read_text(encoding="utf-8"))
    blocks = []
    for section in sections:
        title, _, body = section.strip().partition("\n")
        for paragraph in body.strip().split("\n\n"):
            paragraph = paragraph.strip()
            if paragraph:
                blocks.append((title.removeprefix("# "), paragraph))
    return blocks


def clean_paragraph(paragraph):
    """Отделить повествовательный текст от подписей ссылок.

    Внешние URL возвращаются с адресом (кликабельные подписи),
    относительные — без адреса. Остатки пунктуации вокруг удалённых
    ссылок нормализуются: « , .» не допускается.
    """
    links = [
        (label, url if url.startswith(("http://", "https://")) else None)
        for label, url in LINK.findall(paragraph)
    ]
    content = LINK.sub("", paragraph)
    content = content.replace(" · ", " ").replace("**", "")
    content = re.sub(r"\s+([,.;:!?])", r"\1", content)
    content = re.sub(r"[,.;!?](\s*[,.;!?])+", lambda match: match.group(0)[-1], content)
    content = content.strip(" .·\n")
    if content and content[-1] not in ".?!":
        content += "."
    return content, links


def build_pdf(source=SOURCE, output=OUTPUT):
    source = Path(source)
    flow = Flow()
    last_title = None
    for title, paragraph in parse_sections(source):
        if title != last_title:
            flow.section = title
            last_title = title
            flow.heading(title)
        image = IMAGE.fullmatch(paragraph)
        if image:
            flow.image(source.parent / image[1])
            continue
        content, links = clean_paragraph(paragraph)
        if content:
            flow.paragraph(content, links)
        elif links:
            flow.paragraph("", links)
    pages = flow.finish()

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    total = len(pages)
    with PdfPages(output, metadata={"Title": "Историко-культурное наследие Пскова"}) as pdf:
        for number, page in enumerate(pages, start=1):
            fig = plt.figure(figsize=PAGE_SIZE, facecolor="white")
            fig.text(X0, 0.965, "ПСКОВ / МИКРОУРБАНИСТИКА", fontsize=8,
                     color="#6d5963", weight="bold", family=TEXT_FONT)
            starts_section = page["items"] and page["items"][0].get("bold")
            if not starts_section:
                fig.text(X0, 0.935, page["section"], fontsize=9, color="#6d5963",
                         style="italic", family=TITLE_FONT, va="top")
            for item in page["items"]:
                if item["kind"] == "image":
                    ax = fig.add_axes((item["x"], item["y"] - item["h"], item["w"], item["h"]))
                    ax.imshow(plt.imread(item["path"]))
                    ax.set_axis_off()
                else:
                    fig.text(item["x"], item["y"], item["text"], fontsize=item["size"],
                             fontfamily=item["family"], va="top", color=item["color"],
                             weight="bold" if item.get("bold") else "normal",
                             url=item.get("url"))
            fig.text(X0, 0.05, "Историко-культурное наследие Пскова",
                     fontsize=8, color="#53545b", family=TEXT_FONT)
            fig.text(X1, 0.05, f"{number:02d} / {total:02d}", ha="right",
                     fontsize=8, color="#53545b", family=TEXT_FONT)
            pdf.savefig(fig)
            plt.close(fig)
    plt.close(_PROBE)
    return total


if __name__ == "__main__":
    print(build_pdf())
