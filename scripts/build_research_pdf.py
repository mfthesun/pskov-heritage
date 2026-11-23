"""Собрать самостоятельную версию отчёта с картами и ссылками."""
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
LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
IMAGE = re.compile(r"!\[[^\]]+\]\(([^)]+)\)")
plt.rcParams["pdf.fonttype"] = 42


def fit_lines(content, fig):
    width = .87 * fig.bbox.width
    font = FontProperties(size=11, family=TITLE_FONT)
    renderer = fig.canvas.get_renderer()
    line = ""
    for word in content.split():
        candidate = f"{line} {word}".strip()
        if line and renderer.get_text_width_height_descent(candidate, font, False)[0] > width:
            yield line
            line = word
        else:
            line = candidate
    if line:
        yield line


def build_pdf(source=SOURCE, output=OUTPUT):
    sections = re.split(r"(?m)^## ", source.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output, metadata={"Title": "Историко-культурное наследие Пскова"}) as pdf:
        for index, section in enumerate(sections, start=1):
            title, body = section.strip().split("\n", 1)
            fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
            fig.text(.065, .956, "ПСКОВ / МИКРОУРБАНИСТИКА", fontsize=8, color="#6d5963", weight="bold")
            if not body.lstrip().startswith("!["):
                fig.text(.065, .919, title.removeprefix("# "), fontsize=23, fontfamily=TITLE_FONT,
                         weight="bold", va="top")
            y = .861
            for paragraph in body.strip().split("\n\n"):
                image = IMAGE.fullmatch(paragraph.strip())
                if image:
                    ax = fig.add_axes((.035, .45, .93, .465))
                    ax.imshow(plt.imread(source.parent / image[1]))
                    ax.set_axis_off()
                    y = .422
                    continue
                links = LINK.findall(paragraph)
                content = LINK.sub("", paragraph).replace(" · ", " ").replace("**", "").strip(" .·\n")
                if content:
                    if content[-1] not in ".?!":
                        content += "."
                    for line in fit_lines(content, fig):
                        fig.text(.065, y, line, fontsize=11, fontfamily=TITLE_FONT, va="top", color="#202127")
                        y -= .018
                    y -= .012
                for label, url in links:
                    fig.text(.065, y, label, fontsize=8.5, va="top", url=url, color="#31566e")
                    y -= .016
                if links:
                    y -= .016
            if y < .075:
                raise ValueError(f"Страница {index} выходит за поля отчёта")
            fig.text(.065, .05, "Историко-культурное наследие Пскова", fontsize=8, color="#53545b")
            fig.text(.935, .05, f"{index:02d} / {len(sections):02d}", ha="right", fontsize=8, color="#53545b")
            pdf.savefig(fig)
            plt.close(fig)
    return len(sections)


if __name__ == "__main__":
    print(build_pdf())
