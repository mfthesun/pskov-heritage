from pathlib import Path

from scripts.build_research_pdf import (
    BOTTOM,
    LINK_LEAD,
    LINK_SIZE,
    PARAGRAPH_GAP,
    TEXT_FONT,
    TEXT_LEAD,
    TEXT_SIZE,
    TITLE_FONT,
    Flow,
    build_pdf,
    clean_paragraph,
    measure,
)


def test_last_text_line_moves_together_with_links():
    flow = Flow()
    line = "последняя строка абзаца"
    label = "подпись ссылки"
    line_h = measure(line, TEXT_SIZE, TITLE_FONT)[1] + TEXT_LEAD
    link_h = measure(label, LINK_SIZE, TEXT_FONT)[1] + LINK_LEAD
    # остаток страницы: строка помещается, а строка + отступ + подпись — нет
    flow.y = BOTTOM + line_h + PARAGRAPH_GAP + link_h * 0.3
    flow.paragraph(line, [(label, "https://example.com")])
    pages = flow.finish()
    assert len(pages) == 2
    first = pages[1]["items"][0]
    assert first["size"] == TEXT_SIZE and first["text"] == line
    assert pages[1]["items"][-1]["url"] == "https://example.com"


def test_external_links_leave_narrative_and_stay_clickable():
    content, links = clean_paragraph(
        "Выводы подтверждены [карточка кремля](https://museumpskov.ru/locations) "
        "и [таблица ЮНЕСКО](https://whc.unesco.org/en/list/1523/maps/)."
    )
    assert "[" not in content and "http" not in content
    assert content.endswith(".")
    assert links == [("карточка кремля", "https://museumpskov.ru/locations"),
                     ("таблица ЮНЕСКО", "https://whc.unesco.org/en/list/1523/maps/")]


def test_relative_links_leave_narrative_without_url():
    content, links = clean_paragraph(
        "Пространственные сопоставления проверены [ГИС-сверка](../docs/gis_verification.md)."
    )
    assert "[" not in content and "../docs" not in content
    assert content.endswith(".")
    assert links == [("ГИС-сверка", None)]


def test_punctuation_after_link_removal_is_clean():
    content, _ = clean_paragraph(
        "Сопоставление методически продуктивным [карточка кремля](https://museumpskov.ru/locations), "
        "[карточка палат](https://museumpskov.ru/locations?slug=pogankiny_palaty)."
    )
    assert content == "Сопоставление методически продуктивным."
    assert ",." not in content and " ," not in content

_SECTIONS = []
for index in range(1, 11):
    paragraphs = "\n\n".join(
        f"Абзац {paragraph} содержит достаточно длинный текст, чтобы занять "
        f"несколько строк на странице отчёта при типовом кегле основного текста. "
        f"Нумерация страниц и переносы проверяются автоматически [ссылка {paragraph}](https://example.com/{index}/{paragraph})."
        for paragraph in range(1, 7)
    )
    _SECTIONS.append(f"## Раздел {index}\n\n{paragraphs}")
LONG_MD = "# Синтетический длинный отчёт\n\n" + "\n\n".join(_SECTIONS)


def test_long_synthetic_text_paginates(tmp_path):
    source = tmp_path / "long.md"
    source.write_text(LONG_MD, encoding="utf-8")
    output = tmp_path / "long.pdf"
    pages = build_pdf(source=source, output=output)
    assert pages >= 6
    assert output.stat().st_size > 0


def test_real_standalone_builds(tmp_path):
    output = tmp_path / "standalone.pdf"
    pages = build_pdf(output=output)
    assert pages >= 8
    assert output.stat().st_size > 100_000
