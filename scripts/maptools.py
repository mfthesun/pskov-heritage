"""Локальный макет карт Пскова: EPSG:32635, поле слева, легенда справа."""
import math
import textwrap

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle
from matplotlib.transforms import Bbox
from pyproj import Transformer

FIG_SIZE = (10, 7)
DPI = 200
TARGET_CRS = "EPSG:32635"
MAP_RECT = (0.035, 0.13, 0.65, 0.73)
LEGEND_ANCHOR = (0.72, 0.86)
NORTH_LOC = (0.94, 0.86)
SCALE_LOC = (0.04, 0.04)
SCALE_FRAC = 0.25
NICE_DISTANCES_M = (
    1, 2, 5, 10, 20, 50, 100, 200, 500,
    1_000, 2_000, 5_000, 10_000, 20_000, 50_000,
    100_000, 200_000, 500_000,
)
INK, MUTED, SURFACE = "#171717", "#5b5b5b", "#ffffff"
HALO = [pe.withStroke(linewidth=2.5, foreground=SURFACE)]

TEXT_FONT = ["Liberation Sans", "DejaVu Sans"]
TITLE_FONT = ["Liberation Serif", "DejaVu Serif"]
plt.rcParams["font.family"] = TEXT_FONT


def _wrap(text, width):
    return "\n".join(
        textwrap.fill(line, width, break_long_words=False, break_on_hyphens=False)
        for line in text.split("\n")
    )


def map_figure(title="", subtitle="", sources="", attribution=""):
    """Создать холст 2000×1400 с оформлением вне поля карты."""
    fig = plt.figure(figsize=FIG_SIZE, facecolor=SURFACE)
    ax = fig.add_axes(MAP_RECT, facecolor=SURFACE)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    if title:
        fig.text(0.035, 0.955, _wrap(title, 75), ha="left", va="top", fontsize=19, fontfamily=TITLE_FONT, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.035, 0.91, _wrap(subtitle, 105), ha="left", va="top", fontsize=9.5, color=MUTED)
    if sources:
        fig.text(0.035, 0.075, _wrap(f"Источники: {sources}", 126), ha="left", va="top", fontsize=7.5, color=MUTED)
    if attribution:
        fig.text(0.035, 0.032, _wrap(f"Атрибуция подложки: {attribution}", 126), ha="left", va="top", fontsize=7.5, color=MUTED)
    return fig, ax


def set_extent(ax, x0, y0, x1, y1):
    """Вписать охват в поле без различия масштаба по осям; координаты — метры."""
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Охват должен иметь положительную ширину и высоту")
    fw, fh = ax.figure.get_size_inches()
    box = ax.get_position()
    ratio = (box.height * fh) / (box.width * fw)
    width, height = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if height / width < ratio:
        height = width * ratio
    else:
        width = height / ratio
    ax.set_xlim(cx - width / 2, cx + width / 2)
    ax.set_ylim(cy - height / 2, cy + height / 2)


def choose_scale_distance(width_m, frac=SCALE_FRAC):
    """Выбрать наибольшую круглую длину линейки, не превышающую frac ширины."""
    if width_m <= 0 or not 0 < frac <= 1:
        raise ValueError("Ширина должна быть положительной, frac — в интервале (0, 1]")
    return max(distance for distance in NICE_DISTANCES_M if distance <= frac * width_m)


def _scale_label(distance_m, unit):
    value = distance_m / 1000 if unit == "км" else distance_m
    return f"{value:g}".replace(".", ",")


def scale_bar(ax, loc=SCALE_LOC, frac=SCALE_FRAC, segments=4, fontsize=7.5):
    """Нарисовать линейку в метрах проекции; вернуть её полную длину в метрах."""
    if ax.get_aspect() != 1.0:
        raise ValueError("Для масштабной линейки нужен одинаковый масштаб по осям")
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    width, height = x1 - x0, y1 - y0
    distance = choose_scale_distance(width, frac)
    segment = distance / segments
    bx, by = x0 + loc[0] * width, y0 + loc[1] * height
    bar_height = 0.011 * height
    elements = []
    for index in range(segments):
        elements.append(ax.add_patch(Rectangle(
            (bx + index * segment, by), segment, bar_height,
            facecolor=INK if index % 2 == 0 else SURFACE,
            edgecolor=INK, linewidth=0.6, zorder=20,
        )))
    unit = "км" if distance >= 1000 else "м"
    for index, label in (
        (0, "0"),
        (segments // 2, _scale_label(distance / 2, unit)),
        (segments, f"{_scale_label(distance, unit)} {unit}"),
    ):
        elements.append(ax.text(
            bx + index * segment, by + bar_height * 1.8, label,
            ha="center", va="bottom", fontsize=fontsize, color=INK,
            path_effects=HALO, zorder=21,
        ))
    ax.apply_aspect()
    renderer = ax.figure.canvas.get_renderer()
    pad = 3.5 * ax.figure.dpi / 72
    frame = Bbox.union([part.get_window_extent(renderer) for part in elements]).padded(pad)
    (left, bottom), (right, top) = ax.transData.inverted().transform(frame.get_points())
    ax.add_patch(Rectangle(
        (left, bottom), right - left, top - bottom,
        facecolor=SURFACE, edgecolor="none", zorder=19,
    ))
    return distance


def true_north_angle(x, y, crs=TARGET_CRS):
    """Угол истинного севера от вертикали карты в радианах, положительный вправо."""
    lon, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(x, y)
    nx, ny = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat + 0.01)
    return math.atan2(nx - x, ny - y)


def north_arrow(ax, crs=TARGET_CRS, loc=NORTH_LOC, size=0.075, fontsize=10):
    """Нарисовать стрелку истинного севера с учётом сближения меридианов."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    angle = true_north_angle(cx, cy, crs) if crs is not None else 0
    width, height = x1 - x0, y1 - y0
    length = size * height
    bx, by = x0 + loc[0] * width, y0 + loc[1] * height
    dx, dy = length * math.sin(angle), length * math.cos(angle)
    ax.add_patch(FancyArrow(
        bx, by, dx, dy, width=length * 0.1, head_width=length * 0.38,
        head_length=length * 0.42, length_includes_head=True,
        color=INK, path_effects=HALO, zorder=20,
    ))
    ax.text(
        bx + dx * 1.2, by + dy * 1.2, "С", ha="center", va="bottom",
        fontsize=fontsize, fontweight="bold", color=INK,
        path_effects=HALO, zorder=21,
    )


def finish(fig, ax, extent, handles, legend_title="Условные обозначения", crs=TARGET_CRS):
    """Применить охват и добавить рамку, север, линейку и русскую легенду."""
    set_extent(ax, *extent)
    ax.add_patch(Rectangle(
        (0, 0), 1, 1, transform=ax.transAxes, fill=False,
        edgecolor=MUTED, linewidth=0.8, zorder=30,
    ))
    north_arrow(ax, crs)
    scale_bar(ax)
    for handle in handles:
        handle.set_label(_wrap(handle.get_label(), 32))
    return fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=LEGEND_ANCHOR,
        frameon=True, facecolor=SURFACE, edgecolor="#d6d6d6", framealpha=1,
        fontsize=9, title=_wrap(legend_title, 32), title_fontsize=9.5,
        alignment="left", labelspacing=0.7, handlelength=2.2,
    )


def save_map(fig, path):
    """Сохранить PNG без обрезки и автоматического изменения размера."""
    fig.canvas.draw()
    for legend in fig.legends:
        bounds = legend.get_window_extent(fig.canvas.get_renderer())
        if not fig.bbox.contains(bounds.x0, bounds.y0) or not fig.bbox.contains(bounds.x1, bounds.y1):
            raise ValueError(f"Легенда выходит за холст: {path}")
    fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
