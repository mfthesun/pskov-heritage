"""Схема границ Покровского угла по координатным таблицам МСК-60."""
import csv
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as PlotPath
from shapely.geometry import Polygon
from shapely.ops import orient

from .maptools import HALO, finish, map_figure, save_map
from .plot_heritage import ACTION, OSM, draw_basemap, label, read_basemap, symbol
from .prepare_heritage import PSKOKN

SOURCE = Path("data/raw/pokrova_boundaries_msk60.csv")
OUTPUT = Path("maps/04_охранные_границы.png")
SITES = Path("maps/03_точки_фломастер_плана.geojson")
ASSESSMENT = Path("data/processed/pokrova_site_screening.csv")
# Рабочая привязка OSM для обзорной схемы. Параметры см. в журнале проверки.
MSK60 = (
    "+proj=tmerc +lat_0=0 +lon_0=27.95 +k=1 +x_0=1250000 +y_0=-5911057.63 "
    "+ellps=krass +towgs84=23.57,-140.95,-79.8,0,0.35,0.79,-0.22 +units=m +no_defs"
)
OCHRE, TEAL, RED = "#d7854d", "#92bbb1", "#8c245c"
LABELS = {
    "EOZ-1": "ЕОЗ · 1", "EOZ-2": "ЕОЗ · 2",
    "EZRZ-1": "ЕЗРЗ-1", "EZRZ-2": "ЕЗРЗ-2", "EZRZ-3": "ЕЗРЗ-3",
    "EZRZ-4": "ЕЗРЗ-4", "EZRZ-5": "ЕЗРЗ-5", "TERR": "Территория\nцеркви",
}


def read_boundaries(path=SOURCE):
    rings = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            ring = rings[row["area"], row["ring"]]
            if len(ring) + 1 != int(row["vertex"]):
                raise ValueError("Нарушена последовательность вершин")
            ring.append((float(row["y_east_m"]), float(row["x_north_m"])))
    if set(rings) != {(key, "shell") for key in LABELS} | {("EZRZ-3", "hole"), ("EZRZ-5", "hole")}:
        raise ValueError("Неверный набор контуров")
    if any(ring[0] != ring[-1] for ring in rings.values()):
        raise ValueError("Незамкнутый контур")
    return {
        key: Polygon(rings[key, "shell"], [rings[key, "hole"]] if (key, "hole") in rings else [])
        for key in LABELS
    }


def check_context(blocks, roads):
    """Сравнить положение ориентиров OSM с их положением на схеме приложения."""
    vertices = pd.read_csv(SOURCE).query("area != 'TERR'").drop_duplicates("point").set_index("point")
    pixels = {48: (658, 32), 101: (876, 990), 102: (751, 1060), 109: (347, 88),
              108: (383, 258), 103: (663, 895), 104: (532, 654)}
    design = np.array([(*xy, 1) for xy in pixels.values()])
    target = vertices.loc[list(pixels), ["y_east_m", "x_north_m"]].to_numpy()
    affine = np.linalg.lstsq(design, target, rcond=None)[0]
    landmarks = [
        ("церковь", (532, 398), blocks.loc[blocks["name"].eq("Церковь Покрова и Рождества от Пролома")]),
        ("башня", (501, 473), blocks.loc[blocks["name"].eq("Покровская башня")]),
        ("мост", (418, 684), roads.loc[roads["osm_id"].eq("304638706")]),
    ]
    distances = {}
    for name, pixel, selected in landmarks:
        point = selected.geometry.union_all().centroid
        distances[name] = float(np.linalg.norm(np.array([point.x, point.y]) - np.array((*pixel, 1)) @ affine))
    if not all(distance < 15 for distance in distances.values()):
        raise ValueError(f"Подложка расходится с ориентирами приложения: {distances}")
    return distances


def draw_boundaries(output=OUTPUT, pskokn=PSKOKN, scenario_path=SITES, assessment=ASSESSMENT):
    areas = read_boundaries()
    archive = pskokn / "данные и материалы" / "pskov_oblast.pbf"
    bounds = (28.322, 57.799, 28.345, 57.811)
    blocks = read_basemap(
        archive, "multipolygons", bounds,
        "building IS NOT NULL OR leisure IN ('park','garden') OR landuse IN ('forest','grass')",
    ).to_crs(MSK60)
    roads = read_basemap(
        archive, "lines", bounds,
        "highway IS NOT NULL AND highway NOT IN ('construction','proposed','platform')",
    ).to_crs(MSK60)
    water = read_basemap(archive, "multipolygons", bounds, "natural = 'water'").to_crs(MSK60)
    distances = check_context(blocks, roads)
    sites = gpd.read_file(scenario_path).to_crs(MSK60)
    rows = []
    for row in sites.itertuples():
        covered = [name for name, shape in areas.items() if shape.is_valid and shape.covers(row.geometry)]
        nearest = min(areas, key=lambda name: areas[name].boundary.distance(row.geometry))
        rows.append({
            "scenario_id": row.scenario_id,
            "modelled_areas": "|".join(covered),
            "nearest_boundary_area": nearest,
            "distance_to_nearest_boundary_m": round(areas[nearest].boundary.distance(row.geometry)),
            "status": "Обзорная привязка; режим участка не установлен",
        })
    pd.DataFrame(rows).to_csv(assessment, index=False)
    fig, ax = map_figure(
        "Охранные границы Покровского угла",
        "Границы в МСК-60 и ориентировочная привязка подложки",
        "Изменение от 14.04.2022 к № 879 и приказ Минкультуры № 1542 (2024). Север координатной сетки.",
        OSM,
    )
    draw_basemap(ax, blocks, roads, water)
    for key, polygon in areas.items():
        if key == "EZRZ-3":
            for ring in [polygon.exterior, *polygon.interiors]:
                ax.plot(*ring.xy, "--", color="#405b74", linewidth=1.5, zorder=6)
            ax.text(1273030, 498871, LABELS[key], ha="center", fontsize=8, path_effects=HALO, zorder=8)
            continue
        polygon = orient(polygon)
        shape = PlotPath.make_compound_path(*[PlotPath(ring.coords) for ring in [
            polygon.exterior, *polygon.interiors,
        ]])
        is_territory = key == "TERR"
        fill = OCHRE if key.startswith("EOZ") else TEAL
        ax.add_patch(PathPatch(
            shape, facecolor="none" if is_territory else (*to_rgb(fill), .66),
            edgecolor=RED if is_territory else "#526475", linewidth=1.5 if is_territory else .7,
            hatch="///" if is_territory else None, zorder=7 if is_territory else 5,
        ))
        point = polygon.representative_point()
        if key == "EZRZ-2":
            ax.annotate(LABELS[key], (point.x, point.y), xytext=(-14, -5), textcoords="offset points",
                        ha="right", va="center", fontsize=8,
                        arrowprops={"arrowstyle": "-", "color": "#526475"}, path_effects=HALO, zorder=9)
        else:
            ax.text(point.x, point.y, LABELS[key], ha="center", va="center", fontsize=8,
                    path_effects=HALO, zorder=9)
    sites.plot(ax=ax, marker="^", markersize=43, color=ACTION, edgecolor="white", zorder=10)
    for row in sites.itertuples():
        label(ax, row.geometry, row.scenario_id, (-8, 3), fontweight="bold")
    ax.text(1272730, 498712, "Великая", fontsize=9, path_effects=HALO, zorder=8)
    finish(fig, ax, (1272600, 498200, 1273380, 499005), [
        Patch(facecolor="none", edgecolor=RED, hatch="///", label="Территория церкви"),
        Patch(facecolor=OCHRE, label="Единая охранная зона"),
        Patch(facecolor=TEAL, label="Зона регулирования застройки"),
        Line2D([], [], color="#405b74", linestyle="--", label="Контур ЕЗРЗ-3 по таблице"),
        symbol("Точка обследования", ACTION, "^"),
    ], crs=None)
    fig.text(.735, .51, "В табличном контуре ЕЗРЗ-3\nесть самопересечение.", fontsize=8.5, va="top", linespacing=1.5)
    fig.text(.735, .40, "Положение S1 и S2\nотносительно зон требует\nкадастровой проверки.", fontsize=8.5, va="top", linespacing=1.5)
    save_map(fig, output)
    return distances


if __name__ == "__main__":
    draw_boundaries()
