"""Карты по проверенным объектам, локальной подложке OSM и сценарным точкам."""
from collections import defaultdict
from heapq import heappop, heappush
from math import dist
from pathlib import Path

import geopandas as gpd
import matplotlib
import pandas as pd
from shapely.geometry import LineString, Point, box

matplotlib.use("Agg")
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    from .maptools import HALO, TARGET_CRS, finish, map_figure, save_map
    from .prepare_heritage import PSKOKN
except ImportError:
    from maptools import HALO, TARGET_CRS, finish, map_figure, save_map
    from prepare_heritage import PSKOKN

WATER, ROAD, LAND = "#c5e1ee", "#b9b6b0", "#f5f2ec"
BUILDINGS, GREEN, RAIL = "#d9d4cb", "#dbe6d1", "#8c8885"
CHURCH, WALL, PALACE, STATION = "#663b83", "#924c30", "#ad6a16", "#317489"
ACTION, ACT = "#00775d", "#8c245c"
OSM = "© участники OpenStreetMap, ODbL. Локальный срез 17.06.2026, 23:00 UTC."
UNESCO = "https://whc.unesco.org/en/list/1523/maps/"
ZONE_ACT = "https://legalacts.ru/doc/prikaz-minkultury-rossii-ot-13082024-n-1542-ob-utverzhdenii/"
PLACES = {
    "27025689": ("Псков", (6, 0)),
    "297521319": ("Изборск", (-7, -9)),
    "706163973": ("Печоры", (-7, 8)),
    "248335491": ("Самолва", (7, 0)),
    "738074805": ("Пушкинские Горы", (7, 0)),
    "27072632": ("Великие Луки", (-8, -1)),
}
SOUTH_COMPARISON = "27072632"


def symbol(label, color, marker):
    return Line2D([], [], marker=marker, color=color, linestyle="none", label=label, markersize=7)


def label(ax, point, name, offset=(6, 6), **kwargs):
    ax.annotate(
        name, (point.x, point.y), xytext=offset, textcoords="offset points",
        ha="right" if offset[0] < 0 else "left", va="center", fontsize=8,
        color="#252422", path_effects=HALO, zorder=12, **kwargs,
    )


def read_basemap(archive, layer, bounds, where):
    raw = gpd.read_file(archive, layer=layer, bbox=bounds, where=where)
    usable = raw.geometry.notna() & ~raw.geometry.is_empty & raw.geometry.is_valid
    return raw.loc[usable].clip(box(*bounds)).to_crs(TARGET_CRS)


def draw_basemap(ax, blocks, roads, water):
    ax.patch.set_visible(True)
    ax.set_facecolor(LAND)
    green = blocks[blocks["building"].isna()]
    green.plot(ax=ax, color=GREEN, edgecolor="none", zorder=1)
    blocks[blocks["building"].notna()].plot(ax=ax, color=BUILDINGS, edgecolor="none", zorder=2)
    water.plot(ax=ax, color=WATER, edgecolor="#adcbd9", linewidth=.24, zorder=3)
    paths = roads["highway"].isin(["footway", "path", "steps", "pedestrian"])
    rail = roads["railway"].eq("rail")
    roads[~(paths | rail)].plot(ax=ax, color=ROAD, linewidth=.62, zorder=4)
    roads[paths].plot(ax=ax, color="#bab0a1", linewidth=.52, linestyle="--", zorder=4)
    if rail.any():
        roads[rail].plot(ax=ax, color=RAIL, linewidth=.65, linestyle="--", zorder=4)


def mapped_walk(streets, start, end):
    """Связать точки по общим вершинам OSM и измерить кратчайший путь в метрах."""
    paths = streets[streets["highway"].isin(["footway", "path", "steps", "pedestrian"])].explode()
    graph = defaultdict(list)
    for row in paths.loc[paths.geom_type.eq("LineString")].itertuples():
        vertices = list(row.geometry.coords)
        for left, right in zip(vertices, vertices[1:]):
            graph[left].append((right, row.osm_id))
            graph[right].append((left, row.osm_id))
    first, last = tuple(start.coords[0]), tuple(end.coords[0])
    queue = [(0, first)]
    distances = {first: 0}
    previous = {}
    while queue:
        length, node = heappop(queue)
        if length > distances[node]:
            continue
        if node == last:
            coords, ways = [last], []
            while node != first:
                node, way = previous[node]
                coords.append(node)
                ways.append(way)
            return LineString(coords[::-1]), list(dict.fromkeys(ways[::-1]))
        for next_node, way in graph[node]:
            next_length = length + dist(node, next_node)
            if next_length < distances.get(next_node, float("inf")):
                distances[next_node] = next_length
                previous[next_node] = node, way
                heappush(queue, (next_length, next_node))
    raise ValueError("В выбранной сети OSM отсутствует путь S1–S2")


def build_maps(pskokn=PSKOKN, processed=Path("data/processed"), maps=Path("maps")):
    maps.mkdir(parents=True, exist_ok=True)
    archive = pskokn / "данные и материалы" / "pskov_oblast.pbf"
    clip_city = (28.19, 57.76, 28.46, 57.88)
    places = gpd.read_file(
        archive, layer="points", where="osm_id IN (" + ",".join(f"'{key}'" for key in PLACES) + ")",
    ).set_index("osm_id").loc[list(PLACES)].to_crs(TARGET_CRS)
    water = read_basemap(archive, "multipolygons", clip_city, "natural = 'water'")
    assert len(water) and water.geometry.is_valid.all() and places.geometry.is_valid.all()
    streets = read_basemap(
        archive, "lines", clip_city,
        "highway IN ('trunk','primary','secondary','tertiary','residential','pedestrian',"
        "'footway','path','steps','service','unclassified') OR railway = 'rail'",
    )
    blocks = read_basemap(
        archive, "multipolygons", clip_city,
        "building IS NOT NULL OR leisure IN ('park','garden') OR landuse IN ('forest','grass','cemetery')",
    )
    footprints = gpd.read_file(
        archive, layer="multipolygons", bbox=clip_city,
        where="osm_way_id IN ('1064894462','78816809') OR osm_id = '4060635'",
    ).to_crs(TARGET_CRS)
    assert len(footprints) == 3 and footprints.geometry.is_valid.all()

    oblast = gpd.read_file(pskokn / "pskov_okrugs.gpkg", layer="oblast_boundary")
    assert len(oblast) == 1 and oblast.crs.to_epsg() == 4326 and oblast.geometry.is_valid.all()
    geographic_bounds = tuple(oblast.total_bounds)
    oblast = oblast.to_crs(TARGET_CRS)
    assert places.geometry.within(oblast.geometry.iloc[0]).all()
    regional_water = read_basemap(
        archive, "multipolygons", geographic_bounds,
        "natural = 'water' AND name IN ('Псковское озеро','Чудское озеро','Тёплое озеро','Пейпус','Великая')",
    )
    regional_roads = read_basemap(
        archive, "lines", geographic_bounds, "highway IN ('trunk','primary')",
    ).clip(oblast)
    fig, ax = map_figure(
        "Псков и наследие области",
        "Музейные и мемориальные связи с примером юга области",
        "Музеи Изборска, Самолвы и Великих Лук, Псково-Печерский монастырь и Пушкинский заповедник.",
        OSM,
    )
    oblast.plot(ax=ax, facecolor=LAND, edgecolor="#8e8a83", linewidth=.8, zorder=1)
    regional_water.plot(ax=ax, color=WATER, edgecolor="none", zorder=2)
    regional_roads.plot(ax=ax, color=ROAD, linewidth=.45, zorder=3)
    pskov = places.geometry.loc["27025689"]
    places.drop(index=["27025689", SOUTH_COMPARISON]).plot(ax=ax, color=CHURCH, markersize=23, zorder=5)
    places.loc[[SOUTH_COMPARISON]].plot(ax=ax, color=ACTION, marker="s", markersize=34, zorder=5)
    ax.scatter(pskov.x, pskov.y, marker="D", s=40, color=WALL, zorder=6)
    for osm_id, (name, offset) in PLACES.items():
        point = places.geometry.loc[osm_id]
        label(ax, point, name, offset)
        if osm_id not in {"27025689", SOUTH_COMPARISON}:
            x, y = LineString([pskov, point]).xy
            ax.plot(x, y, linestyle="--", linewidth=1, color="#81778d", zorder=4)
    x0, y0, x1, y1 = oblast.total_bounds
    finish(fig, ax, (x0-37000, y0-7000, x1+20000, y1+7000), [
        symbol("Псков", WALL, "D"),
        symbol("Населённый пункт с объектами наследия", CHURCH, "o"),
        symbol("Сравнение с югом области", ACTION, "s"),
        Line2D([], [], color="#81778d", linestyle="--", label="Тематическая связь без трассы маршрута"),
        Patch(facecolor=LAND, edgecolor="#8e8a83", label="Граница области по OSM"),
        Patch(facecolor=WATER, label="Псковское озеро и Великая"),
        Line2D([], [], color=ROAD, label="Основные дороги"),
    ])
    fig.text(.735, .50, "Самолва · место памяти\nо Ледовом побоище.\nВеликие Луки · земляная крепость.", fontsize=8.5, va="top", linespacing=1.45)
    fig.text(.735, .38, "Дата исходной границы\nобласти не установлена.", fontsize=8, va="top", linespacing=1.45)
    save_map(fig, maps / "01_связи_области.png")

    components = pd.read_csv("data/raw/unesco_components.csv")
    coords = [
        components[f"{axis}_deg"] + components[f"{axis}_min"] / 60 + components[f"{axis}_sec"] / 3600
        for axis in ("lon", "lat")
    ]
    churches = gpd.GeoDataFrame(
        components[["component_id", "name", "dating_text"]].assign(
            feature_id="unesco:" + components.component_id, unesco_property_id=1523, source_url=UNESCO,
        ),
        geometry=gpd.points_from_xy(*coords), crs=4326,
    )
    church = churches.loc[churches.component_id.eq("1523-004")].copy()
    registry = gpd.read_file(processed / "heritage.gpkg", layer="egrokn_points_city")
    examples = registry.set_index("registry_number").loc[
        ["601510242910006", "601510265670056", "611210001500006", "611310001490005"]
    ]
    offset = church.to_crs(TARGET_CRS).geometry.iloc[0].distance(examples.to_crs(TARGET_CRS).geometry.iloc[0])
    assert len(churches) == 10 and churches.component_id.is_unique and offset < 10
    church["object_id"] = examples.iloc[0].object_id
    churches.loc[church.index, "object_id"] = church.object_id
    churches.to_file(processed / "unesco_components.geojson", driver="GeoJSON", index=False)
    churches = churches.to_crs(TARGET_CRS)
    tower, palace, station = examples.to_crs(TARGET_CRS).geometry.iloc[1:]

    named_ways = gpd.read_file(
        archive, layer="lines", bbox=clip_city,
        where="osm_id IN ('31312852','947627667')",
    ).set_index("osm_id")
    scenarios = pd.read_csv("reports/tables/scenarios.csv", dtype={"osm_way_id": str})
    scenarios = gpd.GeoDataFrame(
        scenarios,
        geometry=[
            Point(named_ways.geometry.loc[row.osm_way_id].coords[0 if row.endpoint == "first" else -1])
            for row in scenarios.itertuples()
        ],
        crs=4326,
    )
    assert len(scenarios) == 2 and scenarios.scenario_id.is_unique
    scenarios["feature_id"] = "scenario:" + scenarios.scenario_id
    scenarios["related_object_id"] = examples.iloc[0].object_id
    scenarios["territory_act"] = "Приказ № 879 от 31.12.2014, таблица координат в приложении к изменению от 14.04.2022"
    scenarios["zone_act"] = "Приказ Минкультуры России от 13.08.2024 № 1542"
    scenarios["zone_act_url"] = ZONE_ACT
    scenarios["geometry_role"] = "Точка обследования; решение условное"
    scenarios.to_file(maps / "03_точки_фломастер_плана.geojson", driver="GeoJSON", index=False)
    scenarios = scenarios.to_crs(TARGET_CRS)
    walk, ways = mapped_walk(streets, *scenarios.set_index("scenario_id").geometry.loc[["S1", "S2"]])
    gpd.GeoDataFrame(
        {"feature_id": ["osm:walk:S1-S2"], "length_m": [walk.length], "osm_way_ids": ["|".join(ways)]},
        geometry=[walk], crs=TARGET_CRS,
    ).to_crs(4326).to_file(processed / "pokrova_walk.geojson", driver="GeoJSON", index=False)

    fig, ax = map_figure(
        "Исторические слои Пскова",
        "От средневекового речного ядра до железнодорожного узла XIX века",
        "ЮНЕСКО, список 1523 (2019), Псковский музей-заповедник, Ленгипротранс, локальная выгрузка ЕГРОКН.",
        OSM,
    )
    draw_basemap(ax, blocks, streets[~streets["highway"].isin(["footway", "path", "steps"])], water)
    footprints.plot(ax=ax, facecolor=WALL, edgecolor=WALL, alpha=.70, zorder=6)
    churches.plot(ax=ax, color=CHURCH, markersize=34, edgecolor="white", linewidth=.7, zorder=7)
    ax.scatter(palace.x, palace.y, marker="s", color=PALACE, s=36, zorder=8)
    ax.scatter(station.x, station.y, marker="D", color=STATION, s=36, zorder=8)
    label(ax, palace, "Поганкины палаты", (7, 4))
    label(ax, station, "Вокзал", (7, -2))
    label(ax, footprints.loc[footprints.osm_way_id.eq("1064894462")].geometry.iloc[0].representative_point(), "Кром", (-8, 12))
    label(ax, churches.geometry.iloc[9], "Снетогорский монастырь", (8, 5))
    label(ax, churches.geometry.iloc[1], "Мирожский монастырь", (-8, 6))
    label(ax, church.to_crs(TARGET_CRS).geometry.iloc[0], "Покровский угол", (-8, -9))
    x0, y0, x1, y1 = churches.total_bounds
    finish(fig, ax, (x0-550, min(y0, station.y)-530, station.x+900, y1+510), [
        symbol("Десять компонентов ЮНЕСКО", CHURCH, "o"),
        Patch(facecolor=WALL, label="Фрагменты крепости по OSM"),
        symbol("Купеческие палаты XVII в.", PALACE, "s"),
        symbol("Вокзал XIX в. с послевоенным восстановлением", STATION, "D"),
        Patch(facecolor=WATER, label="Водные объекты"),
        Patch(facecolor=BUILDINGS, label="Фоновая застройка"),
        Patch(facecolor=GREEN, label="Парки и зелёные территории"),
        Line2D([], [], color=RAIL, linestyle="--", label="Железная дорога"),
    ])
    save_map(fig, maps / "02_исторические_слои.png")

    fig, ax = map_figure(
        "Покровский угол и набережная",
        "Самостоятельная схема обследования пешеходного подхода",
        "ЮНЕСКО 1523-004, Псковский музей-заповедник, приказы № 879 с изменениями и № 1542.",
        OSM,
    )
    draw_basemap(ax, blocks, streets, water)
    footprints.plot(ax=ax, facecolor=WALL, edgecolor=WALL, zorder=6)
    named_ways.loc[["31312852"]].to_crs(TARGET_CRS).plot(
        ax=ax, color="#31647b", linewidth=1.35, linestyle="--", zorder=7,
    )
    ax.plot(*walk.xy, color=ACTION, linewidth=2.2, zorder=9)
    ch = churches.geometry.iloc[3]
    ax.scatter(ch.x, ch.y, s=48, color=CHURCH, edgecolor="white", zorder=8)
    label(ax, ch, "Церковь Покрова\nот Пролома", (8, 11))
    label(ax, tower, "Покровская башня", (6, -9))
    scenarios.plot(ax=ax, marker="^", markersize=78, color=ACTION, edgecolor="white", zorder=10)
    for row in scenarios.itertuples():
        label(ax, row.geometry, row.scenario_id, (-7, -5), fontweight="bold")
    label(ax, Point(ch.x-140, ch.y-40), "Великая", (-2, 6))
    street = named_ways.to_crs(TARGET_CRS).geometry.loc["947627667"]
    label(ax, street.interpolate(.45, normalized=True), "ул. Свердлова", (4, 8))
    square = blocks.loc[blocks["name"].eq("Площадь Героев-десантников")]
    if not square.empty:
        label(ax, square.geometry.union_all().representative_point(), "пл. Героев-десантников", (5, -6))
    finish(fig, ax, (ch.x-330, ch.y-265, ch.x+360, ch.y+250), [
        symbol("Церковь, компонент ЮНЕСКО", CHURCH, "o"),
        Patch(facecolor=WALL, label="Контур башни по OSM"),
        Line2D([], [], color="#31647b", linewidth=1.35, linestyle="--", label="Пешеходная набережная"),
        Line2D([], [], color=ACTION, linewidth=2, label=f"Связь S1–S2 по OSM · {round(walk.length)} м"),
        symbol("Точка обследования", ACTION, "^"),
        Patch(facecolor=BUILDINGS, label="Фоновая застройка"),
        Patch(facecolor=GREEN, label="Озеленение"),
    ])
    fig.text(.735, .53, "S1  Подход с ул. Свердлова\nS2  Остановка на набережной",
             fontsize=8.5, va="top", linespacing=1.5)
    fig.text(.735, .43, "Документированные ограничения", color=ACT, fontsize=8.4, fontweight="bold")
    fig.text(.735, .408, "Территория церкви — № 879.\nОбъединённая зона — № 1542.\nГраницы показаны на карте 04.",
             fontsize=8, va="top", linespacing=1.5)
    save_map(fig, maps / "03_покровский_угол.png")
    return {
        "components": len(churches), "field_check_sites": len(scenarios),
        "pokrova_registry_offset_m": offset, "walk_length_m": walk.length,
    }


if __name__ == "__main__":
    print(build_maps())
