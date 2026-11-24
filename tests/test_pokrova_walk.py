import geopandas as gpd
from shapely.geometry import LineString, Point

from scripts.plot_heritage import mapped_walk


def test_mapped_walk_uses_shared_vertices_and_walkable_way_types():
    streets = gpd.GeoDataFrame(
        {"highway": ["footway", "steps", "primary"], "osm_id": ["1", "2", "3"]},
        geometry=[
            LineString([(0, 0), (3, 0)]),
            LineString([(3, 0), (3, 4)]),
            LineString([(0, 0), (3, 4)]),
        ], crs=32635,
    )
    route, ways = mapped_walk(streets, Point(0, 0), Point(3, 4))
    assert route.length == 7
    assert ways == ["1", "2"]
