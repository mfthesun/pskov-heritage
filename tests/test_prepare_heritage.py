import geopandas as gpd
from shapely.geometry import Point, box

from scripts.prepare_heritage import egrokn_gpkg_links


def test_exact_registry_join_keeps_the_shared_registry_number():
    columns = {
        "object_id": ["okn:123"],
        "registry_number": ["123"],
        "name": ["Объект"],
        "status": ["federal"],
        "in_city_intersects": [True],
    }
    egrokn = gpd.GeoDataFrame(
        dict(columns, record_uid=["egrokn:1"], feature_id=["egrokn:1"]),
        geometry=[Point(28.33, 57.82)], crs=4326,
    )
    points = gpd.GeoDataFrame(
        dict(columns, record_uid=["points:1"], feature_id=["points:1"]),
        geometry=[Point(28.33, 57.82)], crs=4326,
    )
    polygons = gpd.GeoDataFrame(
        dict(columns, record_uid=["polygons:1"], feature_id=["polygons:1"]),
        geometry=[box(28.329, 57.819, 28.331, 57.821)], crs=4326,
    )
    links = egrokn_gpkg_links(egrokn, {"Heritage_Points": points, "Heritage_Polygons": polygons})
    exact = links.loc[links.match_method.eq("registry_number_exact")].iloc[0]
    assert exact.egrokn_registry_number == "123"
    assert exact.distance_m == 0
    assert exact.verification_status == "confirmed_registry_identifier"
