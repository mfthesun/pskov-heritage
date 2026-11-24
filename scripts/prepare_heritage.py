"""Воспроизводимая ГИС-сверка ЕГРОКН, OKN_Pskov_Region и heritage GeoJSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urljoin

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

PSKOKN = Path("/home/mfthesun/Workspace/ИИ, анализ данных и моделирование в урбанистике/PSKOKN")
BLOCKSNET = Path("/home/mfthesun/Projects/pskov-blocksnet")
GPKG_LAYERS = ["Heritage_Points", "Heritage_Polygons", "Protection_Zones", "Settlements"]
CITY_LAYERS = {
    "Heritage_Points": "heritage_points_city",
    "Heritage_Polygons": "heritage_polygons_city",
    "Protection_Zones": "protection_zones_city",
    "Settlements": "settlements_city",
}
BLOCKSNET_LAYERS = {
    "Heritage_Points": "heritage_points.geojson",
    "Heritage_Polygons": "heritage_polygons.geojson",
    "Protection_Zones": "heritage_protection_zones.geojson",
}
STATUS_MAP = {
    "1": "federal",
    "2": "regional",
    "4": "identified",
    "федеральный": "federal",
    "федерального значения": "federal",
    "региональный": "regional",
    "регионального значения": "regional",
    "выявленный": "identified",
}
ZONE_TYPE_MAP = {
    "Зона охраны культурного объектаЗона охраны культурного объекта": "Зона охраны культурного объекта",
}


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def clean_series(series: pd.Series) -> pd.Series:
    return series.map(clean_text)


def normalise_name(value: object) -> str:
    return re.sub(r"\s+", " ", clean_text(value).lower().replace("ё", "е"))


def normalise_status(value: object) -> str:
    return STATUS_MAP.get(normalise_name(value), "")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_boundary(pskokn: Path = PSKOKN) -> gpd.GeoDataFrame:
    path = pskokn / "boundaries" / "городской округ Псков.gpkg"
    boundary = gpd.read_file(path)
    if len(boundary) != 1 or boundary.geometry.isna().any() or not boundary.geometry.is_valid.all():
        raise ValueError("Ожидалась одна непустая валидная граница городского округа Пскова")
    if clean_text(boundary.iloc[0]["osm_id"]) != "1050399":
        raise ValueError("Граница не соответствует OSM relation 1050399")
    return boundary.to_crs(4326)


def _duplicate_group_ids(geometries: gpd.GeoSeries) -> pd.Series:
    wkb = geometries.to_wkb()
    counts = wkb[wkb.notna()].map(wkb[wkb.notna()].value_counts())
    hashes = wkb.map(lambda value: hashlib.sha256(value).hexdigest()[:16] if value is not None else "")
    return hashes.where(counts.fillna(0).gt(1), ""), counts.fillna(0).astype(int)


def add_feature_identifiers(gdf: gpd.GeoDataFrame, layer: str) -> gpd.GeoDataFrame:
    """Сохраняет все записи; record_uid уникален даже при повторном GLOBALID."""
    out = gdf.copy()
    out["source_fid"] = out.index.astype(str)
    out["source_layer"] = layer
    gml = clean_series(out["gml_id"]) if "gml_id" in out else pd.Series("", index=out.index, dtype=str)
    globalid = clean_series(out["GLOBALID"]) if "GLOBALID" in out else pd.Series("", index=out.index, dtype=str)
    fallback = pd.Series("fid:" + out["source_fid"], index=out.index, dtype=str)
    source_key = globalid.where(globalid.ne(""), gml.where(gml.ne(""), fallback))
    record_key = gml.where(gml.ne(""), fallback)
    out["feature_id"] = layer + ":" + source_key
    out["record_uid"] = layer + ":" + record_key
    registry = clean_series(out["REG_NUMBER"]) if "REG_NUMBER" in out else pd.Series("", index=out.index, dtype=str)
    out["registry_number"] = registry
    local_key = globalid.where(globalid.ne(""), record_key)
    out["object_id"] = (layer + ":" + local_key).mask(registry.ne(""), "okn:" + registry)
    out["component_id"] = ""
    out["ensemble_source_id"] = ""
    out["name"] = clean_series(out["NAME"]) if "NAME" in out else ""
    out["address"] = clean_series(out["ADDRESS"]) if "ADDRESS" in out else ""
    out["source_number"] = clean_series(out["NUMBER"]) if "NUMBER" in out else ""
    out["status"] = clean_series(out["Category"]).map(normalise_status) if "Category" in out else ""
    out["geometry_is_null"] = out.geometry.isna()
    out["geometry_is_empty"] = out.geometry.is_empty.fillna(False)
    out["geometry_is_valid"] = out.geometry.is_valid.where(out.geometry.notna(), pd.NA)
    out["geometry_repaired"] = False
    invalid = out.geometry.notna() & ~out.geometry.is_valid
    if invalid.any():
        out.loc[invalid, "geometry"] = out.loc[invalid, "geometry"].make_valid()
        out.loc[invalid, "geometry_repaired"] = True
    duplicate_group, duplicate_count = _duplicate_group_ids(gdf.geometry)
    out["geometry_duplicate_group"] = duplicate_group
    out["geometry_duplicate_count"] = duplicate_count
    out["globalid_duplicate_count"] = globalid.map(globalid[globalid.ne("")].value_counts()).fillna(0).astype(int)
    out["feature_id_duplicate_count"] = out["feature_id"].map(out["feature_id"].value_counts()).astype(int)
    if not out["record_uid"].is_unique:
        raise ValueError(f"record_uid не уникален в слое {layer}")
    return out.reset_index(drop=True)


def mark_city_selection(gdf: gpd.GeoDataFrame, boundary: BaseGeometry) -> gpd.GeoDataFrame:
    out = gdf.copy()
    nonempty = out.geometry.notna() & ~out.geometry.is_empty
    out["in_city_intersects"] = False
    out["in_city_within"] = False
    out.loc[nonempty, "in_city_intersects"] = out.loc[nonempty, "geometry"].intersects(boundary)
    out.loc[nonempty, "in_city_within"] = out.loc[nonempty, "geometry"].within(boundary)
    return out


def clip_to_city(gdf: gpd.GeoDataFrame, boundary: BaseGeometry, clip_polygons: bool) -> gpd.GeoDataFrame:
    city = gdf[gdf["in_city_intersects"]].copy()
    city["geometry_clipped_to_city"] = False
    if clip_polygons and not city.empty:
        city["geometry"] = city.geometry.make_valid().intersection(boundary)
        city = city[city.geometry.notna() & ~city.geometry.is_empty].copy()
        city["geometry_clipped_to_city"] = True
    return city.reset_index(drop=True)


def read_gpkg_layers(pskokn: Path, boundary: BaseGeometry) -> dict[str, gpd.GeoDataFrame]:
    path = pskokn / "данные и материалы" / "OKN_Pskov_Region.gpkg"
    layers = {}
    for layer in GPKG_LAYERS:
        source = gpd.read_file(path, layer=layer, fid_as_index=True).to_crs(4326)
        prepared = add_feature_identifiers(source, layer)
        if layer == "Protection_Zones":
            prepared["zone_type_raw"] = clean_series(prepared["ZONE_DESC"])
            prepared["zone_type"] = prepared["zone_type_raw"].replace(ZONE_TYPE_MAP)
            prepared["zone_requisites_status"] = "act_requisites_not_documented"
        prepared = mark_city_selection(prepared, boundary)
        layers[layer] = prepared
    return layers


def parse_map_point(value: object) -> Point | None:
    text = clean_text(value)
    if not text:
        return None
    try:
        data = json.loads(text)
        if data.get("type") != "Point":
            return None
        x, y = data["coordinates"][:2]
        point = Point(float(x), float(y))
        return point if point.is_valid and np.isfinite([point.x, point.y]).all() else None
    except (ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError):
        return None


def parse_document_requisites(value: object) -> tuple[str, str]:
    text = clean_text(value)
    if not text:
        return "", ""
    try:
        documents = json.loads(text)
    except json.JSONDecodeError:
        return "", ""
    requisites, urls = [], []
    for document in documents if isinstance(documents, list) else []:
        if not isinstance(document, dict):
            continue
        kind = clean_text(document.get("name"))
        number = clean_text(document.get("number"))
        date = clean_text(document.get("date"))
        label = " ".join(part for part in [kind, f"№ {number}" if number else "", f"от {date}" if date else ""] if part)
        if label:
            requisites.append(label)
        archive = document.get("archive")
        if isinstance(archive, dict) and clean_text(archive.get("url")):
            urls.append(clean_text(archive["url"]))
    return " | ".join(requisites), " | ".join(urls)


def read_egrokn(pskokn: Path, boundary: BaseGeometry) -> tuple[pd.DataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    source = pd.read_csv(pskokn / "egrokn_pskov.csv", dtype=str, keep_default_na=False)
    documents = source["наименование документа"].map(parse_document_requisites)
    geometries = source["На карте"].map(parse_map_point)
    has_coordinates = geometries.notna()
    records = pd.DataFrame(
        {
            "source_row": source.index + 2,
            "source_dataset": "egrokn_pskov.csv",
            "object_id": "okn:" + source["Номер в реестре"].str.strip(),
            "component_id": "",
            "feature_id": "egrokn:" + source["Id - идентификатор"].str.strip(),
            "record_uid": "egrokn:" + source["Id - идентификатор"].str.strip(),
            "registry_number": source["Номер в реестре"].str.strip(),
            "registry_record_id": source["Id - идентификатор"].str.strip(),
            "name": source["Объект"].str.strip(),
            "address": source["Полный адрес"].str.strip(),
            "region": source["Регион"].str.strip(),
            "category_raw": source["Категория историко-культурного значения"].str.strip(),
            "status": source["Категория историко-культурного значения"].map(normalise_status),
            "dating_text": source["дата создания"].str.strip(),
            "state_protection_year": source["год постановки ОКН на Госохрану"].str.strip(),
            "restoration_year": source["год проведения реставрации"].str.strip(),
            "subject_of_protection": source["описание предмета охраны"].str.strip(),
            "boundary_description": source["текстовое описание границ"].str.strip(),
            "zone_regimes_raw": source["зоны охраны и режимы использования земель"].str.strip(),
            "ensemble_source_id": source["идентификатор ансамбля"].str.strip(),
            "record_created_at": source["дата создания объекта"].str.strip(),
            "record_updated_at": source["дата обновления объекта"].str.strip(),
            "actual_flag": source["признак актуальности"].str.strip(),
            "public_flag": source["признак публичности"].str.strip(),
            "source_url": source["относительный URL объекта"].map(lambda value: urljoin("https://okn-mk.mkrf.ru", value.strip())),
            "documents_raw": source["наименование документа"].str.strip(),
            "document_requisites": documents.map(lambda item: item[0]),
            "document_urls": documents.map(lambda item: item[1]),
            "coordinate_status": has_coordinates.map({True: "point_geojson", False: "missing_or_unparsed"}),
            "has_coordinates": has_coordinates,
            "longitude": geometries.map(lambda geom: geom.x if geom is not None else np.nan),
            "latitude": geometries.map(lambda geom: geom.y if geom is not None else np.nan),
        }
    )
    records["component_id"] = records["ensemble_source_id"].mask(
        records["ensemble_source_id"].ne(""), "egrokn_component:" + records["registry_record_id"]
    )
    records["in_city_intersects"] = False
    records["in_city_within"] = False
    for index in records.index[has_coordinates]:
        geom = geometries.iloc[index]
        records.at[index, "in_city_intersects"] = geom.intersects(boundary)
        records.at[index, "in_city_within"] = geom.within(boundary)
    points = gpd.GeoDataFrame(
        records.loc[has_coordinates].copy(), geometry=geometries.loc[has_coordinates], crs=4326
    )
    return records, points, source


def geometry_summary(layers: dict[str, gpd.GeoDataFrame], egrokn_points: gpd.GeoDataFrame, egrokn_count: int) -> pd.DataFrame:
    rows = []
    for layer, gdf in [("egrokn_points", egrokn_points), *layers.items()]:
        null_count = int(egrokn_count - len(gdf)) if layer == "egrokn_points" else int(gdf.geometry.isna().sum())
        duplicate = gdf["geometry_duplicate_count"] if "geometry_duplicate_count" in gdf else pd.Series(0, index=gdf.index)
        globalid_duplicate = gdf["globalid_duplicate_count"] if "globalid_duplicate_count" in gdf else pd.Series(0, index=gdf.index)
        feature_duplicate = gdf["feature_id"].duplicated(keep=False)
        rows.append(
            {
                "layer": layer,
                "rows": len(gdf) if layer != "egrokn_points" else egrokn_count,
                "with_geometry": int(gdf.geometry.notna().sum()),
                "null_or_missing_geometry": null_count,
                "empty_geometry": int(gdf.geometry.is_empty.sum()),
                "invalid_nonempty_geometry": int(gdf["geometry_is_valid"].eq(False).sum()) if "geometry_is_valid" in gdf else int((gdf.geometry.notna() & ~gdf.geometry.is_valid).sum()),
                "repaired_geometry": int(gdf.get("geometry_repaired", pd.Series(False, index=gdf.index)).sum()),
                "exact_duplicate_geometry_rows": int(duplicate.gt(1).sum()),
                "exact_duplicate_geometry_groups": int(gdf.get("geometry_duplicate_group", pd.Series("", index=gdf.index)).ne("").sum() and gdf["geometry_duplicate_group"].replace("", pd.NA).dropna().nunique()),
                "duplicate_feature_id_rows": int(feature_duplicate.sum()),
                "duplicate_globalid_rows": int(globalid_duplicate.gt(1).sum()),
                "duplicate_globalid_groups": int(globalid_duplicate.gt(1).groupby(gdf.get("GLOBALID", pd.Series("", index=gdf.index))).any().sum()),
                "unique_record_uid": int(gdf["record_uid"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def spatial_summary(layers: dict[str, gpd.GeoDataFrame], records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    datasets = {"egrokn": records}
    datasets.update(layers)
    for layer, data in datasets.items():
        city = data[data["in_city_intersects"]]
        rows.append(
            {
                "dataset": layer,
                "records": len(data),
                "records_with_geometry": int(data["has_coordinates"].sum()) if layer == "egrokn" else int(data.geometry.notna().sum()),
                "records_without_geometry": int((~data["has_coordinates"]).sum()) if layer == "egrokn" else int(data.geometry.isna().sum()),
                "city_intersects": int(data["in_city_intersects"].sum()),
                "city_within": int(data["in_city_within"].sum()),
                "distinct_object_id_values_city": int(city["object_id"].nunique()),
                "unique_registry_numbers_city": int(city["registry_number"].replace("", pd.NA).dropna().nunique()),
                "selection_rule": "intersects OSM relation 1050399; polygons/zones additionally clipped; missing geometries not selected",
            }
        )
    return pd.DataFrame(rows)


def point_polygon_links(layers: dict[str, gpd.GeoDataFrame]) -> pd.DataFrame:
    points = layers["Heritage_Points"]
    polygons = layers["Heritage_Polygons"]
    left = points[points.geometry.notna()][["record_uid", "object_id", "feature_id", "name", "address", "GLOBALID", "in_city_intersects", "geometry"]]
    right = polygons[["record_uid", "object_id", "feature_id", "name", "address", "GLOBALID", "in_city_intersects", "geometry"]]
    joined = left.sjoin(right, predicate="intersects", how="inner", lsuffix="point", rsuffix="polygon")
    if joined.empty:
        return pd.DataFrame(columns=["point_record_uid", "polygon_record_uid", "relation_kind", "verification_status"])
    result = pd.DataFrame(
        {
            "point_record_uid": joined["record_uid_point"],
            "polygon_record_uid": joined["record_uid_polygon"],
            "point_object_id": joined["object_id_point"],
            "polygon_object_id": joined["object_id_polygon"],
            "point_feature_id": joined["feature_id_point"],
            "polygon_feature_id": joined["feature_id_polygon"],
            "point_name": joined["name_point"],
            "polygon_name": joined["name_polygon"],
            "point_in_city": joined["in_city_intersects_point"],
            "polygon_in_city": joined["in_city_intersects_polygon"],
            "globalid_match": clean_series(joined["GLOBALID_point"]).eq(clean_series(joined["GLOBALID_polygon"])) & clean_series(joined["GLOBALID_point"]).ne(""),
            "name_exact": joined["name_point"].map(normalise_name).eq(joined["name_polygon"].map(normalise_name)),
            "address_exact": joined["address_point"].map(normalise_name).eq(joined["address_polygon"].map(normalise_name)) & joined["address_point"].map(normalise_name).ne(""),
        }
    )
    result["relation_kind"] = np.where(result["globalid_match"], "globalid_and_spatial", "spatial_only")
    result["verification_status"] = np.where(
        result["globalid_match"], "source_identifier_and_spatial_match", "candidate_spatial_relation_requires_review"
    )
    return result


def _link_columns(left: pd.DataFrame, right: pd.DataFrame, method: str, target_layer: str) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "egrokn_record_uid": left["record_uid"].to_numpy(),
            "egrokn_object_id": left["object_id"].to_numpy(),
            "egrokn_registry_number": left["registry_number"].to_numpy(),
            "egrokn_name": left["name"].to_numpy(),
            "egrokn_status": left["status"].to_numpy(),
            "gpkg_record_uid": right["record_uid"].to_numpy(),
            "gpkg_object_id": right["object_id"].to_numpy(),
            "gpkg_feature_id": right["feature_id"].to_numpy(),
            "gpkg_name": right["name"].to_numpy(),
            "gpkg_status": right["status"].to_numpy(),
            "target_layer": target_layer,
            "match_method": method,
            "name_exact": left["name"].map(normalise_name).to_numpy() == right["name"].map(normalise_name).to_numpy(),
            "status_match": left["status"].to_numpy() == right["status"].to_numpy(),
            "egrokn_in_city": left["in_city_intersects"].to_numpy(),
            "gpkg_in_city": right["in_city_intersects"].to_numpy(),
        }
    )
    result["verification_status"] = "confirmed_registry_identifier" if method == "registry_number_exact" else "candidate_spatial_match_requires_review"
    return result


def egrokn_gpkg_links(egrokn_points: gpd.GeoDataFrame, layers: dict[str, gpd.GeoDataFrame]) -> pd.DataFrame:
    points = layers["Heritage_Points"]
    polygons = layers["Heritage_Polygons"]
    egr_registry = egrokn_points[egrokn_points["registry_number"].ne("")]
    gpkg_registry = points[points["registry_number"].ne("")]
    registry = egr_registry.merge(gpkg_registry, on="registry_number", suffixes=("_egr", "_gpkg"))
    links = []
    if not registry.empty:
        exact = _link_columns(
            registry[["record_uid_egr", "object_id_egr", "registry_number", "name_egr", "status_egr", "in_city_intersects_egr"]].rename(columns=lambda column: column.removesuffix("_egr")),
            registry[[column + "_gpkg" for column in ["record_uid", "object_id", "feature_id", "name", "status", "in_city_intersects"]]].rename(columns=lambda column: column[:-5]),
            "registry_number_exact",
            "Heritage_Points",
        )
        exact["distance_m"] = registry["geometry_egr"].to_crs(32635).reset_index(drop=True).distance(registry["geometry_gpkg"].to_crs(32635).reset_index(drop=True)).to_numpy()
        links.append(exact)
    nearest = egrokn_points.to_crs(32635).sjoin_nearest(
        points[points.geometry.notna()].to_crs(32635), how="inner", max_distance=1.0, distance_col="distance_m", lsuffix="egr", rsuffix="gpkg"
    )
    if not nearest.empty:
        near = _link_columns(
            nearest[[column + "_egr" for column in ["record_uid", "object_id", "registry_number", "name", "status", "in_city_intersects"]]].rename(columns=lambda column: column[:-4]),
            nearest[[column + "_gpkg" for column in ["record_uid", "object_id", "feature_id", "name", "status", "in_city_intersects"]]].rename(columns=lambda column: column[:-5]),
            "nearest_point_within_1m",
            "Heritage_Points",
        )
        near["distance_m"] = nearest["distance_m"].to_numpy()
        links.append(near)
    spatial = egrokn_points.sjoin(polygons, predicate="intersects", how="inner", lsuffix="egr", rsuffix="gpkg")
    if not spatial.empty:
        polygon_links = _link_columns(
            spatial[[column + "_egr" for column in ["record_uid", "object_id", "registry_number", "name", "status", "in_city_intersects"]]].rename(columns=lambda column: column[:-4]),
            spatial[[column + "_gpkg" for column in ["record_uid", "object_id", "feature_id", "name", "status", "in_city_intersects"]]].rename(columns=lambda column: column[:-5]),
            "point_intersects_polygon",
            "Heritage_Polygons",
        )
        polygon_links["distance_m"] = 0.0
        links.append(polygon_links)
    if not links:
        return pd.DataFrame()
    return pd.concat(links, ignore_index=True).drop_duplicates(
        subset=["egrokn_record_uid", "gpkg_record_uid", "match_method"]
    )


def status_discrepancies(layers: dict[str, gpd.GeoDataFrame], links: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for layer in ["Heritage_Points", "Heritage_Polygons"]:
        gdf = layers[layer]
        hist_status = gdf["HIST_CAT"].map(normalise_status)
        mismatch = hist_status.ne(gdf["status"]) & hist_status.ne("") & gdf["status"].ne("")
        for index in gdf.index[mismatch]:
            rows.append(
                {
                    "scope": "gpkg_internal",
                    "egrokn_record_uid": "",
                    "gpkg_record_uid": gdf.at[index, "record_uid"],
                    "match_method": "HIST_CAT_vs_Category",
                    "left_status": hist_status.iloc[index],
                    "right_status": gdf.at[index, "status"],
                    "verification_status": "confirmed_internal_attribute_discrepancy",
                }
            )
    if not links.empty:
        mismatch = links[~links["status_match"] & links["egrokn_status"].ne("") & links["gpkg_status"].ne("")]
        for _, row in mismatch.iterrows():
            rows.append(
                {
                    "scope": "egrokn_vs_gpkg",
                    "egrokn_record_uid": row["egrokn_record_uid"],
                    "gpkg_record_uid": row["gpkg_record_uid"],
                    "match_method": row["match_method"],
                    "left_status": row["egrokn_status"],
                    "right_status": row["gpkg_status"],
                    "verification_status": row["verification_status"],
                }
            )
    return pd.DataFrame(rows)


def ensemble_candidates(records: pd.DataFrame, layers: dict[str, gpd.GeoDataFrame]) -> pd.DataFrame:
    rows = []

    def add(source: str, row: pd.Series) -> None:
        name = clean_text(row["name"])
        explicit_id = clean_text(row.get("ensemble_source_id", ""))
        if explicit_id:
            rows.append(
                {
                    "source": source,
                    "record_uid": row["record_uid"],
                    "object_id": row["object_id"],
                    "feature_id": row["feature_id"],
                    "component_id": row["component_id"],
                    "ensemble_id": "egrokn_ensemble:" + explicit_id,
                    "ensemble_name_candidate": "",
                    "relation_type": "explicit_ensemble_identifier",
                    "verification_status": "source_identifier_present",
                }
            )
            return
        match = re.search(r"ансамбль\s+([^:：]+)\s*[:：]\s*(.+)", name, flags=re.IGNORECASE)
        if match:
            ensemble_name = match.group(1).strip(" «»\"")
            rows.append(
                {
                    "source": source,
                    "record_uid": row["record_uid"],
                    "object_id": row["object_id"],
                    "feature_id": row["feature_id"],
                    "component_id": "candidate:" + row["record_uid"],
                    "ensemble_id": "candidate:" + hashlib.sha256(normalise_name(ensemble_name).encode()).hexdigest()[:16],
                    "ensemble_name_candidate": ensemble_name,
                    "relation_type": "component_name_candidate",
                    "verification_status": "candidate_from_name_requires_registry_check",
                }
            )
        elif re.search(r"ансамбль|в составе", name, flags=re.IGNORECASE):
            rows.append(
                {
                    "source": source,
                    "record_uid": row["record_uid"],
                    "object_id": row["object_id"],
                    "feature_id": row["feature_id"],
                    "component_id": "",
                    "ensemble_id": "",
                    "ensemble_name_candidate": name,
                    "relation_type": "ensemble_or_composition_mention",
                    "verification_status": "candidate_from_name_requires_registry_check",
                }
            )

    for _, row in records.iterrows():
        add("egrokn", row)
    for layer in ["Heritage_Points", "Heritage_Polygons"]:
        for _, row in layers[layer].iterrows():
            add(layer, row)
    return pd.DataFrame(rows)


def compare_blocksnet(city_layers: dict[str, gpd.GeoDataFrame], blocksnet: Path) -> pd.DataFrame:
    rows = []
    for layer, filename in BLOCKSNET_LAYERS.items():
        source = city_layers[layer].reset_index(drop=True)
        other = gpd.read_file(blocksnet / "data" / "input" / filename).to_crs(4326).reset_index(drop=True)
        count = min(len(source), len(other))
        metric_offsets = source.iloc[:count].to_crs(32635).geometry.hausdorff_distance(
            other.iloc[:count].to_crs(32635).geometry
        )
        exact, equivalent = [], []
        for index in range(count):
            left, right = source.geometry.iloc[index], other.geometry.iloc[index]
            exact.append(left is not None and right is not None and left.equals_exact(right, 0))
            equivalent.append(left is not None and right is not None and left.equals(right))
        shared = [column for column in other.columns if column in source.columns and column not in {"geometry", "prepared_by"}]
        attribute_rows = []
        for index in range(count):
            attribute_rows.append(all(clean_text(source.at[index, column]) == clean_text(other.at[index, column]) for column in shared))
        rows.append(
            {
                "layer": layer,
                "new_selection_rows": len(source),
                "blocksnet_rows": len(other),
                "row_count_matches": len(source) == len(other),
                "aligned_geometry_equals_exact": int(sum(exact)),
                "aligned_geometry_topologically_equal": int(sum(equivalent)),
                "aligned_geometry_within_1m": int(metric_offsets.le(1).sum()),
                "max_aligned_hausdorff_m": float(metric_offsets.max()),
                "aligned_shared_attributes_equal": int(sum(attribute_rows)),
                "shared_attribute_columns": "|".join(shared),
                "comparison_note": "aligned rows; polygons and zones are clipped in both outputs; blocksnet lacks source identifiers",
            }
        )
    return pd.DataFrame(rows)


def gpkg_last_changes(path: Path) -> dict[str, str]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return dict(connection.execute("SELECT table_name, last_change FROM gpkg_contents").fetchall())
    finally:
        connection.close()


def source_inventory(pskokn: Path, blocksnet: Path, records: pd.DataFrame, raw_egrokn: pd.DataFrame, layers: dict[str, gpd.GeoDataFrame]) -> pd.DataFrame:
    gpkg_path = pskokn / "данные и материалы" / "OKN_Pskov_Region.gpkg"
    boundary_path = pskokn / "boundaries" / "городской округ Псков.gpkg"
    changes = gpkg_last_changes(gpkg_path)
    created = pd.to_datetime(records["record_created_at"], errors="coerce", utc=True)
    updated = pd.to_datetime(records["record_updated_at"], errors="coerce", utc=True)
    rows = [
        {
            "source_path": str(pskokn / "egrokn_pskov.csv"),
            "layer": "csv",
            "sha256": sha256(pskokn / "egrokn_pskov.csv"),
            "rows": len(raw_egrokn),
            "columns": len(raw_egrokn.columns),
            "crs": "EPSG:4326 for parsed GeoJSON points",
            "coverage": records["region"].value_counts().to_dict(),
            "temporal_evidence": f"record_created_at {created.min()}..{created.max()}; record_updated_at {updated.min()}..{updated.max()}",
            "date_slice_status": "export date not established; file timestamps are not a legal snapshot date",
            "provenance": "EGROKN fields and official record/document URLs; official dataset page https://opendata.mkrf.ru/opendata/7705851331-egrkn/",
        }
    ]
    for layer, gdf in layers.items():
        rows.append(
            {
                "source_path": str(gpkg_path),
                "layer": layer,
                "sha256": sha256(gpkg_path),
                "rows": len(gdf),
                "columns": len(gdf.columns),
                "crs": "source EPSG:3857; processed EPSG:4326",
                "coverage": "Pskov region according to filename/source catalog; completeness not independently confirmed",
                "temporal_evidence": "gpkg_contents.last_change=" + changes[layer],
                "date_slice_status": "technical layer change timestamp, not a legal snapshot date",
                "provenance": "inherited GPKG; no gpkg_metadata table; legal status requires object documents",
            }
        )
    boundary = gpd.read_file(boundary_path)
    rows.append(
        {
            "source_path": str(boundary_path),
            "layer": "city_boundary",
            "sha256": sha256(boundary_path),
            "rows": len(boundary),
            "columns": len(boundary.columns),
            "crs": str(boundary.crs),
            "coverage": "city of Pskov; OSM relation 1050399; not an official cadastral boundary",
            "temporal_evidence": "local file has population:date=2021; current OSM API checked separately",
            "date_slice_status": "local geometry extraction date not established",
            "provenance": "https://www.openstreetmap.org/api/0.6/relation/1050399.json",
        }
    )
    for layer, filename in BLOCKSNET_LAYERS.items():
        path = blocksnet / "data" / "input" / filename
        gdf = gpd.read_file(path)
        rows.append(
            {
                "source_path": str(path),
                "layer": layer,
                "sha256": sha256(path),
                "rows": len(gdf),
                "columns": len(gdf.columns),
                "crs": str(gdf.crs),
                "coverage": "features intersecting city boundary; polygons/zones clipped; not unique OKN",
                "temporal_evidence": "no independent snapshot date in GeoJSON",
                "date_slice_status": "derived file; inherit GPKG limitations",
                "provenance": "pskov-blocksnet/scripts/prepare_data.py:72-93; source identifiers omitted",
            }
        )
    return pd.DataFrame(rows)


def _json_value(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def run_pipeline(output_dir: Path, pskokn: Path = PSKOKN, blocksnet: Path = BLOCKSNET) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    boundary_gdf = read_boundary(pskokn)
    boundary = boundary_gdf.union_all()
    layers = read_gpkg_layers(pskokn, boundary)
    records, egrokn_points, raw_egrokn = read_egrokn(pskokn, boundary)
    egrokn_duplicate_group, egrokn_duplicate_count = _duplicate_group_ids(egrokn_points.geometry)
    egrokn_points["geometry_duplicate_group"] = egrokn_duplicate_group
    egrokn_points["geometry_duplicate_count"] = egrokn_duplicate_count
    city_layers = {layer: clip_to_city(gdf, boundary, layer != "Heritage_Points") for layer, gdf in layers.items()}
    egrokn_city = egrokn_points[egrokn_points["in_city_intersects"]].reset_index(drop=True)
    links = point_polygon_links(layers)
    registry_links = egrokn_gpkg_links(egrokn_points, layers)
    discrepancies = status_discrepancies(layers, registry_links)
    ensembles = ensemble_candidates(records, layers)
    blocksnet_comparison = compare_blocksnet(city_layers, blocksnet)
    geometry = geometry_summary(layers, egrokn_points, len(records))
    spatial = spatial_summary(layers, records)
    inventory = source_inventory(pskokn, blocksnet, records, raw_egrokn, layers)

    records.to_csv(output_dir / "egrokn_records.csv", index=False)
    geometry.to_csv(output_dir / "geometry_issues.csv", index=False)
    spatial.to_csv(output_dir / "spatial_selection.csv", index=False)
    links.to_csv(output_dir / "point_polygon_links.csv", index=False)
    registry_links.to_csv(output_dir / "egrokn_gpkg_links.csv", index=False)
    discrepancies.to_csv(output_dir / "status_discrepancies.csv", index=False)
    ensembles.to_csv(output_dir / "ensemble_candidates.csv", index=False)
    blocksnet_comparison.to_csv(output_dir / "blocksnet_comparison.csv", index=False)
    inventory.to_csv(output_dir / "source_inventory.csv", index=False)
    zone_types = layers["Protection_Zones"].groupby(["zone_type", "zone_requisites_status"], dropna=False).agg(
        region_features=("record_uid", "size"), city_features=("in_city_intersects", "sum")
    ).reset_index()
    zone_types.to_csv(output_dir / "zone_types.csv", index=False)

    gpkg_output = output_dir / "heritage.gpkg"
    outputs = {
        "city_boundary": boundary_gdf,
        "heritage_points_region": layers["Heritage_Points"],
        "heritage_polygons_region": layers["Heritage_Polygons"],
        "protection_zones_region": layers["Protection_Zones"],
        "settlements_region": layers["Settlements"],
        **{name: city_layers[layer] for layer, name in CITY_LAYERS.items()},
        "egrokn_points": egrokn_points,
        "egrokn_points_city": egrokn_city,
    }
    with TemporaryDirectory(dir=output_dir) as temp:
        staged = Path(temp) / gpkg_output.name
        for layer_name, gdf in outputs.items():
            renamed = {
                column: f"source_{column}"
                for column in gdf.columns
                if column != column.lower() and column.lower() in gdf.columns
            }
            gdf.rename(columns=renamed).to_file(staged, layer=layer_name, driver="GPKG", index=False)
        staged.replace(gpkg_output)

    summary = {
        "verification_date": "2026-11-23",
        "crs": {"exchange": "EPSG:4326", "metric": "EPSG:32635", "source_gpkg": "EPSG:3857"},
        "egrokn": {
            "records": len(records),
            "columns": len(raw_egrokn.columns),
            "unique_registry_numbers": records["registry_number"].nunique(),
            "with_coordinates": int(records["has_coordinates"].sum()),
            "without_coordinates": int((~records["has_coordinates"]).sum()),
            "city_points": len(egrokn_city),
            "regions": records["region"].value_counts().to_dict(),
            "explicit_ensemble_ids": int(records["ensemble_source_id"].ne("").sum()),
            "record_created_min": str(pd.to_datetime(records["record_created_at"], errors="coerce", utc=True).min()),
            "record_updated_max": str(pd.to_datetime(records["record_updated_at"], errors="coerce", utc=True).max()),
        },
        "gpkg": {
            layer: {
                "features": len(gdf),
                "city_intersects": int(gdf["in_city_intersects"].sum()),
                "city_output": len(city_layers[layer]),
                "null_geometry": int(gdf.geometry.isna().sum()),
                "invalid_nonempty_geometry": int(gdf["geometry_is_valid"].eq(False).sum()),
                "exact_duplicate_geometry_rows": int(gdf["geometry_duplicate_count"].gt(1).sum()),
                "duplicate_globalid_rows": int(gdf["globalid_duplicate_count"].gt(1).sum()),
            }
            for layer, gdf in layers.items()
        },
        "links": {
            "point_polygon_pairs": len(links),
            "point_polygon_globalid_matches": int(links["globalid_match"].sum()) if not links.empty else 0,
            "egrokn_gpkg_pairs": len(registry_links),
            "egrokn_gpkg_registry_matches": int((registry_links["match_method"] == "registry_number_exact").sum()) if not registry_links.empty else 0,
            "egrokn_gpkg_nearest_1m_pairs": int((registry_links["match_method"] == "nearest_point_within_1m").sum()) if not registry_links.empty else 0,
            "egrokn_gpkg_polygon_pairs": int((registry_links["match_method"] == "point_intersects_polygon").sum()) if not registry_links.empty else 0,
        },
        "ensembles": {
            "candidate_rows": len(ensembles),
            "component_name_candidates": int((ensembles["relation_type"] == "component_name_candidate").sum()) if not ensembles.empty else 0,
            "explicit_ensemble_ids": int((ensembles["relation_type"] == "explicit_ensemble_identifier").sum()) if not ensembles.empty else 0,
        },
        "status_discrepancies": {
            "gpkg_internal": int((discrepancies.get("scope", pd.Series(dtype=str)) == "gpkg_internal").sum()),
            "cross_source_all": int((discrepancies.get("scope", pd.Series(dtype=str)) == "egrokn_vs_gpkg").sum()),
            "cross_source_confirmed_registry": int((discrepancies.get("verification_status", pd.Series(dtype=str)) == "confirmed_registry_identifier").sum()),
        },
        "zone_types": zone_types.to_dict(orient="records"),
        "blocksnet": blocksnet_comparison.to_dict(orient="records"),
        "limitations": [
            "Feature counts are not counts of unique cultural heritage objects.",
            "Records without coordinates are not spatially assigned to the city.",
            "Spatial and name-based ensemble links are candidates, not confirmed registry identity.",
            "GPKG legal statuses and protection-zone acts require object-level primary documents.",
            "The city boundary is OSM relation 1050399, not an official cadastral boundary.",
        ],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(_json_value(summary), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--pskokn", type=Path, default=PSKOKN)
    parser.add_argument("--blocksnet", type=Path, default=BLOCKSNET)
    args = parser.parse_args()
    summary = run_pipeline(args.output_dir, args.pskokn, args.blocksnet)
    print(json.dumps(_json_value(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
