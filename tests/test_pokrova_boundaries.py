from scripts.plot_pokrova_boundaries import read_boundaries


def test_church_territory_area_and_documented_zone_topology():
    areas = read_boundaries()
    assert round(areas["TERR"].area) == 4498
    assert len(areas["EZRZ-3"].interiors) == len(areas["EZRZ-5"].interiors) == 1
    assert [name for name, area in areas.items() if not area.is_valid] == ["EZRZ-3"]
