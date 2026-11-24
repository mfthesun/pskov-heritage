import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pyproj import Transformer
import pytest

from scripts import maptools


def test_layout_and_saved_image_size(tmp_path):
    fig, ax = maptools.map_figure(
        "Проверка макета",
        "Пояснение вне поля карты",
        "локальный источник, дата среза",
        "© участники OpenStreetMap, ODbL",
    )
    legend = maptools.finish(
        fig, ax, (480_000, 6_400_000, 490_000, 6_410_000),
        [Line2D([], [], marker="o", color="#9b2c2c", linestyle="None", label="Объекты наследия")],
    )
    assert ax.get_position().x1 < maptools.LEGEND_ANCHOR[0]
    assert legend.get_frame().get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    assert legend.get_title().get_text() == "Условные обозначения"
    assert fig.texts[0].get_position()[1] > ax.get_position().y1

    output = tmp_path / "layout.png"
    maptools.save_map(fig, output)
    image = plt.imread(output)
    assert image.shape[:2] == (1400, 2000)


@pytest.mark.parametrize(
    ("width", "height", "expected", "label"),
    [
        (100, 78.6, 20, "20 м"),
        (800, 500, 200, "200 м"),
        (200_000, 120_000, 50_000, "50 км"),
    ],
)
def test_scale_bar_is_metric_and_adaptive(width, height, expected, label):
    fig, ax = maptools.map_figure()
    maptools.set_extent(ax, 500_000, 6_400_000, 500_000 + width, 6_400_000 + height)
    distance = maptools.scale_bar(ax)

    assert distance == expected
    segments = [patch for patch in ax.patches if patch.get_zorder() == 20]
    assert len(segments) == 4
    assert math.isclose(sum(patch.get_width() for patch in segments), distance)
    assert ax.texts[-1].get_text() == label

    fig.canvas.draw()
    bounds = ax.get_window_extent()
    renderer = fig.canvas.get_renderer()
    background = next(patch for patch in ax.patches if patch.get_zorder() == 19)
    frame = background.get_window_extent(renderer)
    for label_artist in ax.texts:
        label_box = label_artist.get_window_extent(renderer)
        assert frame.contains(label_box.x0 - 2, label_box.y0 - 2)
        assert frame.contains(label_box.x1 + 2, label_box.y1 + 2)
    assert bounds.contains(frame.x0, frame.y0) and bounds.contains(frame.x1, frame.y1)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    meters_per_pixel_x = (x1 - x0) / bounds.width
    meters_per_pixel_y = (y1 - y0) / bounds.height
    assert math.isclose(meters_per_pixel_x, meters_per_pixel_y, rel_tol=1e-12)
    plt.close(fig)


def test_true_north_at_pskov():
    x, y = Transformer.from_crs(4326, maptools.TARGET_CRS, always_xy=True).transform(28.33, 57.82)
    angle = maptools.true_north_angle(x, y)
    expected = -math.atan(math.tan(math.radians(28.33 - 27)) * math.sin(math.radians(57.82)))
    assert angle < 0
    assert math.isclose(angle, expected, abs_tol=1e-5)
