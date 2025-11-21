
import pytest
from datetime import datetime
from app.core.indexing.spatial_index import SpatialIndex

@pytest.fixture
def sample_imagery():
    return [
        {
            "tile_id": "tile_1",
            "zip": "77002",
            "bbox": [-95.37, 29.75, -95.36, 29.76],
            "timestamp": "2017-08-28T12:00:00"
        },
        {
            "tile_id": "tile_2",
            "zip": "77002",
            "bbox": [-95.38, 29.75, -95.37, 29.76],
            "timestamp": "2017-08-29T12:00:00"
        },
        {
            "tile_id": "tile_nearby",
            "zip": "77003", # Neighboring ZIP
            "bbox": [-95.35, 29.75, -95.34, 29.76],
            "timestamp": "2017-08-28T12:00:00"
        }
    ]

@pytest.fixture
def sample_sensors():
    return [
        {
            "sensor_id": "sensor_1",
            "zip": "77002",
            "lat": 29.755,
            "lon": -95.365,
            "timestamp": "2017-08-28T12:00:00"
        },
        {
            "sensor_id": "sensor_nearby",
            "zip": "77003",
            "lat": 29.755,
            "lon": -95.345,
            "timestamp": "2017-08-28T12:00:00"
        }
    ]

def test_spatial_index_exact_match(sample_imagery, sample_sensors):
    idx = SpatialIndex(sample_imagery, sample_sensors)
    start = datetime(2017, 8, 27)
    end = datetime(2017, 8, 30)
    
    tiles = idx.get_tiles_by_zip("77002", start, end, k=5)
    # Code falls back to spatial radius if len(exact) < k.
    # So we might get the neighbor tile too.
    assert len(tiles) >= 2
    ids = {t["tile_id"] for t in tiles}
    assert "tile_1" in ids
    assert "tile_2" in ids

def test_spatial_index_fallback(sample_imagery, sample_sensors):
    idx = SpatialIndex(sample_imagery, sample_sensors)
    start = datetime(2017, 8, 27)
    end = datetime(2017, 8, 30)
    
    # Request more tiles than available in exact ZIP to trigger fallback
    # 77002 has 2 tiles. We ask for 5.
    # It should find tile_nearby (77003) because it's physically close.
    tiles = idx.get_tiles_by_zip("77002", start, end, k=5)
    
    # We expect 3 tiles: 2 exact + 1 nearby
    assert len(tiles) >= 3
    ids = {t["tile_id"] for t in tiles}
    assert "tile_1" in ids
    assert "tile_2" in ids
    # This assertion depends on pgeocode correctly placing 77002 near 77003
    # and the radius being large enough.
    # If pgeocode fails (e.g. no internet), this might fail.
    
def test_sensor_spatial_search(sample_imagery, sample_sensors):
    idx = SpatialIndex(sample_imagery, sample_sensors)
    
    # 77002 has 1 sensor. We ask for 3.
    sensors = idx.nearest_sensors_by_zip("77002", n=3)
    assert len(sensors) >= 2
    ids = {s["sensor_id"] for s in sensors}
    assert "sensor_1" in ids
    assert "sensor_nearby" in ids
