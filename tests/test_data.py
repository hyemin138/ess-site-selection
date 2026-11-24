"""데이터 처리 검증"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data as D  # noqa: E402


def test_weather_citywide():
    """기상값은 모든 읍면동에 같은 시 평균, 항목별 값 범위 확인"""
    w = D.load_weather()
    assert w[["GHI", "기온", "습도"]].nunique().eq(1).all()
    assert w["지표 기압"].between(98, 100).all() and w["기온"].between(10, 16).all()


def test_population_16_dong():
    pop = D.load_population()
    assert len(pop) == 16 and pop["인구수"].gt(0).all()


def test_minmax():
    s = pd.Series([1, 2, 3])
    assert D.minmax(s).tolist() == [0, 0.5, 1]
    assert D.minmax(s, lower_is_better=True).tolist() == [1, 0.5, 0]
    assert D.minmax(pd.Series([5, 5])).tolist() == [1, 1]


def test_haversine_one_degree_lat():
    assert abs(D.haversine(36, 129, 37, 129) - 111.2) < 0.5


def test_duplicate_station_names():
    ch = pd.DataFrame({"statId": ["A", "B", "C"], "statNm": ["주차장", "주차장", "센터"],
                       "lat": [36, 36, 36], "lng": [129, 129, 129], "읍면동": ["동부동"] * 3,
                       "output": [50, 50, 7], "chgerId": ["1", "1", "1"]})
    names = D.stations(ch)["statNm"].tolist()
    assert names == ["주차장(1)", "주차장(2)", "센터"]


def test_weights_sum_to_one():
    assert np.isclose(sum(D.WEIGHTS.values()), 1)


def test_dong_assignment_matches_address():
    """주소에 읍면이 적힌 충전기는 경계 판별 결과와 일치"""
    import importlib
    prep = importlib.import_module("01_prepare_chargers")
    raw = pd.read_csv(D.RAW / "ev_chargers_20230531.csv").dropna(subset=["위도경도"])
    latlng = raw["위도경도"].str.split(",", expand=True).astype(float)
    addr = raw["주소"].str.extract(r"영천시\s+(\S+[읍면])\b")[0]
    geo = prep.assign_dong(latlng[0].values, latlng[1].values, D.RAW / "yeongcheon_dong.geojson")
    known = addr.notna().values & pd.notna(geo)
    assert known.sum() > 150
    assert (addr.values[known] == geo[known]).all()
