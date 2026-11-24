"""01. 충전기 데이터 정제: 출력(kW), 운영기관, 위경도 → 행정동 판별

입력: data/raw/ev_chargers_20230531.csv (한국환경공단 충전소 위치 및 운영정보, 영천시)
      data/raw/yeongcheon_dong.geojson (행정동 경계 16개)
출력: data/processed/chargers.csv (충전기 단위)
"""
import json

import numpy as np
import pandas as pd

import data as D

SLOW_KW = 7     # 완속 표준 출력
FAST_KW = 50    # 출력 표기 없는 급속(DC콤보 등) 기본값


def parse_output(row):
    """충전 출력(kW): 표기값 우선, 없으면 기종별 기본값"""
    for text in (row["급속충전량"], row["기종(소)"]):
        kw = pd.Series([str(text)]).str.extract(r"(\d+)kW")[0].iloc[0]
        if pd.notna(kw):
            return float(kw)
    return FAST_KW if row["기종(대)"] == "급속" else SLOW_KW


def in_polygon(x, y, ring):
    """ray casting 방식 점-다각형 포함 여부 (벡터화)"""
    ring = np.asarray(ring)
    xi, yi = ring[:, 0], ring[:, 1]
    xj, yj = np.roll(xi, 1), np.roll(yi, 1)
    cross = ((yi[None, :] > y[:, None]) != (yj[None, :] > y[:, None])) & \
            (x[:, None] < (xj - xi) * (y[:, None] - yi) / (yj - yi + 1e-15) + xi)
    return cross.sum(axis=1) % 2 == 1


def assign_dong(lat, lng, geojson_path):
    """위경도가 속한 행정동명"""
    features = json.load(open(geojson_path, encoding="utf-8"))["features"]
    dong = np.full(len(lat), None, dtype=object)
    for f in features:
        geom = f["geometry"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        inside = np.zeros(len(lat), dtype=bool)
        for poly in polys:
            # 외곽 포함 & 내부 구멍 제외
            hit = in_polygon(lng, lat, poly[0])
            for hole in poly[1:]:
                hit &= ~in_polygon(lng, lat, hole)
            inside |= hit
        dong[inside] = f["properties"]["ADM_NM"]
    return dong


def main():
    raw = pd.read_csv(D.RAW / "ev_chargers_20230531.csv")
    df = pd.DataFrame()
    latlng = raw["위도경도"].str.split(",", expand=True)
    df["lat"] = pd.to_numeric(latlng[0], errors="coerce")
    df["lng"] = pd.to_numeric(latlng[1], errors="coerce")
    df["statNm"] = raw["충전소명"].str.strip()
    df["busiNm"] = raw["운영기관(소)"].str.replace("(협회)", "", regex=False)
    df["기종"] = raw["기종(대)"]
    df["output"] = raw.apply(parse_output, axis=1)
    df["설치년도"] = raw["설치년도"]
    df = df.dropna(subset=["lat", "lng"]).reset_index(drop=True)

    # 파일 데이터에 충전소 ID가 없어 이름+좌표로 충전소 구분
    key = df["statNm"] + "_" + df["lat"].round(5).astype(str) + "_" + df["lng"].round(5).astype(str)
    df["statId"] = "S" + pd.Series(pd.factorize(key)[0] + 1).astype(str).str.zfill(3)
    df["chgerId"] = df.groupby("statId").cumcount() + 1

    df["읍면동"] = assign_dong(df["lat"].values, df["lng"].values, D.RAW / "yeongcheon_dong.geojson")

    outside = df["읍면동"].isna()
    print(f"충전기 {len(raw)}대 중 좌표 있는 {len(df)}대, 시 경계 밖 좌표 오류 {outside.sum()}대 제외")
    df = df[~outside]
    print(f"최종 충전기 {len(df)}대, 충전소 {df['statId'].nunique()}곳")
    print(df["읍면동"].value_counts().to_string())

    D.PROCESSED.mkdir(parents=True, exist_ok=True)
    cols = ["statId", "statNm", "chgerId", "lat", "lng", "output", "기종", "busiNm", "읍면동", "설치년도"]
    df[cols].to_csv(D.PROCESSED / "chargers.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
