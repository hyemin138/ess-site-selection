"""공통: 데이터 로드, 입지 지표 계산, AHP 가중합"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

# 보고서 기준 공공 운영기관 (그 외는 민간)
PUBLIC_OPERATORS = "환경부|한국전력|한국전기차충전서비스"

# 발표자료 기준 7개 지표 가중치
WEIGHTS = {
    "demand_density": 0.30,
    "peak_potential": 0.25,
    "solar_potential": 0.15,
    "grid_feasibility": 0.10,
    "charger_density": 0.10,
    "climate_impact": 0.05,
    "space_availability": 0.05,
}
LOWER_IS_BETTER = ["charger_density"]

# 기후 및 환경 영향 = 기온 적합도, 상대습도, 총강수량 정규화 가중평균 (습도, 강수는 낮을수록 유리)
CLIMATE_WEIGHTS = {"temp_fit": 0.05, "습도": 0.05, "강수량": 0.025}

LABELS = {
    "demand_density": "수요밀도", "peak_potential": "피크수요", "solar_potential": "재생에너지",
    "grid_feasibility": "계통연계", "charger_density": "충전소밀집도",
    "climate_impact": "기후환경", "space_availability": "공간활용성",
}

WEATHER_COLS = ["지표면 단파 복사(산란)", "지표면 단파 복사(하향)", "지표 기압", "2m 지상 상대습도",
                "2m 지상 기온", "10m 지상풍속", "수직 일사량", "총 강수량"]
WEATHER_REF = "고경면"  # 컬럼 정렬 기준 읍면동


def set_korean_font():
    """설치된 한글 폰트로 matplotlib 설정"""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for font in ["NanumGothic", "NanumBarunGothic", "Noto Sans CJK KR", "Malgun Gothic", "AppleGothic"]:
        if font in names:
            plt.rc("font", family=font)
            break
    plt.rc("axes", unicode_minus=False)


def out_dir(name):
    """결과 폴더 생성"""
    path = RESULTS / name
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------- 로드 ----------
def load_chargers(public_only=True):
    """정제된 충전기 데이터 (01번 결과, 충전기 단위)"""
    df = pd.read_csv(PROCESSED / "chargers.csv", dtype={"statId": str})
    if public_only:
        df = df[df["busiNm"].astype(str).str.contains(PUBLIC_OPERATORS)]
    return df


def stations(chargers):
    """충전소 단위 집계: 총 출력, 충전기 수, 중복 이름 구분"""
    st = (chargers.groupby("statId")
          .agg(statNm=("statNm", "first"), lat=("lat", "first"), lng=("lng", "first"),
               읍면동=("읍면동", "first"), 총출력=("output", "sum"), 충전기수=("chgerId", "count"))
          .reset_index())
    # 같은 이름의 충전소는 (1), (2)로 구분
    dup = st["statNm"].duplicated(keep=False)
    st.loc[dup, "statNm"] = (st.loc[dup, "statNm"] + "("
                             + (st[dup].groupby("statNm").cumcount() + 1).astype(str) + ")")
    return st


def load_population():
    """읍면동별 20세 이상 인구"""
    df = pd.read_csv(RAW / "population_2023.csv")
    return (df.groupby("행정구역별(읍면동)")["총인구(명)"].sum()
            .rename("인구수").rename_axis("읍면동").reset_index())


def align_weather_columns(df, ref=WEATHER_REF):
    """읍면동별 컬럼 순서를 기준 읍면동과 일별 시계열 상관으로 정렬"""
    ref_df = df[df["읍면동"] == ref].sort_values(["월", "일"])[WEATHER_COLS].reset_index(drop=True)
    fixed = []
    for dong, g in df.groupby("읍면동"):
        g = g.sort_values(["월", "일"]).reset_index(drop=True)
        corr = np.array([[np.corrcoef(g[a], ref_df[b])[0, 1] if g[a].std() > 0 else 0
                          for b in WEATHER_COLS] for a in WEATHER_COLS])
        mapping, used = {}, set()
        # 상관 높은 쌍부터 1:1 배정
        for i, j in sorted(np.ndindex(corr.shape), key=lambda ij: -corr[ij]):
            if i in mapping or j in used:
                continue
            mapping[i], used = j, used | {j}
        out = g.copy()
        out[[WEATHER_COLS[mapping[i]] for i in range(len(WEATHER_COLS))]] = g[WEATHER_COLS].values
        fixed.append(out)
    return pd.concat(fixed, ignore_index=True)


def load_weather():
    """2023년 기상 연평균 (영천시 ASOS 관측 지점 1곳 → 모든 읍면동에 시 평균 적용)"""
    df = align_weather_columns(pd.read_csv(RAW / "weather_2023.csv"))
    agg = df.groupby("읍면동")[WEATHER_COLS].mean().reset_index()
    agg[WEATHER_COLS] = agg[WEATHER_COLS].mean().values
    return agg.rename(columns={"지표면 단파 복사(하향)": "GHI", "수직 일사량": "DNI", "2m 지상 기온": "기온",
                               "2m 지상 상대습도": "습도", "총 강수량": "강수량"})


# ---------- 지표 ----------
def haversine(lat1, lon1, lat2, lon2):
    """두 지점 간 거리(km)"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def minmax(s, lower_is_better=False):
    """0-1 정규화, 값이 모두 같으면 1"""
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(1.0, index=s.index)
    return (hi - s) / (hi - lo) if lower_is_better else (s - lo) / (hi - lo)


def build_indicators(st, pop, weather):
    """충전소별 7개 입지 지표"""
    df = st.merge(pop, on="읍면동", how="left").merge(weather, on="읍면동", how="left")
    # 읍면동 매칭 실패 시 평균 대체
    for col in ["인구수", "GHI", "DNI", "기온", "습도", "강수량"]:
        df[col] = df[col].fillna(df[col].mean())

    lat, lng = df["lat"].values, df["lng"].values
    dist = haversine(lat[:, None], lng[:, None], lat[None, :], lng[None, :])
    np.fill_diagonal(dist, np.inf)

    df["demand_density"] = df["총출력"] * df["인구수"]
    df["peak_potential"] = df["총출력"]
    df["solar_potential"] = 0.7 * minmax(df["GHI"]) + 0.3 * minmax(df["DNI"])
    df["grid_feasibility"] = 1  # 계통 데이터 미확보, 상수
    df["charger_density"] = (dist < 1).sum(axis=1)  # 반경 1km 내 타 충전소 수
    df["temp_fit"] = np.maximum(0, (20 - (df["기온"] - 22.5).abs()) / 20)  # 기온 적합도 (22.5℃ 기준)
    climate = {"temp_fit": minmax(df["temp_fit"]),
               "습도": minmax(df["습도"], lower_is_better=True),
               "강수량": minmax(df["강수량"], lower_is_better=True)}
    df["climate_impact"] = sum(climate[k] * w for k, w in CLIMATE_WEIGHTS.items()) / sum(CLIMATE_WEIGHTS.values())
    df["space_availability"] = 1  # 부지 데이터 미확보, 상수
    return df


def ahp_score(ind, weights=WEIGHTS):
    """Min-Max 정규화 후 가중합"""
    norm = pd.DataFrame({c: minmax(ind[c], c in LOWER_IS_BETTER) for c in weights})
    score = sum(norm[c] * w for c, w in weights.items())
    return norm, score


def score_table(weights=WEIGHTS):
    """충전소별 지표 점수와 최종 점수 (내림차순)"""
    st = stations(load_chargers())
    ind = build_indicators(st, load_population(), load_weather())
    norm, score = ahp_score(ind, weights)
    out = pd.concat([ind[["statId", "statNm", "읍면동", "lat", "lng", "총출력", "충전기수"]],
                     norm.rename(columns=LABELS)], axis=1)
    out["최종점수"] = score
    return out.sort_values("최종점수", ascending=False).reset_index(drop=True)
