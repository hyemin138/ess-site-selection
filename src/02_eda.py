"""02. 현황 분석: 보급 추세, 태양광 비중, 운영 주체, 시간대별 이용, 읍면동별 인프라"""
import matplotlib.pyplot as plt
import pandas as pd

import data as D

OUT = D.out_dir("02_eda")
D.set_korean_font()

# 전기차 보급 물량 (영천시 보급사업 공고문 기준)
supply = pd.read_excel(D.RAW / "ev_supply_2021_2025.xlsx", sheet_name="Sheet1")
supply.columns = ["연도", "보급물량"]
supply = supply.sort_values("연도")
print("[보급 물량]\n", supply.to_string(index=False))

# 신재생에너지 중 태양광 비중
ren = pd.read_excel(D.RAW / "renewable_2023.xlsx")
ren.columns = ren.columns.str.strip()
ren["생산량"] = pd.to_numeric(ren["생산량"].astype(str).str.replace(",", ""))
solar_share = ren.loc[ren["에너지"] == "태양광", "생산량"].sum() / ren["생산량"].sum()
print(f"\n[태양광 비중] {solar_share:.1%}")

# 운영 주체: 공공 / 민간
chargers = D.load_chargers(public_only=False)
is_public = chargers["busiNm"].astype(str).str.contains(D.PUBLIC_OPERATORS)
st_public = is_public.groupby(chargers["statId"]).first()
print(f"\n[운영 주체] 충전기 {len(chargers)}대 중 공공 {is_public.mean():.1%}, 민간 {1 - is_public.mean():.1%}"
      f" / 충전소 {len(st_public)}곳 중 공공 {st_public.mean():.1%}")

# 시간대별 충전 이용 (충전 시작 시각 기준)
sessions = pd.read_csv(D.RAW / "charging_sessions.csv")
hour = pd.to_datetime(sessions["충전시작시각"]).dt.hour
hourly = hour.value_counts(normalize=True).sort_index().rename("비율")
print(f"\n[시간대별 이용] 12~17시 비중 {hourly.loc[12:17].sum():.1%}")

# 읍면동별 충전기 1대당 인구, 총 충전용량
pop = D.load_population()
by_dong = (chargers.groupby("읍면동")
           .agg(충전기수=("chgerId", "count"), 총충전용량=("output", "sum"))
           .reset_index())
by_dong = pop.merge(by_dong, on="읍면동", how="left").fillna({"충전기수": 0, "총충전용량": 0})
by_dong["충전기당인구"] = by_dong["인구수"] / by_dong["충전기수"].replace(0, pd.NA)
by_dong = by_dong.sort_values("충전기당인구", ascending=False)
print("\n[읍면동별 인프라]\n", by_dong.to_string(index=False))

by_dong.to_csv(OUT / "infra_by_dong.csv", index=False, encoding="utf-8-sig")
hourly.to_csv(OUT / "hourly_usage.csv", encoding="utf-8-sig")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].bar(supply["연도"].astype(str), supply["보급물량"], color="#4C72B0")
axes[0].set_title("연도별 전기차 보급 물량(대)")
axes[1].bar(hourly.index, hourly.values * 100, color=["#DD8452" if 12 <= h <= 17 else "#BBBBBB" for h in hourly.index])
axes[1].set_title("시간대별 충전 시작 비율(%)")
axes[1].set_xlabel("시")
plot = by_dong.dropna(subset=["충전기당인구"])
axes[2].barh(plot["읍면동"], plot["충전기당인구"], color="#55A868")
axes[2].invert_yaxis()
axes[2].set_title("충전기 1대당 인구(20세 이상)")
plt.tight_layout()
plt.savefig(OUT / "eda_summary.png", dpi=150)
