"""03. 입지 점수: 7개 지표 Min-Max 정규화 후 AHP 가중합"""
import matplotlib.pyplot as plt
import pandas as pd

import data as D

OUT = D.out_dir("03_site_scoring")
D.set_korean_font()

table = D.score_table()
table.to_csv(OUT / "site_scores.csv", index=False, encoding="utf-8-sig")
print(f"공공 충전소 {len(table)}곳")
print(table.head(10)[["statNm", "읍면동", "총출력", "최종점수"]].round(3).to_string())

# 제출 당시 상위 10곳과 겹치는 곳
submitted = pd.read_csv(D.ROOT / "data" / "submitted" / "ess_result_submitted.csv")
base = lambda s: s.str.replace(r"\(\d+\)$", "", regex=True).str.strip()
overlap = set(base(table["statNm"].head(10))) & set(base(submitted["statNm"].head(10)))
print(f"\n제출 결과 상위 10곳과 겹치는 곳: {len(overlap)}곳 {sorted(overlap)}")

# 상위 10곳 지표 기여도
top = table.head(10).set_index("statNm")
contrib = pd.DataFrame({D.LABELS[c]: top[D.LABELS[c]] * w for c, w in D.WEIGHTS.items()})
ax = contrib.iloc[::-1].plot.barh(stacked=True, figsize=(10, 6), colormap="tab10")
ax.set_title("상위 10곳 지표별 점수 기여")
ax.set_xlabel("가중 점수")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(OUT / "top10_contribution.png", dpi=150)

# 위치별 점수
plt.figure(figsize=(7, 7))
sc = plt.scatter(table["lng"], table["lat"], c=table["최종점수"], cmap="YlOrRd", s=60, edgecolor="gray")
for _, r in table.head(3).iterrows():
    plt.annotate(r["statNm"], (r["lng"], r["lat"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
plt.colorbar(sc, label="최종점수")
plt.title("공공 충전소 입지 점수")
plt.tight_layout()
plt.savefig(OUT / "score_map.png", dpi=150)
