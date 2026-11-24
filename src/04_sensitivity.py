"""04. 결과 검증: 가중치 민감도, OLS 회귀 검토 (대회 이후 추가)"""
import numpy as np
import pandas as pd

import data as D

OUT = D.out_dir("04_sensitivity")
SEED = 42


def ols(X, y):
    """최소제곱 회귀: 계수, R²"""
    A = np.c_[np.ones(len(X)), X.astype(float).values]
    beta = np.linalg.lstsq(A, y.values, rcond=None)[0]
    r2 = 1 - ((y.values - A @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return beta[1:], r2


# 1) 가중치 민감도: 기준 가중치 중심 디리클레 분포 1,000회
st = D.stations(D.load_chargers())
ind = D.build_indicators(st, D.load_population(), D.load_weather())
norm, y = D.ahp_score(ind)
w0 = np.array(list(D.WEIGHTS.values()))
draws = np.random.default_rng(SEED).dirichlet(w0 * 50, size=1000)
scores = norm[list(D.WEIGHTS)].values @ draws.T
top3 = np.argsort(-scores, axis=0)[:3]
freq = pd.Series(np.bincount(top3.ravel(), minlength=len(ind)) / draws.shape[0], index=ind["statNm"])
freq = freq[freq > 0].sort_values(ascending=False)
print("\n[가중치 민감도] 상위 3위 진입 비율\n", freq.round(3).to_string())
freq.rename("top3_비율").to_csv(OUT / "weight_sensitivity_top3.csv", encoding="utf-8-sig")

# 모든 후보지에 같은 점수인 지표 → 순위 영향 없음
constant = [c for c in D.WEIGHTS if norm[c].nunique() == 1]
print(f"\n[상수 지표] {[D.LABELS[c] for c in constant]}, 가중치 합 {sum(D.WEIGHTS[c] for c in constant):.2f}")

# 2) OLS 회귀 검토: 점수를 같은 지표로 회귀 → 가중치 재추정에 불과
varying = [c for c in D.WEIGHTS if c not in constant]
raw_coef, raw_r2 = ols(ind[varying], y)
norm_coef, norm_r2 = ols(norm[varying], y)
table = pd.DataFrame({"원본단위_계수": raw_coef, "정규화_계수": norm_coef,
                      "AHP_가중치": [D.WEIGHTS[c] for c in varying]},
                     index=[D.LABELS[c] for c in varying])
print(f"\n[OLS] 원본 단위 R² {raw_r2:.4f}, 정규화 R² {norm_r2:.4f}")
print(table.to_string(float_format=lambda v: f"{v:.3g}"))
table.to_csv(OUT / "ols_check.csv", encoding="utf-8-sig")
