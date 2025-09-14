import numpy as np
import pandas as pd
from typing import Iterable, List, Sequence, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# =========================
#  Transformer
# =========================
class GroupTimeLagRoller(BaseEstimator, TransformerMixin):
    """
    依 group_key 與 time_col 排序，在各 group 內對指定欄位 cols 產生多個 lag 與 rolling 特徵。
    - transform 再次檢查/強制排序（穩定、無洩漏）
    - rolling 先 shift(shift_for_rolling) 再 rolling，避免洩漏
    - 保留 NaN（不改變列數）；由後續 Imputer/Dropper 處理
    - fit 記住每組尾端歷史（所需上下文），transform 接上後再算
    - 正確回傳 feature_names_out（配合 sklearn set_output='pandas'）
    """

    _VALID_FUNCS = {"mean", "std", "max", "min"}

    def __init__(
        self,
        group_key: Union[str, Sequence[str]],
        time_col: str,
        cols: Sequence[str],
        lags: Sequence[int] = (1,),
        roll_windows: Sequence[int] = (3, 6),
        roll_funcs: Sequence[str] = ("mean",),
        min_periods: Optional[int] = None,
        ascending: bool = True,
        shift_for_rolling: int = 1,
        keep_original_order: bool = True,
        return_only_new_features: bool = False,
        drop_original_y: bool = False,
    ):
        # __init__ 只存參數，不能改型（為了 sklearn clone 相容）
        self.group_key = group_key
        self.time_col = time_col
        self.cols = cols
        self.lags = lags
        self.roll_windows = roll_windows
        self.roll_funcs = roll_funcs
        self.min_periods = min_periods
        self.ascending = ascending
        self.shift_for_rolling = shift_for_rolling
        self.keep_original_order = keep_original_order
        self.return_only_new_features = return_only_new_features
        self.drop_original_y = drop_original_y

    # ---------- private helpers ----------
    def _as_list(self, x):
        return x if isinstance(x, (list, tuple)) else [x]

    def _context_len(self):
        max_lag = max(self._lags_, default=0)
        max_roll = self.shift_for_rolling + (max(self._roll_windows_, default=0) or 0) - 1
        return max(max_lag, max_roll, 0)

    # ---------- public helpers ----------
    def get_column_names(self, cols: Optional[Sequence[str]] = None) -> List[str]:
        """不需 fit；回傳依目前參數會產生的 lag/rolling 欄名（僅新特徵名）。"""
        cols = list(self.cols if cols is None else cols)
        lags = list(self.lags)
        windows = list(self.roll_windows)
        funcs = list(self.roll_funcs)
        mp = (lambda w: w) if self.min_periods is None else (lambda w: self.min_periods)
        names = [f"{c}_lag{k}" for c in cols for k in lags]
        names += [
            f"{c}_roll{fn}{w}_s{self.shift_for_rolling}_mp{mp(w)}"
            for c in cols for w in windows for fn in funcs
        ]
        return names

    def get_feature_names_out(self, input_features=None):
        """sklearn set_output='pandas' 會用到；長度必須等於 transform 的實際欄數。"""
        check_is_fitted(self, ["_feature_names_out_"])
        return np.array(self._feature_names_out_, dtype=object)

    # ---------- sklearn hooks ----------
    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()

        # 正規化參數（放 fit，不放 __init__，確保可 clone）
        self._gcols_ = self._as_list(self.group_key)
        self._cols_ = list(self.cols)
        self._lags_ = list(self.lags)
        self._roll_windows_ = list(self.roll_windows)
        self._roll_funcs_ = list(self.roll_funcs)

        # 檢查欄位
        missing = [c for c in self._gcols_ + [self.time_col] + self._cols_ if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")

        t = pd.to_datetime(df[self.time_col], errors="coerce")
        if t.isna().any():
            raise ValueError(f"time_col '{self.time_col}' contains unparsable values")

        # 新特徵名
        mp = (lambda w: w) if self.min_periods is None else (lambda w: self.min_periods)
        self._generated_feature_names_ = (
            [f"{c}_lag{k}" for c in self._cols_ for k in self._lags_]
            + [
                f"{c}_roll{fn}{w}_s{self.shift_for_rolling}_mp{mp(w)}"
                for c in self._cols_ for w in self._roll_windows_ for fn in self._roll_funcs_
            ]
        )

        # 保存輸入欄與最終輸出欄名（依參數決定）
        self._feature_names_in_ = list(df.columns)
        if self.return_only_new_features:
            self._feature_names_out_ = list(self._generated_feature_names_)
        else:
            base = list(self._feature_names_in_)
            if self.drop_original_y:
                base = [c for c in base if c not in self._cols_]
            self._feature_names_out_ = base + list(self._generated_feature_names_)

        # 記錄歷史尾巴（避免在 val/test 看不到過去）
        need = self._context_len()
        self._history_ = {}
        if need > 0:
            df_sorted = df.assign(_tmp_time_=t).sort_values(
                self._gcols_ + ["_tmp_time_"], ascending=self.ascending, kind="mergesort"
            )
            for key, part in df_sorted.groupby(self._gcols_, sort=False):
                key = tuple(np.atleast_1d(key))
                self._history_[key] = part.tail(need)[self._gcols_ + [self.time_col] + self._cols_].copy()

        self.n_features_in_ = df.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["_generated_feature_names_", "_feature_names_out_", "_history_"])
        df = X.copy()
        original_index = df.index

        t_cur = pd.to_datetime(df[self.time_col], errors="raise")
        df_cur = df.assign(_tmp_time_=t_cur)

        # 接歷史（逐組）
        if self._history_:
            parts = []
            for key, part in df_cur.groupby(self._gcols_, sort=False):
                key = tuple(np.atleast_1d(key))
                hist = self._history_.get(key)
                if hist is not None and not hist.empty:
                    h = hist.copy()
                    h["_tmp_time_"] = pd.to_datetime(h[self.time_col], errors="raise")
                    parts.append(pd.concat([h, part], axis=0, sort=False))
                else:
                    parts.append(part)
            df_work = pd.concat(parts, axis=0, sort=False)
        else:
            df_work = df_cur

        # 排序
        df_work = df_work.sort_values(
            self._gcols_ + ["_tmp_time_"], ascending=self.ascending, kind="mergesort"
        )

        # 逐欄造特徵
        g = df_work.groupby(self._gcols_, sort=False)
        for c in self._cols_:
            # lags
            for k in self._lags_:
                df_work[f"{c}_lag{k}"] = g[c].shift(k)
            # rolling（先 shift 再 rolling）
            shifted = g[c].shift(self.shift_for_rolling)
            for w in self._roll_windows_:
                mp = w if self.min_periods is None else self.min_periods
                roll = shifted.rolling(window=w, min_periods=mp)
                for fn in self._roll_funcs_:
                    df_work[f"{c}_roll{fn}{w}_s{self.shift_for_rolling}_mp{mp}"] = getattr(roll, fn)()

        # 只回傳本次 X 的列；去重 index（保留最後一筆＝當期）
        df_out = df_work.loc[df_cur.index]
        df_out = df_out[~df_out.index.duplicated(keep="last")]
        df_out = df_out.reindex(df_cur.index)

        # 清理
        if self.drop_original_y and not self.return_only_new_features:
            df_out = df_out.drop(columns=self._cols_, errors="ignore")
        df_out = df_out.drop(columns=["_tmp_time_"], errors="ignore")
        if self.keep_original_order:
            df_out = df_out.loc[original_index]

        # 依選項決定輸出
        return (
            df_out[self._generated_feature_names_]
            if self.return_only_new_features
            else df_out[self._feature_names_out_]
        )