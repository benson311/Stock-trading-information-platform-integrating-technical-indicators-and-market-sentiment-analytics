import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""台灣交易日曆工具：民國年↔西元年轉換、交易日推導。"""
import datetime as dt
import pandas as pd


def roc_to_datetime(roc_str: str) -> dt.datetime:
    """將民國年字串 (如 '104/01/05') 轉為西元 datetime。

    民國年 = 西元年 - 1911 (民國元年 = 1912)。
    """
    parts = str(roc_str).strip().split("/")
    if len(parts) != 3:
        raise ValueError(f"無法解析民國日期: {roc_str}")
    roc_year = int(parts[0])
    month = int(parts[1])
    day = int(parts[2])
    year = roc_year + 1911
    return dt.datetime(year, month, day)


def datetime_to_roc(d: dt.date) -> str:
    """西元 date → 民國年字串 'YYY/MM/DD'。"""
    roc_year = d.year - 1911
    return f"{roc_year:03d}/{d.month:02d}/{d.day:02d}"


def build_trading_calendar(price_df: pd.DataFrame, date_col: str = "date"):
    """由價格資料推導台灣交易日曆 (僅交易日會出現在日線資料中)。

    回傳有序交易日 list[str]。
    """
    dates = sorted(pd.to_datetime(price_df[date_col]).dt.strftime("%Y-%m-%d").unique())
    return dates