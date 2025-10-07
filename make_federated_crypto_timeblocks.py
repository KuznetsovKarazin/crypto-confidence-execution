#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Federated split: by_time_blocks_per_symbol

У каждого клиента — ВСЕ символы, но непересекающиеся временные блоки по КАЖДОМУ символу.
- Пересечений нет; объединение клиентских датасетов == исходный parquet (за вычетом опц. "gap" строк).
- Временные границы считаются равномерно по ВРЕМЕНИ (а не по количеству строк): min_ts..max_ts делится на N-частей.
- Вокруг границ можно задать "санитарный зазор" (--gap_minutes), чтобы избежать утечки при оконных фичах.
  Строки, попавшие в "зазор", опционально выкидываются (по умолчанию — да).

Режим низкой памяти:
- PASS1: стримим parquet, собираем per-symbol {min_ts_ns, max_ts_ns, count}.
- PASS2: снова стримим parquet, каждому батчу/символу присваиваем client_idx по бин-позиции.

Таймзона:
- timestamp нормализуется в UTC-aware, затем в epoch_ns (int64) для сравнения — без конфликтов tz-aware/naive.

Выход:
- data/processed_federated_timeblocks/client_XX/unified_dataset.parquet (Snappy)
- meta.json (границы и общая статистика)

Пример:
python make_federated_crypto_timeblocks.py \
  --unified data/processed/unified_dataset.parquet \
  --out_dir data/processed_federated_timeblocks \
  --n_clients 7 \
  --gap_minutes 5 \
  --batch_rows 200000
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# -----------------------------
# Helpers
# -----------------------------

def to_epoch_ns(series: pd.Series) -> pd.Series:
    """UTC-aware -> UTC-naive -> int64 ns since epoch."""
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    ts_naive = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return pd.Series(ts_naive.values.astype("datetime64[ns]").astype("int64"), index=series.index)


def compute_time_boundaries(
    pf: pq.ParquetFile,
    symbol_col: str,
    timestamp_col: str,
    n_clients: int,
    batch_rows: int,
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, List[int]]]:
    """
    PASS1: вернёт словари по символам:
      min_ns[sym], max_ns[sym], count[sym], cuts_ns[sym] (внутренние границы длиной n_clients-1)
    """
    min_ns: Dict[str, int] = {}
    max_ns: Dict[str, int] = {}
    count:  Dict[str, int] = {}

    for batch in pf.iter_batches(batch_size=batch_rows, columns=[symbol_col, timestamp_col]):
        df = batch.to_pandas()
        if symbol_col not in df.columns or timestamp_col not in df.columns:
            raise ValueError(f"Нет колонок {symbol_col} / {timestamp_col} в parquet батче")

        # epoch ns
        ts_ns = to_epoch_ns(df[timestamp_col])
        syms = df[symbol_col].astype(str)

        # На всякий случай гарантируем одинаковую длину
        assert len(ts_ns) == len(syms)

        # Группируем в pandas (в батче)
        for sym, g in pd.DataFrame({"sym": syms, "ts": ts_ns}).groupby("sym", sort=False):
            if len(g) == 0:
                continue
            g_min = int(np.nanmin(g["ts"].values))
            g_max = int(np.nanmax(g["ts"].values))
            g_cnt = int(g.shape[0])

            if sym not in min_ns:
                min_ns[sym] = g_min
                max_ns[sym] = g_max
                count[sym]  = g_cnt
            else:
                if g_min < min_ns[sym]:
                    min_ns[sym] = g_min
                if g_max > max_ns[sym]:
                    max_ns[sym] = g_max
                count[sym] += g_cnt

        del df

    # Рассчёт внутренних границ (n_clients-1) равномерно по времени
    cuts_ns: Dict[str, List[int]] = {}
    for sym in min_ns.keys():
        lo = min_ns[sym]
        hi = max_ns[sym]
        if lo >= hi or n_clients <= 1:
            cuts_ns[sym] = []
            continue
        # равномерные временные узлы (исключая крайние)
        # np.linspace выдает float, переводим в int64
        inner = np.linspace(lo, hi, num=n_clients+1, dtype=np.float64)[1:-1]
        cuts_ns[sym] = [int(x) for x in inner]

    return min_ns, max_ns, count, cuts_ns


def assign_clients_for_symbol_ts(
    ts_ns: np.ndarray,
    cuts_ns: np.ndarray,
    gap_ns: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Возвращает:
      client_idx: int32[ N ] — 0..K-1 (куда пишем)
      keep_mask:  bool[ N ] — True для строк, которые НЕ попали в "gap" вокруг границ

    Логика:
      idx = searchsorted(cuts, t, side='right') ∈ [0..K-1]
      Если gap_ns > 0, то строки в окрестности любой границы (±gap) будут помечены на drop.
      Интервалы закрыто-открытые, пересечений между клиентами нет.
    """
    if cuts_ns.size == 0:
        # один клиент: весь диапазон
        idx = np.zeros_like(ts_ns, dtype=np.int32)
        keep = np.ones_like(ts_ns, dtype=bool)
        return idx, keep

    # Индекс клиента по позиции относительно cuts
    idx = np.searchsorted(cuts_ns, ts_ns, side="right").astype(np.int32)

    if gap_ns <= 0:
        keep = np.ones_like(ts_ns, dtype=bool)
        return idx, keep

    # Ближайшая граница: сравним расстояния до cuts[j-1] и cuts[j]
    j = np.searchsorted(cuts_ns, ts_ns, side="right")  # 0..len(cuts)
    # расстояние до левой границы (если есть)
    dist_left = np.full_like(ts_ns, fill_value=np.iinfo(np.int64).max, dtype=np.int64)
    mask_has_left = j > 0
    dist_left[mask_has_left] = ts_ns[mask_has_left] - cuts_ns[j[mask_has_left] - 1]
    # расстояние до правой границы (если есть)
    dist_right = np.full_like(ts_ns, fill_value=np.iinfo(np.int64).max, dtype=np.int64)
    mask_has_right = j < cuts_ns.size
    dist_right[mask_has_right] = cuts_ns[j[mask_has_right]] - ts_ns[mask_has_right]

    nearest = np.minimum(dist_left, dist_right)
    keep = nearest >= gap_ns
    return idx, keep


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser("Make federated splits by time-blocks per symbol")
    ap.add_argument("--unified", required=True, help="data/processed/unified_dataset.parquet")
    ap.add_argument("--out_dir", default="data/processed_federated_timeblocks")
    ap.add_argument("--n_clients", type=int, default=6)
    ap.add_argument("--batch_rows", type=int, default=200_000)
    ap.add_argument("--symbol_col", default="symbol")
    ap.add_argument("--timestamp_col", default="timestamp")
    ap.add_argument("--gap_minutes", type=float, default=0.0,
                    help="Зазор вокруг временных границ (минуты). Строки в зазоре будут отброшены.")
    ap.add_argument("--drop_gap_rows", action="store_true", default=True,
                    help="Если включено, строки в зоне gap отбрасываются (по умолчанию включено).")
    args = ap.parse_args()

    src = Path(args.unified)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(src)
    full_schema = pf.schema_arrow

    print("[Pass1] Computing per-symbol time boundaries...")
    min_ns, max_ns, counts, cuts_map = compute_time_boundaries(
        pf, args.symbol_col, args.timestamp_col, args.n_clients, args.batch_rows
    )
    symbols = sorted(min_ns.keys())
    print(f"[Pass1] Symbols: {len(symbols)}")

    # Подготовка писателей (по одному файлу на клиента)
    writers = {}
    for i in range(args.n_clients):
        cname = f"client_{i+1:02d}"
        cdir = out_dir / cname
        cdir.mkdir(parents=True, exist_ok=True)
        writers[cname] = pq.ParquetWriter(
            (cdir / "unified_dataset.parquet").as_posix(),
            full_schema,
            compression="snappy"
        )

    # Глобальная статистика
    written_rows = {f"client_{i+1:02d}": 0 for i in range(args.n_clients)}
    dropped_gap_rows = 0
    total_rows = 0

    # Преобразуем cuts к np.ndarray для быстрого поиска
    cuts_np: Dict[str, np.ndarray] = {
        sym: np.array(cuts_map[sym], dtype=np.int64) if cuts_map[sym] else np.array([], dtype=np.int64)
        for sym in symbols
    }
    gap_ns = int(args.gap_minutes * 60.0 * 1e9) if args.gap_minutes and args.gap_minutes > 0 else 0

    print("[Pass2] Streaming and writing rows to clients...")
    for batch in pf.iter_batches(batch_size=args.batch_rows):
        df = batch.to_pandas()
        if args.symbol_col not in df.columns or args.timestamp_col not in df.columns:
            raise ValueError(f"Нет колонок {args.symbol_col} / {args.timestamp_col} в parquet батче")

        # Нормализуем время
        ts_ns = to_epoch_ns(df[args.timestamp_col])
        df["_ts_ns"] = ts_ns
        df["_sym"] = df[args.symbol_col].astype(str)

        # Группируем в батче по символу для локального назначения
        for sym, g in df.groupby("_sym", sort=False):
            tvals = g["_ts_ns"].values.astype(np.int64)
            cuts = cuts_np.get(sym, np.array([], dtype=np.int64))

            idxs, keep = assign_clients_for_symbol_ts(tvals, cuts, gap_ns=gap_ns)

            if args.drop_gap_rows and keep.sum() < keep.size:
                dropped_gap_rows += int((~keep).sum())

            # Оставляем только строки, не попавшие в gap
            g2 = g.loc[keep]
            idxs2 = idxs[keep]

            # Пишем по клиентам
            if len(g2) > 0:
                for ci, gg in g2.groupby(idxs2, sort=False):
                    cname = f"client_{int(ci)+1:02d}"
                    tbl = pa.Table.from_pandas(
                        gg.drop(columns=["_ts_ns", "_sym"]), preserve_index=False, schema=full_schema
                    )
                    writers[cname].write_table(tbl)
                    written_rows[cname] += len(gg)

            total_rows += len(g)

        del df

    # Закрываем писателей
    for w in writers.values():
        w.close()

    # Meta: аккуратные границы в ISO и в ns
    meta = {
        "strategy": "by_time_blocks_per_symbol",
        "n_clients": args.n_clients,
        "gap_minutes": args.gap_minutes,
        "drop_gap_rows": args.drop_gap_rows,
        "symbol_col": args.symbol_col,
        "timestamp_col": args.timestamp_col,
        "batch_rows": args.batch_rows,
        "source": str(src),
        "clients": written_rows,
        "dropped_gap_rows": int(dropped_gap_rows),
        "per_symbol": {}
    }

    # Для читаемости сохраняем и ns, и ISO-формат внутренних границ
    for sym in symbols:
        cuts_ns_list = cuts_map[sym]
        cuts_iso = [
            pd.to_datetime(int(ns), unit="ns", utc=True).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            for ns in cuts_ns_list
        ]
        meta["per_symbol"][sym] = {
            "min_ts_ns": int(min_ns[sym]),
            "max_ts_ns": int(max_ns[sym]),
            "min_ts_iso": pd.to_datetime(int(min_ns[sym]), unit="ns", utc=True).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "max_ts_iso": pd.to_datetime(int(max_ns[sym]), unit="ns", utc=True).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "rows": int(counts[sym]),
            "inner_cuts_ns": cuts_ns_list,
            "inner_cuts_iso": cuts_iso
        }

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("[Done]")
    print(f"  Total source rows (seen in pass2 batches): {total_rows:,}")
    print(f"  Dropped gap rows: {dropped_gap_rows:,}")
    print("  Rows per client:", {k: int(v) for k, v in written_rows.items()})
    print(f"  Meta saved to: {out_dir / 'meta.json'}")
    

if __name__ == "__main__":
    main()
