#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Federated split: by_row_hash_synced (IID, simultaneous clients)
- Каждая строка назначается клиенту по детерминированному хешу (symbol, timestamp[, salt]).
- Объединение всех клиентов == исходный датасет (без дублей).
- Двухпроходный стриминговый режим (низкая память): фактически нужен только один проход,
  но второй удобен, если захочешь сначала проверить наличие колонок и схему.
"""

import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def row_to_client_idx(symbol: str, ts, n_clients: int, salt: str = "") -> int:
    # ts может быть pandas.Timestamp или строкой — приводим к строке
    ts_str = str(ts)
    s = f"{symbol}|{ts_str}|{salt}"
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest()
    return int(h, 16) % n_clients

def main():
    ap = argparse.ArgumentParser("Make IID federated splits by row hash (low-memory)")
    ap.add_argument("--unified", required=True, help="data/processed/unified_dataset.parquet")
    ap.add_argument("--out_dir", default="data/processed_federated_iid")
    ap.add_argument("--n_clients", type=int, default=6)
    ap.add_argument("--salt", default="", help="опц. соль в хеш для воспроизводимого, но отличного деления")
    ap.add_argument("--batch_rows", type=int, default=200_000)
    ap.add_argument("--timestamp_col", default="timestamp")
    ap.add_argument("--symbol_col", default="symbol")
    args = ap.parse_args()

    src = Path(args.unified)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(src)
    full_schema = pf.schema_arrow

    # Подготовим писатели на клиентов (один файл на клиента)
    writers = {}
    for i in range(args.n_clients):
        cname = f"client_{i+1:02d}"
        cdir = out_dir / cname
        cdir.mkdir(parents=True, exist_ok=True)
        writers[cname] = pq.ParquetWriter((cdir / "unified_dataset.parquet").as_posix(),
                                          full_schema, compression="snappy")

    # Статистика
    counts = {f"client_{i+1:02d}": 0 for i in range(args.n_clients)}

    # Стрим: читаем батчи, назначаем строки клиентам
    for batch in pf.iter_batches(batch_size=args.batch_rows):
        df = batch.to_pandas(types_mapper=pd.ArrowDtype)

        # Нормализуем время к строке (детерминированность)
        if args.timestamp_col in df.columns:
            df[args.timestamp_col] = pd.to_datetime(df[args.timestamp_col], utc=True, errors="coerce")
        else:
            raise ValueError(f"Не найден столбец времени: {args.timestamp_col}")

        if args.symbol_col not in df.columns:
            raise ValueError(f"Не найден столбец символа: {args.symbol_col}")

        # Вычислим индексы клиентов векторно
        # Для скорости — соберём ключи один раз
        # Нормализуем время: UTC-aware -> UTC-naive (без tz), затем в строку ns-precision
        ts = pd.to_datetime(df[args.timestamp_col], utc=True, errors="coerce")
        ts_naive = ts.dt.tz_convert("UTC").dt.tz_localize(None)  # убрали tz
        ts_str = ts_naive.astype("datetime64[ns]").astype(str)   # теперь безопасно

        # Ключ для детерминированного хеш-разбиения
        keys = (df[args.symbol_col].astype(str) + "|" + ts_str + "|" + args.salt)

        # Векторный blake2b в чистом Python нет, поэтому делаем группами
        idxs = np.empty(len(df), dtype=np.int32)
        chunk = 50000
        for i in range(0, len(df), chunk):
            sl = keys.iloc[i:i+chunk]
            hashed = [int(hashlib.blake2b(k.encode("utf-8"), digest_size=8).hexdigest(), 16) for k in sl]
            idxs[i:i+chunk] = np.array(hashed, dtype=np.uint64) % args.n_clients

        df["_client_idx"] = idxs

        # Записываем разбиение
        for ci, g in df.groupby("_client_idx", sort=False):
            cname = f"client_{int(ci)+1:02d}"
            tbl = pa.Table.from_pandas(g.drop(columns=["_client_idx"]), preserve_index=False, schema=full_schema)
            writers[cname].write_table(tbl)
            counts[cname] += len(g)

        del df

    for w in writers.values():
        w.close()

    meta = {
        "strategy": "by_row_hash_synced",
        "n_clients": args.n_clients,
        "salt": args.salt,
        "timestamp_col": args.timestamp_col,
        "symbol_col": args.symbol_col,
        "client_rows": counts
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved IID splits to {out_dir}. Rows per client: {counts}")

if __name__ == "__main__":
    main()
