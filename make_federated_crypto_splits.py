#!/usr/bin/env python3
import os, json, random
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

def assign_symbols_balanced(symbol_counts: dict, n_clients: int, seed: int = 42):
    """Жадное распределение символов по клиентам с выравниванием объёмов."""
    rng = random.Random(seed)
    symbols = list(symbol_counts.items())  # [(sym, count), ...]
    rng.shuffle(symbols)
    buckets = [[] for _ in range(n_clients)]
    loads   = [0  for _ in range(n_clients)]
    for sym, cnt in symbols:
        i = int(np.argmin(loads))
        buckets[i].append(sym)
        loads[i] += cnt
    assignment = {f"client_{i+1:02d}": buckets[i] for i in range(n_clients)}
    return assignment, loads

def main():
    ap = argparse.ArgumentParser("Make federated crypto splits (streaming, low-memory)")
    ap.add_argument("--unified", required=True, help="data/processed/unified_dataset.parquet")
    ap.add_argument("--out_dir", default="data/processed_federated")
    ap.add_argument("--n_clients", type=int, default=6)
    ap.add_argument("--strategy", choices=["by_symbol_balanced"], default="by_symbol_balanced")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_rows", type=int, default=200_000, help="стриминговый размер батча")
    args = ap.parse_args()

    src = Path(args.unified)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- PASS 1: лёгкий подсчёт объёмов по символу (читаем только 1 колонку) ----
    s = pd.read_parquet(src, columns=["symbol"])
    sym_counts = s["symbol"].value_counts(dropna=False).to_dict()

    # ---- назначение символов клиентам ----
    if args.strategy == "by_symbol_balanced":
        assignment, loads = assign_symbols_balanced(sym_counts, args.n_clients, seed=args.seed)
    else:
        raise NotImplementedError("Только by_symbol_balanced в этой версии")

    # запишем meta
    meta = {
        "strategy": args.strategy,
        "n_clients": args.n_clients,
        "symbol_counts": sym_counts,
        "assignment": assignment,
        "loads": {k:int(v) for k,v in zip(assignment.keys(), loads)}
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # ---- PASS 2: стримовое чтение и запись ----
    pf = pq.ParquetFile(src)
    full_schema = pf.schema_arrow  # общая схема исходного файла

    # Подготовим писатели (по одному parquet-файлу на клиента)
    writers = {}
    for cname in assignment.keys():
        cdir = out_dir / cname
        cdir.mkdir(parents=True, exist_ok=True)
        dest_path = cdir / "unified_dataset.parquet"
        # Важно: один файл на клиента, поэтому используем ParquetWriter
        writers[cname] = pq.ParquetWriter(dest_path.as_posix(), full_schema, compression="snappy")

    # Быстрый обратный индекс: symbol -> client_name
    sym_to_client = {}
    for cname, syms in assignment.items():
        for sym in syms:
            sym_to_client[sym] = cname

    # Стримим по батчам
    for batch in pf.iter_batches(batch_size=args.batch_rows):
        # batch: pyarrow.RecordBatch; конвертируем в pandas для простого фильтра по символам
        pdf = batch.to_pandas(types_mapper=pd.ArrowDtype)  # экономнее памяти
        # Разделяем строки батча по клиентам
        for sym, subdf in pdf.groupby("symbol", sort=False):
            cname = sym_to_client.get(sym, None)
            if cname is None:
                # На случай NaN/неожиданных символов — можно отправить в самого "лёгкого" клиента
                cname = min(assignment.keys(), key=lambda k: len(assignment[k]))
            if len(subdf) == 0:
                continue
            table = pa.Table.from_pandas(subdf, preserve_index=False, schema=full_schema)
            writers[cname].write_table(table)
        del pdf  # освободим память на батч

    # Закрываем писатели
    for w in writers.values():
        w.close()

    print(f"Saved federated splits to {out_dir}. Clients: {', '.join(assignment.keys())}")

if __name__ == "__main__":
    main()
