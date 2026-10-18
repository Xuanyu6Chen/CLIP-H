# Mortality and discharge (optional): builds the filtered lab-events cache that run_mortality.py and run_discharge.py read, faster than they build it themselves.
"""
Build the filtered lab-events parquet cache for the mortality and discharge cleaning (optional).

``load_mimic_tables`` (``../src_2/mimic_general.py``) reads lab events from
``mimic-iv-3.1/_clip_h_cache/labevents_<itemids>.parquet`` when that file exists. Otherwise it
filters ``hosp/labevents.csv.gz`` with pandas on first load and writes the same file. This script
builds the cache ahead of time and faster, using PyArrow's streaming CSV reader (multi-threaded,
C++ backend). It keeps only the rows whose itemid is in ``ITEMIDS`` and the columns in ``COLS``,
and shows a tqdm progress bar of compressed bytes read against the file size, with running counts
of rows scanned and kept.

Usage:
    python run/preprocess_labevents.py                      # reads ../mimic-iv-3.1/hosp
    python run/preprocess_labevents.py --mimic-path <MIMIC-IV hosp/ folder>

``--mimic-path`` is the MIMIC-IV ``hosp/`` folder that contains ``labevents.csv.gz``
(default: ``mimic-iv-3.1/hosp`` in this folder's parent, where the run scripts read it). The cache is written to ``_clip_h_cache/`` in the parent of that folder.
If the cache file already exists, the script exits without rebuilding it; delete the file to
rebuild. Requires pyarrow and tqdm.
"""
import argparse
import gzip
import sys
import time
from pathlib import Path

# Must match the values of LAB_ITEMIDS in ../src_2/mimic_general.py. Sorted, because the cache
# filename lists the itemids in sorted order.
ITEMIDS = sorted([50862, 50912, 51222, 50983, 51006, 50931, 51301, 51265])
CACHE_NAME = "labevents_" + "_".join(str(i) for i in ITEMIDS) + ".parquet"
COLS = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"]


class _BytesProgress:
    """
    Thin wrapper around a raw binary file that advances a tqdm bar on every
    read. gzip.GzipFile reads through this wrapper, so the bar counts
    compressed bytes consumed, which is directly comparable to the .gz file
    size.
    """
    def __init__(self, path, pbar):
        self._f   = open(path, "rb")
        self._pbar = pbar

    def read(self, n=-1):
        data = self._f.read(n)
        if data:
            self._pbar.update(len(data))
        return data

    def readinto(self, b):
        n = self._f.readinto(b)
        if n:
            self._pbar.update(n)
        return n

    def tell(self):   return self._f.tell()
    def seek(self, *a): return self._f.seek(*a)
    def close(self):  self._f.close()

    def __enter__(self): return self
    def __exit__(self, *a): self._f.close()


def main():
    ap = argparse.ArgumentParser(
        description="Pre-build labevents parquet cache for CLIP-H (fast, streaming).")
    ap.add_argument("--mimic-path", default=str(Path(__file__).resolve().parents[1] / "mimic-iv-3.1" / "hosp"),
                    help="Path to MIMIC-IV hosp/ folder (contains labevents.csv.gz)")
    args = ap.parse_args()

    hosp      = Path(args.mimic_path)
    lab_gz    = hosp / "labevents.csv.gz"
    cache_dir = hosp.parent / "_clip_h_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path  = cache_dir / CACHE_NAME

    if not lab_gz.exists():
        print(f"ERROR: not found: {lab_gz}")
        sys.exit(1)

    if out_path.exists():
        print(f"Cache already exists, nothing to do:\n  {out_path}")
        print("Delete it first if you want to rebuild.")
        return

    try:
        import pyarrow as pa
        import pyarrow.csv as pacsv
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError:
        print("ERROR: pyarrow is not installed. Run: pip install pyarrow")
        sys.exit(1)

    try:
        from tqdm import tqdm
    except ImportError:
        print("ERROR: tqdm is not installed. Run: pip install tqdm")
        sys.exit(1)

    file_size_bytes = lab_gz.stat().st_size
    file_size_gb    = file_size_bytes / 1e9
    print(f"\nlabevents.csv.gz   {file_size_gb:.2f} GB compressed")
    print(f"Output parquet     {out_path}\n")

    itemid_set   = pa.array(ITEMIDS, type=pa.int64())
    convert_opts = pacsv.ConvertOptions(include_columns=COLS)

    t0         = time.time()
    rows_read  = 0
    rows_kept  = 0
    writer     = None

    pbar = tqdm(
        total=file_size_bytes,
        unit="B", unit_scale=True, unit_divisor=1024,
        desc="labevents.csv.gz",
        dynamic_ncols=True,
        smoothing=0.05,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
    )

    try:
        with _BytesProgress(lab_gz, pbar) as bp:
            with gzip.GzipFile(fileobj=bp) as gz_f:
                reader = pacsv.open_csv(gz_f, convert_options=convert_opts)
                for batch in reader:
                    rows_read += len(batch)

                    mask     = pc.is_in(batch.column("itemid"), value_set=itemid_set)
                    filtered = batch.filter(mask)
                    rows_kept += len(filtered)

                    # Live row counts in the tqdm postfix
                    pbar.set_postfix(
                        rows_read=f"{rows_read:,}",
                        kept=f"{rows_kept:,}",
                        refresh=False,
                    )

                    if len(filtered) > 0:
                        if writer is None:
                            writer = pq.ParquetWriter(
                                str(out_path), filtered.schema, compression="snappy"
                            )
                        writer.write_batch(filtered)

    finally:
        pbar.close()
        if writer:
            writer.close()

    elapsed  = time.time() - t0
    size_mb  = out_path.stat().st_size / 1e6
    print(f"\n{'='*60}")
    print(f"  Done in          {elapsed/60:.1f} min")
    print(f"  Rows scanned     {rows_read:,}")
    print(f"  Rows kept        {rows_kept:,}")
    print(f"  Parquet size     {size_mb:.1f} MB  ({out_path})")
    print(f"{'='*60}")
    print("\nrun_mortality.py and run_discharge.py now read lab events from this cache when they load the MIMIC-IV tables.")


if __name__ == "__main__":
    main()
