import os
import json
import time
import hashlib
import pandas as pd
import requests

# ── Schema contract ────────────────────────────────────────────────────────────
# These columns are required by data_cleaning.py.  The field 'inbound' marks
# which tweets come from customers (True) vs. brand (False); 'response_tweet_id'
# chains replies into conversation threads.  Both are structural to the TWCS
# dataset and cannot be synthetically derived from any other source.
REQUIRED_COLUMNS = ['tweet_id', 'author_id', 'inbound', 'text', 'response_tweet_id']

PRIMARY_SOURCE_URL = "https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv"
PRIMARY_SOURCE_NAME = "thoughtvector/customer-support-on-twitter (TWCS)"

# Sample size used when validating an already-existing file
_VALIDATE_SAMPLE_ROWS = 500


def _sha256_sample(path: str, sample_bytes: int = 65536) -> str:
    """SHA-256 of the first `sample_bytes` bytes — fast enough for large CSVs."""
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        h.update(fh.read(sample_bytes))
    return h.hexdigest()


def validate_twcs_schema(df: pd.DataFrame) -> None:
    """
    Validates that a DataFrame matches the TWCS schema contract.

    Raises KeyError if required columns are absent.
    Raises ValueError if semantic invariants fail:
      - 'inbound' must contain boolean-coercible values
      - 'tweet_id' must not be entirely null

    No column renaming is performed: this function is strict by design.
    A different source dataset requires its own dedicated adapter, not a
    generic rename map that silently produces structurally invalid data.
    """
    cols = set(df.columns)
    missing = [c for c in REQUIRED_COLUMNS if c not in cols]
    if missing:
        raise KeyError(
            f"[Schema Error] Missing required TWCS columns: {missing}. "
            f"Present columns: {sorted(cols)}. "
            "If you are using a non-TWCS dataset, write a dedicated source "
            "adapter — do NOT rename columns to pass this check."
        )

    # Semantic invariants
    if df['tweet_id'].isna().all():
        raise ValueError("[Schema Error] Column 'tweet_id' is entirely null.")

    inbound_vals = df['inbound'].dropna().unique()
    # TWCS encodes inbound as True/False strings or booleans
    allowed = {True, False, 'True', 'False', 1, 0, '1', '0'}
    bad = [v for v in inbound_vals if v not in allowed]
    if bad:
        raise ValueError(
            f"[Schema Error] Column 'inbound' contains unexpected values: {bad[:5]}. "
            "Expected boolean True/False."
        )

    print("[Data Ingest Schema] Validation PASSED — all required TWCS columns present.")


def _write_manifest(
    manifest_path: str,
    output_path: str,
    source_url: str,
    source_name: str,
    row_count: int,
    checksum: str,
) -> None:
    manifest = {
        'source_dataset': source_name,
        'download_url': source_url,
        'ingest_timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        'file_path': output_path,
        'file_size_bytes': os.path.getsize(output_path),
        'file_sha256_sample': checksum,
        'sample_row_count_validated': row_count,
        'schema_validated': True,
        'required_columns': REQUIRED_COLUMNS,
    }
    manifest_dir = os.path.dirname(manifest_path)
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(f"[Data Ingest] Data manifest saved to {manifest_path}.")


def verify_file_size_against_manifest(file_path: str, manifest_path: str, tolerance: float = 0.05) -> bool:
    """
    Verifies that the downloaded or existing file size matches the stored file_size_bytes in data_manifest.json
    within the specified relative tolerance (default: 5%).
    """
    if not os.path.exists(file_path) or not os.path.exists(manifest_path):
        return False
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest_data = json.load(f)
            expected_bytes = manifest_data.get('file_size_bytes')
            if expected_bytes and expected_bytes > 0:
                actual_bytes = os.path.getsize(file_path)
                rel_diff = abs(actual_bytes - expected_bytes) / expected_bytes
                if rel_diff > tolerance:
                    raise RuntimeError(
                        f"[Data Ingest Error] Downloaded file size ({actual_bytes:,} bytes) differs from "
                        f"manifest expected size ({expected_bytes:,} bytes) by {rel_diff:.2%} (tolerance: {tolerance:.2%})."
                    )
                print(f"[Data Ingest] File size verification PASSED ({actual_bytes:,} bytes matches manifest expected {expected_bytes:,} bytes within {tolerance:.2%} tolerance).")
                return True
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        pass
    return False


def download_twitter_support_data(
    output_path: str = "data/raw/twcs.csv",
    manifest_path: str = "data/raw/data_manifest.json",
) -> str:
    """
    Ensures twcs.csv is present, schema-valid, and recorded in a manifest.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ── Case A: file already exists — validate before trusting it ─────────────
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1_000_000:
        size = os.path.getsize(output_path)
        print(f"[Data Ingest] Existing file found at {output_path} ({size:,} bytes). "
              "Validating schema before use...")

        try:
            sample_df = pd.read_csv(output_path, nrows=_VALIDATE_SAMPLE_ROWS)
            validate_twcs_schema(sample_df)
        except Exception as e:
            raise RuntimeError(
                f"[Data Ingest] Existing file at '{output_path}' failed schema "
                f"validation: {e}\n"
                "Delete the file and re-run to trigger a fresh download."
            ) from e

        verify_file_size_against_manifest(output_path, manifest_path)
        checksum = _sha256_sample(output_path)
        _write_manifest(
            manifest_path, output_path,
            PRIMARY_SOURCE_URL, PRIMARY_SOURCE_NAME,
            len(sample_df), checksum,
        )
        print(f"[Data Ingest] Verified existing dataset at {output_path} ({size:,} bytes).")
        return output_path

    # ── Case B: file absent or too small — download from primary source ────────
    print(f"[Data Ingest] Downloading TWCS dataset from primary source...")
    print(f"  URL: {PRIMARY_SOURCE_URL}")

    try:
        response = requests.get(PRIMARY_SOURCE_URL, stream=True, timeout=60)
        response.raise_for_status()
        with open(output_path, 'wb') as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
        print(f"[Data Ingest] Download complete. Saved to {output_path} "
              f"({os.path.getsize(output_path):,} bytes).")
    except Exception as e:
        raise RuntimeError(
            f"[Data Ingest] Primary TWCS download failed: {e}\n\n"
            "No compatible fallback source is configured. Please download twcs.csv manually and place it at "
            f"'{output_path}', or fix network connectivity and retry."
        ) from e

    # ── Validate freshly downloaded file ──────────────────────────────────────
    print("[Data Ingest] Validating downloaded file schema...")
    sample_df = pd.read_csv(output_path, nrows=_VALIDATE_SAMPLE_ROWS)
    validate_twcs_schema(sample_df)
    verify_file_size_against_manifest(output_path, manifest_path)

    checksum = _sha256_sample(output_path)
    _write_manifest(
        manifest_path, output_path,
        PRIMARY_SOURCE_URL, PRIMARY_SOURCE_NAME,
        len(sample_df), checksum,
    )
    return output_path


if __name__ == "__main__":
    download_twitter_support_data()
