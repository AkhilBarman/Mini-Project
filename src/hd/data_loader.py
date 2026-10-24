"""
Data loader for MILDSum English legal judgment and summary pairs.

Loads, validates, and pairs English court judgments and expert-written
summaries. Handles missing files, empty texts, and character encoding issues.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def read_text_file(filepath: str) -> str:
    """Read a text file with robust encoding fallback.
    
    Tries utf-8, utf-8-sig, latin-1, and windows-1252 to ensure
    no file reading failure on various operating systems.
    
    Args:
        filepath: Absolute or relative path to the text file.
        
    Returns:
        The content of the file as a string.
    """
    encodings_to_try = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    for enc in encodings_to_try:
        try:
            with open(filepath, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.warning("Error opening %s with encoding %s: %s", filepath, enc, e)
            break
            
    # Final fallback with character replacement
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def load_mildsum_pairs(
    data_dir: str,
    judgment_filename: str = "EN_Judgment.txt",
    summary_filename: str = "EN_Summary.txt",
    min_char_length: int = 20,
    verbose: bool = True
) -> List[Dict[str, str]]:
    """Load and validate English judgment-summary pairs from raw data directory.

    Expects sample subdirectories inside data_dir (e.g. data_dir/Sample_1/),
    each containing EN_Judgment.txt and EN_Summary.txt. Also supports flat
    directory pairings if sample subdirectories are not used.

    Validates:
      - Presence of both judgment and summary files.
      - Non-empty content (at least min_char_length characters).
      - Mismatched or unparseable items.

    Args:
        data_dir: Path to directory containing raw data samples.
        judgment_filename: Expected name of judgment file in each sample folder.
        summary_filename: Expected name of summary file in each sample folder.
        min_char_length: Minimum character length to consider valid text.
        verbose: Whether to print summary of findings.

    Returns:
        List of dicts: [{'id': str, 'judgment_text': str, 'summary_text': str}, ...]
    """
    if not os.path.exists(data_dir):
        logger.warning("Data directory does not exist: %s", data_dir)
        if verbose:
            print(f"Data directory not found: {data_dir}")
        return []

    valid_pairs: List[Dict[str, str]] = []
    missing_items: List[str] = []
    empty_items: List[str] = []
    total_candidates = 0

    subdirs = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(".")
    ]

    # Mode 1: Subdirectories (MILDSum standard: Sample_1/, Sample_2/, etc.)
    if subdirs:
        # Sort subdirectories naturally (e.g. Sample_1, Sample_2, Sample_10)
        def _sort_key(s: str):
            parts = s.split("_")
            if len(parts) > 1 and parts[-1].isdigit():
                return (parts[0], int(parts[-1]))
            return (s, 0)

        subdirs = sorted(subdirs, key=_sort_key)
        total_candidates = len(subdirs)

        for sdir in subdirs:
            pair_id = sdir
            dir_path = os.path.join(data_dir, sdir)
            j_path = os.path.join(dir_path, judgment_filename)
            s_path = os.path.join(dir_path, summary_filename)

            # Check existence
            j_exists = os.path.isfile(j_path)
            s_exists = os.path.isfile(s_path)

            if not j_exists or not s_exists:
                missing_detail = []
                if not j_exists:
                    missing_detail.append(judgment_filename)
                if not s_exists:
                    missing_detail.append(summary_filename)
                missing_items.append(f"{pair_id} (missing: {', '.join(missing_detail)})")
                continue

            # Read text
            j_text = read_text_file(j_path).strip()
            s_text = read_text_file(s_path).strip()

            # Validate length
            if len(j_text) < min_char_length or len(s_text) < min_char_length:
                empty_detail = []
                if len(j_text) < min_char_length:
                    empty_detail.append(f"judgment ({len(j_text)} chars)")
                if len(s_text) < min_char_length:
                    empty_detail.append(f"summary ({len(s_text)} chars)")
                empty_items.append(f"{pair_id} (empty/too short: {', '.join(empty_detail)})")
                continue

            valid_pairs.append({
                "id": pair_id,
                "judgment_text": j_text,
                "summary_text": s_text
            })

    # Mode 2: Flat files fallback (e.g. id1_judgment.txt and id1_summary.txt)
    else:
        all_files = os.listdir(data_dir)
        j_files = {f.split("_judgment")[0]: f for f in all_files if "_judgment" in f.lower()}
        s_files = {f.split("_summary")[0]: f for f in all_files if "_summary" in f.lower()}
        all_ids = sorted(set(j_files.keys()).union(set(s_files.keys())))
        total_candidates = len(all_ids)

        for cid in all_ids:
            if cid not in j_files or cid not in s_files:
                missing_items.append(cid)
                continue
            j_text = read_text_file(os.path.join(data_dir, j_files[cid])).strip()
            s_text = read_text_file(os.path.join(data_dir, s_files[cid])).strip()
            if len(j_text) < min_char_length or len(s_text) < min_char_length:
                empty_items.append(cid)
                continue
            valid_pairs.append({
                "id": cid,
                "judgment_text": j_text,
                "summary_text": s_text
            })

    if verbose:
        print("=" * 60)
        print(f"MILDSum Data Loader Report")
        print("=" * 60)
        print(f"Source Directory:      {os.path.abspath(data_dir)}")
        print(f"Total Candidates:      {total_candidates}")
        print(f"Valid Matched Pairs:   {len(valid_pairs)}")
        print(f"Missing Items Count:   {len(missing_items)}")
        if missing_items:
            for m in missing_items[:5]:
                print(f"  - Missing: {m}")
            if len(missing_items) > 5:
                print(f"  ... and {len(missing_items) - 5} more")
        print(f"Empty/Short Items:     {len(empty_items)}")
        if empty_items:
            for e in empty_items[:5]:
                print(f"  - Empty: {e}")
            if len(empty_items) > 5:
                print(f"  ... and {len(empty_items) - 5} more")
        print("=" * 60)

    return valid_pairs
