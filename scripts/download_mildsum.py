"""
Download English judgments and summaries from the MILDSum benchmark dataset.

The official MILDSum repository (https://github.com/Law-AI/MILDSum) hosts 10
sample pairs directly in Data/MILDSum_Samples/. Each folder contains:
  - EN_Judgment.txt  (Original case judgment)
  - EN_Summary.txt   (Expert-written English summary)
  - HI_Summary.txt   (Hindi summary - skipped, as our project is English only)

For the full 3,122 judgment dataset, authors instruct to email:
debtanudatta04@gmail.com. Any additional folders placed into data/raw/
will automatically be picked up by the data loader.
"""

import argparse
import os
import sys
import urllib.request
import urllib.error


def download_mildsum_samples(output_dir: str, num_samples: int = 10) -> int:
    """Download available English samples from MILDSum repository.
    
    Args:
        output_dir: Target directory (typically data/raw).
        num_samples: Number of sample folders to download (default 10).
        
    Returns:
        Number of successfully downloaded/verified sample pairs.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_url = "https://raw.githubusercontent.com/Law-AI/MILDSum/main/Data/MILDSum_Samples"
    files_to_download = ["EN_Judgment.txt", "EN_Summary.txt"]
    
    successful = 0
    print(f"Downloading English MILDSum samples into: {output_dir}")
    print("Note: Only English judgments and summaries are downloaded (HI_Summary.txt is excluded).\n")

    for i in range(1, num_samples + 1):
        sample_name = f"Sample_{i}"
        sample_dir = os.path.join(output_dir, sample_name)
        os.makedirs(sample_dir, exist_ok=True)
        
        pair_complete = True
        for fname in files_to_download:
            dest_file = os.path.join(sample_dir, fname)
            if os.path.exists(dest_file) and os.path.getsize(dest_file) > 0:
                continue
                
            url = f"{base_url}/{sample_name}/{fname}"
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "HallucinationDetector/1.0 (BTech-Project)"}
                )
                with urllib.request.urlopen(req, timeout=30) as resp, open(dest_file, "wb") as f:
                    f.write(resp.read())
            except urllib.error.URLError as e:
                print(f"Warning: Failed to download {url}: {e}")
                pair_complete = False
            except Exception as e:
                print(f"Error saving {dest_file}: {e}")
                pair_complete = False

        if pair_complete:
            successful += 1
            print(f"  [OK] {sample_name}: EN_Judgment.txt & EN_Summary.txt verified.")

    print(f"\nDownload complete: {successful} sample pairs ready in {output_dir}.")
    print("If you have obtained the full MILDSum dataset from authors, place additional")
    print("sample folders (Sample_11, etc.) into data/raw/ and they will be recognized.")
    return successful


def main():
    parser = argparse.ArgumentParser(description="Download English MILDSum dataset samples.")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "raw"),
        help="Path to raw data directory (default: data/raw)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of samples to fetch from GitHub repository (default: 10)"
    )
    args = parser.parse_args()
    output_dir = os.path.abspath(args.output_dir)
    download_mildsum_samples(output_dir, num_samples=args.samples)


if __name__ == "__main__":
    main()
