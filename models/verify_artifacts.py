"""Verify the released model archive or its extracted files, without loading models."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "models/ARTIFACT_SHA256SUMS.txt"


def digest(stream):
    checksum = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        checksum.update(block)
    return checksum.hexdigest()


def indexed_files(index):
    expected = {}
    for uid, record in index["models"].items():
        canonical = index["models"][record["canonical_artifact_id"]]
        for field in ("model", "scalers"):
            name = record[field]
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or path.parts[0] != "models":
                raise ValueError("Unsafe artifact path: " + name)
            checksum = record[field + "_sha256"]
            if name != canonical[field] or checksum != canonical[field + "_sha256"]:
                raise ValueError("Canonical mapping differs: " + uid)
            if name in expected and expected[name] != checksum:
                raise ValueError("Conflicting hashes: " + name)
            expected[name] = checksum
    return expected


def parse_manifest(text):
    expected = {}
    for line in text.splitlines():
        checksum, name = line.split("  ", 1)
        if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError("Invalid SHA-256 manifest")
        if name in expected:
            raise ValueError("Duplicate manifest path: " + name)
        expected[name] = checksum
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Check the downloaded ZIP before extraction")
    args = parser.parse_args()
    index = json.loads((ROOT / "models/index.json").read_text(encoding="utf-8"))
    release = json.loads((ROOT / "models/release.json").read_text(encoding="utf-8"))
    expected = indexed_files(index)
    manifest_bytes = (ROOT / MANIFEST).read_bytes()
    if parse_manifest(manifest_bytes.decode("utf-8")) != expected:
        raise ValueError("Artifact manifest differs from the logical model index")
    if len(index["models"]) != release["logical_fits"] or len(index["binary_pairs"]) != release["distinct_model_scaler_pairs"]:
        raise ValueError("Release counts differ from the index")
    covered = [uid for pair in index["binary_pairs"].values() for uid in pair["logical_ids"]]
    if len(covered) != len(set(covered)) or set(covered) != set(index["models"]):
        raise ValueError("Release does not cover every logical fit exactly once")
    for cid, pair in index["binary_pairs"].items():
        for uid in pair["logical_ids"]:
            record = index["models"][uid]
            if record["canonical_artifact_id"] != cid:
                raise ValueError("Pair membership differs: " + uid)
            for field in ("model", "scalers", "model_sha256", "scalers_sha256"):
                if pair[field] != record[field]:
                    raise ValueError("Pair metadata differs: " + uid)
    if len(expected) != release["binary_files"]:
        raise ValueError("Unexpected number of distinct binary files")
    if args.archive:
        with args.archive.open("rb") as stream:
            checksum = digest(stream)
        if checksum != release["asset"]["sha256"] or args.archive.stat().st_size != release["asset"]["bytes"]:
            raise ValueError("Downloaded ZIP size or SHA-256 differs from release metadata")
        with zipfile.ZipFile(args.archive) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != set(expected) | set(release["auxiliary_archive_files"]):
                raise ValueError("Unexpected or duplicate archive entries")
            if archive.read(MANIFEST) != manifest_bytes:
                raise ValueError("Archive manifest differs from the repository manifest")
            for name, wanted in expected.items():
                with archive.open(name) as stream:
                    if digest(stream) != wanted:
                        raise ValueError("Archive artifact SHA-256 mismatch: " + name)
            for name in release["auxiliary_archive_files"]:
                if archive.read(name) != (ROOT / name).read_bytes():
                    raise ValueError("Archive documentation differs: " + name)
        mode = "archive"
    else:
        for name, wanted in expected.items():
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT):
                raise ValueError("Artifact resolves outside the repository: " + name)
            with path.open("rb") as stream:
                if digest(stream) != wanted:
                    raise ValueError("Extracted artifact SHA-256 mismatch: " + name)
        mode = "extracted files"
    print(f"PASS: {mode}; {release['models']} models, {release['scaler_files']} scaler files, "
          f"{release['logical_fits']} logical fits; all SHA-256 hashes match; no model fitting")


if __name__ == "__main__":
    main()
