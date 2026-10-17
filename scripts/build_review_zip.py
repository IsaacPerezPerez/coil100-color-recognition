"""Paquete de revisión con selección explícita de archivos y verificación SHA-256."""

import hashlib
import json
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_archive(root: Path, destination: Path, paths: list[Path]) -> dict:
    partial = destination.with_suffix(destination.suffix + ".partial")
    if destination.exists() or partial.exists():
        raise ValueError("El ZIP o su archivo parcial ya existe; se preserva sin sobrescribir.")
    payload = {}
    for path in sorted(paths):
        name = path.relative_to(root).as_posix()
        if any(part in (".git", ".venv", "__pycache__", ".pytest_cache") for part in Path(name).parts) or name.startswith("data/raw/"):
            raise ValueError(f"Archivo excluido solicitado: {name}")
        data = path.read_bytes()
        if path.suffix in (".py", ".md", ".json", ".txt", ".csv"):
            contents = data.decode("utf-8-sig")
            if re.search(r"[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/]|/(?:Users|home)/[^/\s]+/", contents):
                raise ValueError(f"Revise la ruta personal presente en {name}.")
        payload[name] = data
    manifest = {"project": "coil100-color-recognition", "files": {name: {"size_bytes": len(data), "sha256": digest(data)} for name, data in payload.items()}}
    with ZipFile(partial, "x", compression=ZIP_DEFLATED) as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
        archive.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
    with ZipFile(partial) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(payload) | {"MANIFEST.json"}:
            raise ValueError("Falló la comprobación CRC o del inventario del ZIP.")
        extracted_manifest = json.loads(archive.read("MANIFEST.json"))
        for name, data in payload.items():
            extracted = archive.read(name)
            if digest(extracted) != digest(data) or len(extracted) != extracted_manifest["files"][name]["size_bytes"]:
                raise ValueError(f"Falló la integridad del contenido: {name}")
    partial.rename(destination)
    return {"archive": destination.name, "file_count": len(payload) + 1, "size_bytes": destination.stat().st_size, "sha256": digest(destination.read_bytes()), "crc_verified": True, "content_hashes_verified": True}


def review_paths(root: Path) -> list[Path]:
    paths = [root / name for name in ("README.md", "requirements.txt", ".gitignore", "data/processed/inventory.csv")]
    paths += [path for folder in ("src", "tests", "scripts") for path in (root / folder).rglob("*.py") if "__pycache__" not in path.parts]
    paths += list((root / "docs").glob("*.md"))
    for folder in ("artifacts/phase3a", "artifacts/phase3b", "artifacts/phase4"):
        paths += [path for path in (root / folder).glob("*") if path.suffix in (".csv", ".json", ".md", ".png", ".txt") and path.name != "package_verification.json"]
    required = ["artifacts/phase3a/train_mask_contact_sheet.png", "artifacts/phase3b/summary.csv", "artifacts/phase3b/evaluation_notes.md", "artifacts/phase3b/evaluation_metadata.json", "artifacts/phase4/verification_summary.json", "docs/resumen_academico.md", "docs/verificacion_independiente.md"]
    required += [f"artifacts/phase3b/{kind}_{space}.{extension}" for space in ("rgb", "hsv", "cielab") for kind, extension in (("predictions", "csv"), ("metrics", "csv"), ("confusion", "csv"), ("confusion", "png"))]
    selected = {path.relative_to(root).as_posix() for path in paths}
    if any(name not in selected or not (root / name).is_file() for name in required):
        raise ValueError("Faltan archivos obligatorios para revisión.")
    return paths


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result = build_archive(root, root / "T3_revision.zip", review_paths(root))
    (root / "artifacts/phase4/package_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (root / "T3_revision.zip.sha256").write_text(f"{result['sha256']}  T3_revision.zip\n", encoding="ascii")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
