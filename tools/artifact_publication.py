"""Publish a complete new artifact directory without replacing research evidence."""
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def fresh_artifact_directory(output):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists; use a fresh directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(f".{output.name}.publication-lock")
    lock.mkdir(mode=0o700)
    try:
        with tempfile.TemporaryDirectory(prefix=f".{output.name}.stage-", dir=output.parent) as temporary:
            stage = Path(temporary) / "package"
            stage.mkdir(mode=0o700)
            yield stage
            if output.exists() or output.is_symlink():
                raise FileExistsError(f"output appeared during build; not replacing it: {output}")
            os.rename(stage, output)
    finally:
        lock.rmdir()
