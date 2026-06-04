import subprocess
import sys
from pathlib import Path

QUICKSTART = Path(__file__).resolve().parent.parent / "examples" / "quickstart.py"


def test_quickstart_runs_end_to_end():
    """The quickstart example should run start-to-finish on the core install."""
    result = subprocess.run(
        [sys.executable, str(QUICKSTART)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    # it walks every stage: generation, deduplication, featurisation, prediction
    out = result.stdout
    assert "generate products" in out
    assert "duplicates merged" in out
    assert "featurise" in out
    assert "predict" in out
