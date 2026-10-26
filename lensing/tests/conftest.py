import sys
from types import ModuleType

# kmeans_radec is not on conda-forge: if it is missing, the tests that don't need the
# jackknife regions still run (those that do are skipped)
try:
    import kmeans_radec  # noqa: F401
except ImportError:
    sys.modules["kmeans_radec"] = ModuleType("kmeans_radec")
sys.modules["kmeans_radec"].__dict__.setdefault("kmeans_sample", None)
