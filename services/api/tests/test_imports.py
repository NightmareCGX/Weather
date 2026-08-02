import os
import sys

# Ensure services/api/src is on sys.path for test execution
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, "../src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

"""Package import smoke tests."""

def test_package_imports():
    import api
    import api.core.config
    import api.core.database
    import api.models

    assert api.__version__ is not None
    assert api.core.config.settings is not None
    assert api.core.database.Base is not None
    assert api.models.ForecastCenter is not None
