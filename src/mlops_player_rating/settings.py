"""Project settings.

Anything not overridden here falls back to Kedro's framework defaults, which is
intentional: we keep the surface area small so the project is easy to reason about.
See https://docs.kedro.org/en/stable/kedro_project_setup/settings.html
"""

from kedro.config import OmegaConfigLoader

# Class that manages how configuration is loaded.
CONFIG_LOADER_CLASS = OmegaConfigLoader

# Keyword arguments to pass to the `CONFIG_LOADER_CLASS` constructor.
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    "config_patterns": {
        "spark": ["spark*", "spark*/**"],
    },
}

# Class that manages the Data Catalog.
# from kedro.io import DataCatalog
# DATA_CATALOG_CLASS = DataCatalog
