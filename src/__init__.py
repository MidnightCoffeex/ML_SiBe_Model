# Markiert das Verzeichnis als Paket und erlaubt den bequemen Import der Kernmodule.

from . import data_pipeline  # noqa: F401

# Optional imports: allow lightweight tools (e.g. feature GUI) to run
# even when training dependencies are not installed.
try:  # noqa: SIM105
    from . import train_model  # noqa: F401
except Exception:
    train_model = None
try:  # noqa: SIM105
    from . import evaluate_model  # noqa: F401
except Exception:
    evaluate_model = None
try:  # noqa: SIM105
    from . import nn_data  # noqa: F401
except Exception:
    nn_data = None
try:  # noqa: SIM105
    from . import nn_models  # noqa: F401
except Exception:
    nn_models = None


