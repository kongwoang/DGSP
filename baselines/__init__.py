"""Baselines of the main comparison (see README.md). REGISTRY: row name -> constructor(dataset, detector, looming, params);
imported lazily, so a missing optional dependency only affects its own row."""
import importlib

_ROWS = {"AnA": ("baselines.ana", "AnA"), "Counterfactual": ("baselines.counterfactual", "Counterfactual"),
         "AdaScale": ("baselines.adascale", "AdaScale"), "SAHI": ("baselines.sahi", "SAHI"), "UFP": ("baselines.ufp", "UFP"),
         "FOVEA": ("baselines.fovea", "FOVEA"), "BAR": ("baselines.bar", "BAR"), "Chanakya": ("baselines.chanakya", "Chanakya")}


def _lazy(mod, cls):
    def build(*a, **k):
        return getattr(importlib.import_module(mod), cls)(*a, **k)
    return build


REGISTRY = {name: _lazy(m, c) for name, (m, c) in _ROWS.items()}
