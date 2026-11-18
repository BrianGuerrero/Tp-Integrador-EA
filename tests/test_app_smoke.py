"""Smoke test de la app: ejecuta app.py con un "streamlit" falso que devuelve los valores por defecto
de cada widget, recorriendo los 4 módulos. Detecta errores de ejecución sin levantar un servidor."""

import runpy
import sys
import types
from contextlib import contextmanager

import matplotlib
import pytest

from tpi.config import ARTIFACTS_DIR, ROOT

matplotlib.use("Agg")
pytestmark = pytest.mark.skipif(not (ARTIFACTS_DIR / "meta.json").exists(), reason="artefactos no generados")


class _Stop(Exception):
    pass


class _Fake(types.ModuleType):
    """Imita la API usada por la app; cada widget devuelve su valor por defecto."""

    def __init__(self, modulo_elegido):
        super().__init__("streamlit")
        self._modulo = modulo_elegido
        self.sidebar = self
        self.renders = 0

    # decoradores / layout
    def cache_resource(self, f):
        return f

    cache_data = cache_resource

    def set_page_config(self, **k):
        pass

    def columns(self, spec):
        n = spec if isinstance(spec, int) else len(spec)
        return [self] * n

    def tabs(self, names):
        return [self._ctx() for _ in names]

    @contextmanager
    def _ctx(self):
        yield self

    def form(self, *a, **k):
        return self._ctx()

    def stop(self):
        raise _Stop

    # widgets
    def radio(self, label, options, **k):
        return next(o for o in options if o.startswith(self._modulo))

    def selectbox(self, label, options, index=0, **k):
        return list(options)[index]

    def multiselect(self, label, options, default=None, **k):
        return list(default if default is not None else [])

    def slider(self, label, min_value=None, max_value=None, value=None, step=None, **k):
        return value

    def number_input(self, label, min_value=None, max_value=None, value=None, step=None, **k):
        return value

    def checkbox(self, label, value=False, **k):
        return value

    def form_submit_button(self, *a, **k):
        return True

    # salidas
    def pyplot(self, fig, **k):
        self.renders += 1

    def __getattr__(self, name):  # title, header, metric, markdown, caption, success, dataframe, ...
        return lambda *a, **k: None


@pytest.mark.parametrize("modulo", ["1", "2", "3", "Acerca"])
def test_app_ejecuta_cada_modulo(modulo, monkeypatch):
    fake = _Fake(modulo)
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    try:
        runpy.run_path(str(ROOT / "app" / "app.py"), run_name="__main__")
    except _Stop:
        pytest.fail("la app llamó a st.stop() (¿faltan artefactos o filtros vacíos?)")
    if modulo != "Acerca":
        assert fake.renders >= 1
