"""Convierte notebooks/TP_Integrador_Final.py (formato "percent") en el .ipynb.

Uso:
    python scripts/build_notebook.py              # sólo convierte (sin salidas)
    python scripts/build_notebook.py --execute    # ejecuta celda por celda e incrusta salidas

Con --execute, las celdas marcadas con "# [DoWhy]", "# [EconML]" o "# [DoWhy+EconML]" se ejecutan
sólo si la librería está instalada; si no, quedan sin salida (se ejecutan luego con Restart & Run All).
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import sys
import time
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "TP_Integrador_Final.py"
DST = ROOT / "notebooks" / "TP_Integrador_Final.ipynb"


def parse(text: str) -> list[dict]:
    cells, cur, kind = [], [], None
    for line in text.splitlines():
        if line.startswith("# %%"):
            if kind is not None:
                cells.append({"kind": kind, "lines": cur})
            kind = "markdown" if "[markdown]" in line else "raw" if "[raw]" in line else "code"
            cur = []
        else:
            cur.append(line)
    cells.append({"kind": kind, "lines": cur})
    out = []
    for c in cells:
        lines = c["lines"]
        while lines and not lines[-1].strip():
            lines = lines[:-1]
        while lines and not lines[0].strip():
            lines = lines[1:]
        if c["kind"] in ("markdown", "raw"):
            lines = [ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in lines]
        out.append({"kind": c["kind"], "src": "\n".join(lines)})
    return out


def _lines(s: str) -> list[str]:
    parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]] if parts else []


def execute(cells: list[dict]) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.switch_backend("Agg")  # resolver el backend ANTES de crear el stub de IPython
    plt.close(plt.figure())

    outputs: list = []

    def display(obj):
        if isinstance(obj, pd.DataFrame | pd.Series):
            df = obj.to_frame() if isinstance(obj, pd.Series) else obj
            outputs.append({"output_type": "display_data", "metadata": {},
                            "data": {"text/html": _lines(df.to_html(float_format=lambda x: f"{x:,.4f}")),
                                     "text/plain": _lines(df.to_string())}})
        else:
            outputs.append({"output_type": "display_data", "metadata": {}, "data": {"text/plain": _lines(str(obj))}})

    def show(*a, **k):
        for num in plt.get_fignums():
            fig = plt.figure(num)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=90)
            outputs.append({"output_type": "display_data", "metadata": {},
                            "data": {"image/png": base64.b64encode(buf.getvalue()).decode(),
                                     "text/plain": ["<Figure>"]}})
        plt.close("all")

    ipy = types.ModuleType("IPython")
    ipy_d = types.ModuleType("IPython.display")
    ipy_d.display = display
    ipy.display = ipy_d
    ipy.get_ipython = lambda: None
    sys.modules["IPython"], sys.modules["IPython.display"] = ipy, ipy_d
    plt.show = show

    os.chdir(ROOT / "notebooks")
    ns: dict = {"__name__": "__main__"}
    n_exec = 0
    errores: list = []
    for i, c in enumerate(cells):
        if c["kind"] != "code":
            continue
        first = c["src"].lstrip().splitlines()[0] if c["src"].strip() else ""
        needs = [lib for lib in ("dowhy", "econml") if lib.lower() in first.lower() and first.startswith("# [")]
        if needs and any(__import__("importlib").util.find_spec(lib) is None for lib in needs):
            print(f"[celda {i}] omitida (requiere {', '.join(needs)})", flush=True)
            c["outputs"], c["execution_count"] = [], None
            continue
        outputs.clear()
        buf = io.StringIO()
        t0 = time.time()
        n_exec += 1
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(c["src"], f"<celda {i}>", "exec"), ns)
            show()
        except Exception as exc:
            print(f"[celda {i}] ERROR\n{traceback.format_exc()}", flush=True)
            errores.append(i)
            plt.close("all")
            outputs.append({"output_type": "error", "ename": type(exc).__name__, "evalue": str(exc),
                            "traceback": traceback.format_exc().splitlines()})
        res = []
        if buf.getvalue():
            res.append({"output_type": "stream", "name": "stdout", "text": _lines(buf.getvalue())})
        res.extend(outputs)
        c["outputs"], c["execution_count"] = list(res), n_exec
        print(f"[celda {i}] {'ERROR' if errores and errores[-1] == i else 'ok'} ({time.time() - t0:.1f}s)", flush=True)
    return errores


def to_ipynb(cells: list[dict]) -> dict:
    nb_cells = []
    for c in cells:
        if c["kind"] == "code":
            nb_cells.append({"cell_type": "code", "metadata": {}, "execution_count": c.get("execution_count"),
                             "outputs": c.get("outputs", []), "source": _lines(c["src"])})
        else:
            nb_cells.append({"cell_type": c["kind"], "metadata": {}, "source": _lines(c["src"])})
    return {"cells": nb_cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": sys.version.split()[0]}}}


if __name__ == "__main__":
    cells = parse(SRC.read_text(encoding="utf-8"))
    errores = execute(cells) if "--execute" in sys.argv else []
    DST.write_text(json.dumps(to_ipynb(cells), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Escrito {DST} ({sum(c['kind'] == 'code' for c in cells)} celdas de código) | errores: {errores}")
    sys.exit(1 if errores else 0)
