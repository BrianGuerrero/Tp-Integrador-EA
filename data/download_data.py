"""Descarga el dataset original desde UCI y lo guarda como data/diabetic_data.csv."""

from pathlib import Path

from ucimlrepo import fetch_ucirepo

if __name__ == "__main__":
    ds = fetch_ucirepo(id=296)
    out = Path(__file__).resolve().parent / "diabetic_data.csv"
    df = ds.data.original.copy()
    # "None" (test no realizado) no es faltante; el resto de los faltantes se guardan como "?" (formato original)
    for c in ["A1Cresult", "max_glu_serum"]:
        df[c] = df[c].fillna("None")
    df.fillna("?").to_csv(out, index=False)
    print(f"Guardado en {out} ({len(df):,} filas)")
