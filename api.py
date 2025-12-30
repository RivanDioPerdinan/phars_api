from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

DATA_PATH = "data/covid_19_indonesia_time_series_all.csv"

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("/", "_")
    )
    return df

def load_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = normalize_columns(df)

    # required
    if "date" not in df.columns:
        raise ValueError(f"No date column found. Columns: {df.columns.tolist()}")
    if "location" not in df.columns:
        raise ValueError(f"No location column found. Columns: {df.columns.tolist()}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # optional but useful
    if "location_level" not in df.columns:
        df["location_level"] = "Unknown"

    return df

DF = load_df()

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    location = request.args.get("location")
    level = request.args.get("level")
    start = request.args.get("start")
    end = request.args.get("end")

    out = df

    if level:
        out = out[out["location_level"] == level]

    if location:
        out = out[out["location"] == location]

    if start:
        s = pd.to_datetime(start, errors="coerce")
        if pd.notna(s):
            out = out[out["date"] >= s]

    if end:
        e = pd.to_datetime(end, errors="coerce")
        if pd.notna(e):
            out = out[out["date"] <= e]

    return out

@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})

@app.get("/api/metadata")
def metadata():
    # return levels and locations grouped by level
    min_date = DF["date"].min().date().isoformat()
    max_date = DF["date"].max().date().isoformat()

    levels = sorted(DF["location_level"].dropna().unique().tolist())
    grouped = {}
    for lv in levels:
        grouped[lv] = sorted(DF.loc[DF["location_level"] == lv, "location"].dropna().unique().tolist())

    return jsonify({
        "min_date": min_date,
        "max_date": max_date,
        "row_count": int(len(DF)),
        "columns": DF.columns.tolist(),
        "levels": levels,
        "locations_by_level": grouped
    })

@app.get("/api/summary")
def summary():
    df = apply_filters(DF).sort_values("date")

    if df.empty:
        return jsonify({"count": 0, "kpi": {}, "message": "No data for filters"})

    last = df.iloc[-1]
    kpi = {
        "min_date": df["date"].min().date().isoformat(),
        "max_date": df["date"].max().date().isoformat(),
    }

    if "total_cases" in df.columns and pd.notna(last.get("total_cases")):
        kpi["total_cases"] = int(last["total_cases"])
    if "total_deaths" in df.columns and pd.notna(last.get("total_deaths")):
        kpi["total_deaths"] = int(last["total_deaths"])

    if "new_cases" in df.columns:
        kpi["new_cases_7d"] = int(df.tail(7)["new_cases"].fillna(0).sum())
    if "new_deaths" in df.columns:
        kpi["new_deaths_7d"] = int(df.tail(7)["new_deaths"].fillna(0).sum())

    return jsonify({"count": int(len(df)), "kpi": kpi})

@app.get("/api/cases")
def cases():
    df = apply_filters(DF).sort_values("date")

    limit = request.args.get("limit", default="3000")
    try:
        limit_n = max(1, min(int(limit), 20000))
    except ValueError:
        limit_n = 3000

    df = df.tail(limit_n)

    cols = [c for c in [
        "date", "location", "location_level",
        "new_cases", "new_deaths",
        "total_cases", "total_deaths",
        "latitude", "longitude"
    ] if c in df.columns]

    out = df[cols].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")

    return jsonify({"count": int(len(out)), "data": out.to_dict(orient="records")})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

