import json
import os
import urllib.request
import urllib.error
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Academic Shield")


@app.post("/api/feedback")
async def feedback(req: Request):

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        return JSONResponse(
            {"error": "Supabase env vars not configured"}, status_code=500
        )

    b = await req.json()
    # All the inputs from the user
    row = {
        "study_hours": b.get("study_hours"),
        "sleep_hours": b.get("sleep_hours"),
        "eca_hours": b.get("eca_hours"),
        "social_hours": b.get("social_hours"),
        "physical_hours": b.get("physical_hours"),
        "stress_level_category": b.get("stress_level_category"),
        "exam_pressure": b.get("exam_pressure"),
        "family_expectation": b.get("family_expectation"),
        "financial_stress": b.get("financial_stress"),
        "social_support": b.get("social_support"),
        "anxiety_score": b.get("anxiety_score"),
        "depression_score": b.get("depression_score"),
        "burnout_gauge_score": float(b.get("burnout_gauge_score", 0)),
        "burnout_class": b.get("burnout_class"),
        "predicted_gpa": float(b.get("predicted_gpa", 0)),
        "rating": int(b.get("rating", 0)),
        "comment": b.get("comment", ""),
    }

    payload = json.dumps(row).encode()
    request = urllib.request.Request(
        f"{url}/rest/v1/feedback",
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Prefer": "return=minimal",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as r:
            r.read()
        return {"ok": True}
    except urllib.error.HTTPError as e:
        return JSONResponse(
            {"error": f"Supabase error: {e.read().decode()}"}, status_code=502
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# Cache transpiled models aggressively
class CachedStatic(StaticFiles):
    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        if path.startswith("models/") and resp.status_code == 200:
            resp.headers["Cache-Control"] = (
                "public, max-age=31536000, immutable"  # Cache the models for one year
            )
        return resp


app.mount("/", CachedStatic(directory=STATIC_DIR, html=True), name="static")