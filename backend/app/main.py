from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import text
from app.db import engine, init_db
from app.predictor import predict_match_over12_corners
from app.services.v2_foundation import get_foundation_metrics, apply_foundation_adjustment
from app.services.v2_player_form import get_team_player_form_snapshot, summarize_team_player_form
from app.services.v2_roster_logic import get_roster_context, compute_roster_adjustment
from app.services.v2_combined_logic import compute_combined_v2_adjustment
from app.services.v2_recent_window import explain_match_prediction_v2_recent_window, upcoming_worldcup_recent_window
from app.services.data_room import get_db_overview, list_competitions_summary, get_fixture_dataset
from app.services.agent_analyst import (
    list_ollama_models,
    analyze_fixture_with_ollama,
    chat_fixture_with_ollama,
)

app = FastAPI(title="World Cup Corner Radar")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    init_db()

class ExplainMatchInput(BaseModel):
    fixture_id: int

class AgentAnalyzeFixtureIn(BaseModel):
    fixture_id: int
    mode: str = "prematch"
    perspectives: list[str] | None = None
    model: str | None = None

class AgentChatFixtureIn(BaseModel):
    fixture_id: int
    question: str
    mode: str = "prematch"
    model: str | None = None
    history: list[dict[str, str]] | None = None

@app.get("/radar/health")
def radar_health():
    with engine.begin() as db:
        row = db.execute(text("""
            SELECT
              COUNT(*) FILTER (WHERE api_league_id = 1) AS worldcup_matches,
              COUNT(*) FILTER (WHERE api_league_id = 1 AND fixture_id IN (SELECT DISTINCT fixture_id FROM match_team_stats)) AS worldcup_matches_with_stats
            FROM matches
        """)).mappings().first()

        return {
            "ok": True,
            "project": "World Cup Corner Radar",
            "mvp_model": "v0.6_worldcup_sweetspot_calibrated",
            "scope": "World Cup historical + future-ready",
            "worldcup_matches": row["worldcup_matches"] if row else 0,
            "worldcup_matches_with_stats": row["worldcup_matches_with_stats"] if row else 0,
            "available_endpoints": [
                "/radar/worldcup",
                "/radar/worldcup/match/{fixture_id}",
                "/thermometer/worldcup-v06",
                "/predictions-v06/{fixture_id}",
                "/backtest/worldcup-v06-buckets",
            ],
        }

@app.get("/analytics/db/overview")
def analytics_db_overview():
    with engine.begin() as db:
        return {"ok": True, "overview": get_db_overview(db)}

@app.get("/analytics/competitions")
def analytics_competitions():
    with engine.begin() as db:
        return {"ok": True, "competitions": list_competitions_summary(db)}

@app.get("/radar/worldcup/match/{fixture_id}/dataset")
def radar_worldcup_fixture_dataset(fixture_id: int):
    with engine.begin() as db:
        return get_fixture_dataset(db, fixture_id)

@app.get("/agent/models")
def agent_models():
    return list_ollama_models()

@app.post("/agent/analyze-fixture")
def agent_analyze_fixture(payload: AgentAnalyzeFixtureIn):
    with engine.begin() as db:
        return analyze_fixture_with_ollama(
            db=db,
            fixture_id=payload.fixture_id,
            mode=payload.mode,
            perspectives=payload.perspectives,
            model=payload.model,
        )

@app.post("/agent/chat-fixture")
def agent_chat_fixture(payload: AgentChatFixtureIn):
    with engine.begin() as db:
        return chat_fixture_with_ollama(
            db=db,
            fixture_id=payload.fixture_id,
            question=payload.question,
            mode=payload.mode,
            model=payload.model,
            history=payload.history,
        )

@app.get("/radar/worldcup")
def radar_worldcup(limit: int = Query(20, ge=1, le=200), min_probability: float = Query(0, ge=0, le=100)):
    with engine.begin() as db:
        rows = db.execute(text("""
            WITH latest_predictions AS (
                SELECT DISTINCT ON (fixture_id)
                    fixture_id,
                    model_name,
                    over12_corners,
                    over14_corners,
                    expected_corners,
                    shots_range_low,
                    shots_range_high,
                    sot_range_low,
                    sot_range_high,
                    thermometer,
                    explanation,
                    created_at
                FROM predictions
                ORDER BY fixture_id, created_at DESC
            )
            SELECT
                m.fixture_id,
                m.fixture_date,
                CONCAT(m.home_team_name, ' - ', m.away_team_name) AS match,
                m.home_team_name,
                m.away_team_name,
                m.season,
                lp.over12_corners AS over12_corner_probability,
                lp.over14_corners AS over14_corner_probability,
                lp.expected_corners,
                ARRAY[lp.shots_range_low, lp.shots_range_high] AS total_shots_range,
                ARRAY[lp.sot_range_low, lp.sot_range_high] AS shots_on_target_range,
                lp.thermometer,
                COALESCE(lp.model_name, 'v0.6_worldcup_sweetspot_calibrated') AS model
            FROM matches m
            LEFT JOIN latest_predictions lp ON lp.fixture_id = m.fixture_id
            WHERE m.api_league_id = 1
            ORDER BY m.fixture_date ASC
            LIMIT :limit
        """), {"limit": limit}).mappings().all()

        matches = []
        for r in rows:
            over12 = float(r["over12_corner_probability"] or 0)
            if over12 < min_probability:
                continue
            matches.append({
                "fixture_id": r["fixture_id"],
                "fixture_date": r["fixture_date"].isoformat() if r["fixture_date"] else None,
                "match": r["match"],
                "home_team_name": r["home_team_name"],
                "away_team_name": r["away_team_name"],
                "season": r["season"],
                "over12_corner_probability": round(over12, 2),
                "over14_corner_probability": round(float(r["over14_corner_probability"] or 0), 2),
                "expected_corners": round(float(r["expected_corners"] or 0), 2),
                "total_shots_range": [int((r["total_shots_range"] or [0, 0])[0]), int((r["total_shots_range"] or [0, 0])[1])],
                "shots_on_target_range": [int((r["shots_on_target_range"] or [0, 0])[0]), int((r["shots_on_target_range"] or [0, 0])[1])],
                "thermometer": r["thermometer"] or "🟡 Neutral Match",
                "model": r["model"],
            })

        return {
            "project": "World Cup Corner Radar",
            "view": "all",
            "model": "v0.6_worldcup_sweetspot_calibrated",
            "scope": "World Cup",
            "count": len(matches),
            "matches": matches,
        }

@app.get("/radar/worldcup/upcoming")
def radar_worldcup_upcoming(limit: int = Query(20, ge=1, le=200), min_probability: float = Query(0, ge=0, le=100)):
    with engine.begin() as db:
        rows = upcoming_worldcup_recent_window(db, limit=limit)
        matches = []
        for r in rows:
            over12 = float(r.get("over12_corners") or r.get("over12_corner_probability") or 0)
            if over12 < min_probability:
                continue
            matches.append({
                "fixture_id": r["fixture_id"],
                "fixture_date": r["fixture_date"],
                "match": r["match"],
                "home_team_name": r["home_team_name"],
                "away_team_name": r["away_team_name"],
                "season": r["season"],
                "over12_corner_probability": round(over12, 2),
                "over14_corner_probability": round(float(r.get("over14_corners") or 0), 2),
                "expected_corners": round(float(r.get("expected_corners") or 0), 2),
                "total_shots_range": r.get("shots_range") or [0, 0],
                "shots_on_target_range": r.get("shots_on_target_range") or [0, 0],
                "thermometer": r.get("thermometer") or "🟡 Neutral Match",
                "confidence": r.get("confidence"),
                "model": r.get("model") or "v2_recent_window_roster_aware",
            })
        return {
            "project": "World Cup Corner Radar",
            "view": "upcoming",
            "model": "v0.6_worldcup_sweetspot_calibrated",
            "scope": "World Cup 2026",
            "count": len(matches),
            "matches": matches,
        }

@app.get("/radar/worldcup/live")
def radar_worldcup_live(limit: int = Query(20, ge=1, le=200)):
    with engine.begin() as db:
        rows = db.execute(text("""
            SELECT
                m.fixture_id,
                m.fixture_date,
                CONCAT(m.home_team_name, ' - ', m.away_team_name) AS match,
                m.home_team_name,
                m.away_team_name,
                m.season,
                m.status_short
            FROM matches m
            WHERE m.api_league_id = 1
              AND m.status_short NOT IN ('NS', 'FT', 'AET', 'PEN', 'CANC', 'PST')
            ORDER BY m.fixture_date ASC
            LIMIT :limit
        """), {"limit": limit}).mappings().all()

        matches = [{
            "fixture_id": r["fixture_id"],
            "fixture_date": r["fixture_date"].isoformat() if r["fixture_date"] else None,
            "match": r["match"],
            "home_team_name": r["home_team_name"],
            "away_team_name": r["away_team_name"],
            "season": r["season"],
            "over12_corner_probability": 0,
            "over14_corner_probability": 0,
            "expected_corners": 0,
            "total_shots_range": [0, 0],
            "shots_on_target_range": [0, 0],
            "thermometer": "🔴 Live",
            "model": "live",
        } for r in rows]

        return {
            "project": "World Cup Corner Radar",
            "view": "live",
            "model": "live",
            "scope": "World Cup live",
            "count": len(matches),
            "matches": matches,
        }

@app.get("/radar/worldcup/historical")
def radar_worldcup_historical(limit: int = Query(20, ge=1, le=200), min_probability: float = Query(0, ge=0, le=100)):
    with engine.begin() as db:
        rows = db.execute(text("""
            WITH latest_predictions AS (
                SELECT DISTINCT ON (fixture_id)
                    fixture_id,
                    model_name,
                    over12_corners,
                    over14_corners,
                    expected_corners,
                    shots_range_low,
                    shots_range_high,
                    sot_range_low,
                    sot_range_high,
                    thermometer,
                    created_at
                FROM predictions
                ORDER BY fixture_id, created_at DESC
            )
            SELECT
                m.fixture_id,
                m.fixture_date,
                CONCAT(m.home_team_name, ' - ', m.away_team_name) AS match,
                m.home_team_name,
                m.away_team_name,
                m.season,
                lp.over12_corners,
                lp.over14_corners,
                lp.expected_corners,
                lp.shots_range_low,
                lp.shots_range_high,
                lp.sot_range_low,
                lp.sot_range_high,
                lp.thermometer,
                lp.model_name
            FROM matches m
            LEFT JOIN latest_predictions lp ON lp.fixture_id = m.fixture_id
            WHERE m.api_league_id = 1
              AND m.status_short IN ('FT', 'AET', 'PEN')
            ORDER BY m.fixture_date DESC
            LIMIT :limit
        """), {"limit": limit}).mappings().all()

        matches = []
        for r in rows:
            over12 = float(r["over12_corners"] or 0)
            if over12 < min_probability:
                continue
            matches.append({
                "fixture_id": r["fixture_id"],
                "fixture_date": r["fixture_date"].isoformat() if r["fixture_date"] else None,
                "match": r["match"],
                "home_team_name": r["home_team_name"],
                "away_team_name": r["away_team_name"],
                "season": r["season"],
                "over12_corner_probability": round(over12, 2),
                "over14_corner_probability": round(float(r["over14_corners"] or 0), 2),
                "expected_corners": round(float(r["expected_corners"] or 0), 2),
                "total_shots_range": [int(r["shots_range_low"] or 0), int(r["shots_range_high"] or 0)],
                "shots_on_target_range": [int(r["sot_range_low"] or 0), int(r["sot_range_high"] or 0)],
                "thermometer": r["thermometer"] or "🟡 Neutral Match",
                "model": r["model_name"] or "v0.6_worldcup_sweetspot_calibrated",
            })

        return {
            "project": "World Cup Corner Radar",
            "view": "historical",
            "model": "v0.6_worldcup_sweetspot_calibrated",
            "scope": "World Cup historical",
            "count": len(matches),
            "matches": matches,
        }

@app.get("/radar/worldcup/match/{fixture_id}")
def radar_worldcup_match(fixture_id: int):
    with engine.begin() as db:
        row = db.execute(text("""
            WITH latest_predictions AS (
                SELECT DISTINCT ON (fixture_id)
                    fixture_id,
                    model_name,
                    over12_corners,
                    over14_corners,
                    expected_corners,
                    shots_range_low,
                    shots_range_high,
                    sot_range_low,
                    sot_range_high,
                    thermometer,
                    explanation,
                    created_at
                FROM predictions
                ORDER BY fixture_id, created_at DESC
            )
            SELECT
                m.fixture_id,
                m.fixture_date,
                m.status_short,
                CONCAT(m.home_team_name, ' - ', m.away_team_name) AS match,
                m.home_team_name,
                m.away_team_name,
                m.season,
                lp.model_name,
                lp.over12_corners,
                lp.over14_corners,
                lp.expected_corners,
                lp.shots_range_low,
                lp.shots_range_high,
                lp.sot_range_low,
                lp.sot_range_high,
                lp.thermometer,
                lp.explanation
            FROM matches m
            LEFT JOIN latest_predictions lp ON lp.fixture_id = m.fixture_id
            WHERE m.fixture_id = :fixture_id
        """), {"fixture_id": fixture_id}).mappings().first()

        if not row:
            return {"ok": False, "error": "fixture_not_found", "fixture_id": fixture_id}

        return {
            "ok": True,
            "fixture_id": row["fixture_id"],
            "fixture_date": row["fixture_date"].isoformat() if row["fixture_date"] else None,
            "status_short": row["status_short"],
            "match": row["match"],
            "home_team_name": row["home_team_name"],
            "away_team_name": row["away_team_name"],
            "season": row["season"],
            "model": row["model_name"] or "v0.6_worldcup_sweetspot_calibrated",
            "over12_corner_probability": round(float(row["over12_corners"] or 0), 2),
            "over14_corner_probability": round(float(row["over14_corners"] or 0), 2),
            "expected_corners": round(float(row["expected_corners"] or 0), 2),
            "total_shots_range": [int(row["shots_range_low"] or 0), int(row["shots_range_high"] or 0)],
            "shots_on_target_range": [int(row["sot_range_low"] or 0), int(row["sot_range_high"] or 0)],
            "thermometer": row["thermometer"] or "🟡 Neutral Match",
            "explanation": row["explanation"],
        }

@app.get("/thermometer/worldcup-v06")
def thermometer_worldcup_v06(limit: int = Query(20, ge=1, le=200)):
    return radar_worldcup_upcoming(limit=limit)

@app.get("/predictions-v06/{fixture_id}")
def predictions_v06_fixture(fixture_id: int):
    return radar_worldcup_match(fixture_id)

@app.get("/backtest/worldcup-v06-buckets")
def backtest_worldcup_v06_buckets():
    with engine.begin() as db:
        rows = db.execute(text("""
            SELECT
                CASE
                    WHEN p.over12_corners < 5 THEN '0-5'
                    WHEN p.over12_corners < 10 THEN '5-10'
                    WHEN p.over12_corners < 15 THEN '10-15'
                    WHEN p.over12_corners < 20 THEN '15-20'
                    WHEN p.over12_corners < 25 THEN '20-25'
                    ELSE '25+'
                END AS bucket,
                COUNT(*) AS sample_size,
                AVG(CASE WHEN COALESCE(ht.corner_kicks, 0) + COALESCE(at.corner_kicks, 0) >= 12 THEN 1.0 ELSE 0.0 END) AS actual_over12_rate
            FROM predictions p
            JOIN matches m ON m.fixture_id = p.fixture_id
            LEFT JOIN match_team_stats ht ON ht.fixture_id = m.fixture_id AND ht.team_id = m.home_team_id
            LEFT JOIN match_team_stats at ON at.fixture_id = m.fixture_id AND at.team_id = m.away_team_id
            WHERE m.api_league_id = 1
              AND m.status_short IN ('FT', 'AET', 'PEN')
            GROUP BY 1
            ORDER BY 1
        """)).mappings().all()
        return {"ok": True, "buckets": [dict(r) for r in rows]}
