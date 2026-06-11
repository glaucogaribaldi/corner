import json
import os
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import requests

from app.services.data_room import get_fixture_dataset

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))
API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"
API_FOOTBALL_KEY = os.getenv("APIFOOTBALL_KEY")

PREFERRED_MODELS = [
    "qwen3:8b",
    "llama3.1:8b",
    "gemma3:12b",
    "codellama:13b",
    "qwen2.5-coder:7b",
]

def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    return obj

def _num(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def list_ollama_models() -> dict[str, Any]:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=20)
        r.raise_for_status()
        payload = r.json()
        models = payload.get("models", []) or []
        names = [m.get("name") for m in models if m.get("name")]
        return {"ok": True, "models": names}
    except Exception as e:
        return {"ok": False, "models": [], "error": str(e)}

def choose_ollama_model(requested_model: str | None = None) -> str:
    if requested_model:
        return requested_model
    env_model = os.getenv("OLLAMA_MODEL")
    if env_model:
        return env_model
    tags = list_ollama_models()
    names = tags.get("models", []) if tags.get("ok") else []
    for preferred in PREFERRED_MODELS:
        if preferred in names:
            return preferred
    if names:
        return names[0]
    return "qwen3:8b"

def football_api_get(endpoint: str, params: dict | None = None) -> dict[str, Any] | None:
    if not API_FOOTBALL_KEY:
        return None
    try:
        r = requests.get(
            f"{API_FOOTBALL_BASE_URL}/{endpoint}",
            headers={"x-apisports-key": API_FOOTBALL_KEY},
            params=params or {},
            timeout=25,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def _trim_last_matches(rows, n=8):
    out = []
    for r in (rows or [])[:n]:
        out.append(_json_safe({
            "fixture_date": r.get("fixture_date"),
            "api_league_id": r.get("api_league_id"),
            "opponent_name": r.get("opponent_name"),
            "corner_kicks": r.get("corner_kicks"),
            "total_shots": r.get("total_shots"),
            "shots_on_goal": r.get("shots_on_goal"),
        }))
    return out

def summarize_fixture_list(items):
    out = []
    for item in (items or [])[:5]:
        fixture = item.get("fixture", {}) or {}
        teams = item.get("teams", {}) or {}
        goals = item.get("goals", {}) or {}
        league = item.get("league", {}) or {}
        out.append(_json_safe({
            "date": fixture.get("date"),
            "status": (fixture.get("status", {}) or {}).get("short"),
            "league": league.get("name"),
            "round": league.get("round"),
            "home": (teams.get("home", {}) or {}).get("name"),
            "away": (teams.get("away", {}) or {}).get("name"),
            "home_goals": goals.get("home"),
            "away_goals": goals.get("away"),
        }))
    return out

def compute_winner_lean(dataset: dict[str, Any]) -> dict[str, Any]:
    recent = dataset.get("recent_team_windows") or {}
    home = recent.get("home") or {}
    away = recent.get("away") or {}

    h_sample = _num(home.get("matches_count"))
    a_sample = _num(away.get("matches_count"))

    home_score = (
        _num(home.get("shots_for_avg")) * 0.9 +
        _num(home.get("sot_for_avg")) * 1.4 +
        _num(home.get("corners_for_avg")) * 0.7 -
        _num(home.get("shots_against_avg")) * 0.45 -
        _num(home.get("sot_against_avg")) * 0.8 -
        _num(home.get("corners_against_avg")) * 0.35
    )
    away_score = (
        _num(away.get("shots_for_avg")) * 0.9 +
        _num(away.get("sot_for_avg")) * 1.4 +
        _num(away.get("corners_for_avg")) * 0.7 -
        _num(away.get("shots_against_avg")) * 0.45 -
        _num(away.get("sot_against_avg")) * 0.8 -
        _num(away.get("corners_against_avg")) * 0.35
    )

    diff = home_score - away_score
    min_sample = min(h_sample, a_sample)

    if abs(diff) < 0.75:
        lean = "balanced"
    elif diff > 0:
        lean = "home"
    else:
        lean = "away"

    if min_sample >= 18 and abs(diff) >= 1.5:
        confidence = "medium"
    elif min_sample >= 10:
        confidence = "medium-low"
    else:
        confidence = "low"

    return _json_safe({
        "winner_lean": lean,
        "winner_lean_confidence": confidence,
        "home_strength_score": round(home_score, 2),
        "away_strength_score": round(away_score, 2),
        "strength_gap": round(diff, 2),
        "note": "Heuristic lean from recent windows, shots, shots on target, corners and concession profile."
    })

def fetch_external_context(fixture: dict[str, Any]) -> dict[str, Any]:
    fixture_id = fixture.get("fixture_id")
    home_team_id = fixture.get("home_team_id")
    away_team_id = fixture.get("away_team_id")

    external = {
        "fixture_summary": None,
        "home_last5_summary": [],
        "away_last5_summary": [],
        "head_to_head_last5_summary": [],
    }

    if fixture_id:
        payload = football_api_get("fixtures", {"id": fixture_id})
        if payload and payload.get("response"):
            item = payload["response"][0]
            external["fixture_summary"] = _json_safe({
                "date": ((item.get("fixture", {}) or {}).get("date")),
                "status": (((item.get("fixture", {}) or {}).get("status", {}) or {}).get("short")),
                "league": ((item.get("league", {}) or {}).get("name")),
                "round": ((item.get("league", {}) or {}).get("round")),
                "home": (((item.get("teams", {}) or {}).get("home", {}) or {}).get("name")),
                "away": (((item.get("teams", {}) or {}).get("away", {}) or {}).get("name")),
            })

    if home_team_id:
        payload = football_api_get("fixtures", {"team": home_team_id, "last": 5})
        if payload:
            external["home_last5_summary"] = summarize_fixture_list(payload.get("response", [])[:5])

    if away_team_id:
        payload = football_api_get("fixtures", {"team": away_team_id, "last": 5})
        if payload:
            external["away_last5_summary"] = summarize_fixture_list(payload.get("response", [])[:5])

    if home_team_id and away_team_id:
        payload = football_api_get("fixtures/headtohead", {"h2h": f"{home_team_id}-{away_team_id}", "last": 5})
        if payload:
            external["head_to_head_last5_summary"] = summarize_fixture_list(payload.get("response", [])[:5])

    return external

def build_llm_context(dataset: dict[str, Any], mode: str = "prematch", include_external: bool = True) -> dict[str, Any]:
    fixture = dataset.get("fixture") or {}
    context = {
        "fixture": _json_safe(fixture),
        "mvp_prediction_latest": _json_safe(dataset.get("mvp_prediction_latest")),
        "team_stats_for_fixture": _json_safe(dataset.get("team_stats_for_fixture") or []),
        "recent_team_windows": _json_safe(dataset.get("recent_team_windows") or {}),
        "last_matches": {
            "home": _trim_last_matches((dataset.get("last_matches") or {}).get("home")),
            "away": _trim_last_matches((dataset.get("last_matches") or {}).get("away")),
        },
        "global_baselines": _json_safe(dataset.get("global_baselines") or {}),
        "winner_lean_heuristic": compute_winner_lean(dataset),
        "mode": mode,
    }
    if include_external:
        context["external_football_api"] = fetch_external_context(fixture)
    return _json_safe(context)

def _perspective_names(perspectives):
    return [p for p in (perspectives or []) if p in {"conservative", "aggressive", "contrarian", "live_sensitive"}]

def deterministic_analysis(dataset: dict[str, Any], mode: str, perspectives: list[str] | None = None) -> dict[str, Any]:
    fixture = dataset.get("fixture") or {}
    recent = dataset.get("recent_team_windows") or {}
    home = recent.get("home") or {}
    away = recent.get("away") or {}
    mvp = dataset.get("mvp_prediction_latest") or {}
    ext = fetch_external_context(fixture)
    heuristic = compute_winner_lean(dataset)

    home_name = fixture.get("home_team_name", "Home")
    away_name = fixture.get("away_team_name", "Away")
    match = f"{home_name} - {away_name}"

    requested = _perspective_names(perspectives or ["conservative", "aggressive", "contrarian", "live_sensitive"])

    home_cf = _num(home.get("corners_for_avg"))
    away_cf = _num(away.get("corners_for_avg"))
    home_ca = _num(home.get("corners_against_avg"))
    away_ca = _num(away.get("corners_against_avg"))
    home_sf = _num(home.get("shots_for_avg"))
    away_sf = _num(away.get("shots_for_avg"))
    home_sotf = _num(home.get("sot_for_avg"))
    away_sotf = _num(away.get("sot_for_avg"))
    home_sota = _num(home.get("sot_against_avg"))
    away_sota = _num(away.get("sot_against_avg"))
    h_n = int(_num(home.get("matches_count")))
    a_n = int(_num(away.get("matches_count")))

    exp_corners = mvp.get("expected_corners")
    over12 = mvp.get("over12_corners")
    if exp_corners is None:
        exp_corners = (home_cf + away_cf + home_ca + away_ca) / 2 if (home_cf + away_cf + home_ca + away_ca) else 0
    if over12 is None:
        over12 = 0

    summary = (
        f"{home_name} arriva con {home_cf:.2f} corner medi a favore e {home_sf:.2f} tiri medi "
        f"nelle ultime {h_n} partite del campione disponibile, mentre {away_name} produce "
        f"{away_cf:.2f} corner e {away_sf:.2f} tiri nelle ultime {a_n}. "
        f"Il profilo pre-match è orientato più al volume offensivo che a un equilibrio puramente difensivo, "
        f"con expected corners a {float(exp_corners):.2f} e Over12 a {float(over12):.2f}%."
    )

    analysis_perspectives = []

    if "conservative" in requested:
        analysis_perspectives.append({
            "name": "conservative",
            "thesis": f"Lettura prudente: match con buon volume, ma senza uno sbilanciamento abbastanza netto da giustificare una lettura estrema.",
            "evidence": [
                f"{home_name}: {home_cf:.2f} corner fatti / {home_ca:.2f} concessi.",
                f"{away_name}: {away_cf:.2f} corner fatti / {away_ca:.2f} concessi.",
                f"Campione recente: {h_n} partite per {home_name} e {a_n} per {away_name}."
            ],
            "risk_flags": [
                "Gap di forza non ampio.",
                "Campioni recenti non perfettamente simmetrici."
            ]
        })

    if "aggressive" in requested:
        stronger = home_name if heuristic["winner_lean"] == "home" else away_name if heuristic["winner_lean"] == "away" else "nessuna delle due in modo netto"
        analysis_perspectives.append({
            "name": "aggressive",
            "thesis": f"Lettura aggressiva: il lato {stronger} ha il profilo leggermente migliore per imporre pressione offensiva.",
            "evidence": [
                f"Strength gap euristico: {heuristic['strength_gap']:.2f}.",
                f"Shots on target: {home_name} {home_sotf:.2f} vs {away_name} {away_sotf:.2f}.",
                f"Expected corners pre-match: {float(exp_corners):.2f}."
            ],
            "risk_flags": [
                "Lean euristico, non probabilità vera di vittoria.",
                "Un match aperto può ribaltare il lato del controllo territoriale."
            ]
        })

    if "contrarian" in requested:
        analysis_perspectives.append({
            "name": "contrarian",
            "thesis": "Lettura contrarian: il match potrebbe sembrare più sbilanciato nei numeri aggregati di quanto non sia davvero sul campo.",
            "evidence": [
                f"Concessione tiri in porta: {home_name} {home_sota:.2f}, {away_name} {away_sota:.2f}.",
                f"Over12 resta al {float(over12):.2f}%, quindi non è un setup estremo.",
                "Il profilo corner può restare alto anche senza tradursi in superiorità netta sul risultato."
            ],
            "risk_flags": [
                "Rischio di match più bloccato del previsto.",
                "Heuristic winner lean troppo sottile per diventare sentenza."
            ]
        })

    if "live_sensitive" in requested:
        analysis_perspectives.append({
            "name": "live_sensitive",
            "thesis": "Lettura live-sensitive: i primi 15-20 minuti diranno subito se il volume atteso si sta materializzando.",
            "evidence": [
                "Osservare corner, tiri e tiri in porta nei primi 20 minuti.",
                "Se una squadra accumula subito pressione laterale, il profilo corner sale più del profilo esito.",
                "Se il ritmo parte basso, il pre-match va ridimensionato rapidamente."
            ],
            "risk_flags": [
                "Il live può divergere molto dal pre-match.",
                "Un gol precoce può alterare completamente il profilo corner."
            ]
        })

    h2h = ext.get("head_to_head_last5_summary") or []
    gaps = []
    if not h2h:
        gaps.append("head_to_head_recent_missing")
    if not ext.get("home_last5_summary"):
        gaps.append("home_last5_external_missing")
    if not ext.get("away_last5_summary"):
        gaps.append("away_last5_external_missing")
    if h_n < 10 or a_n < 10:
        gaps.append("recent_sample_small")

    final_take = {
        "most_supported_view": heuristic.get("winner_lean") or "balanced",
        "confidence_note": (
            f"Lean {heuristic.get('winner_lean')} con confidence {heuristic.get('winner_lean_confidence')}, "
            f"basato soprattutto su recent windows, shots, shots on target e corner profile."
        ),
        "data_gaps": gaps
    }

    return _json_safe({
        "match": match,
        "mode": mode,
        "summary": summary,
        "perspectives": analysis_perspectives,
        "final_take": final_take,
    })

def build_chat_prompt(context: dict[str, Any], mode: str, question: str, history: list[dict[str, str]] | None = None) -> str:
    history = history or []
    return f"""
Sei un analyst calcistico che risponde in italiano su UNA SOLA partita.

Regole:
- Usa tutto il contesto disponibile.
- Non spiegare la struttura JSON.
- Non dire mai "questo è un JSON object".
- Rispondi direttamente alla domanda dell'utente.
- Se la domanda è "chi vince?" o "chi ha più probabilità?", rispondi ESATTAMENTE così:

Lean:
Why:
Confidence:
Limits:

- Anche se i dati sono imperfetti, dai comunque il lean più supportato.
- Non inventare facts.

Mode: {mode}

History:
{json.dumps(history, ensure_ascii=False)}

Question:
{question}

Context:
{json.dumps(context, ensure_ascii=False)}
""".strip()

def ollama_generate_text(model: str, prompt: str) -> str:
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "top_p": 0.9},
        },
        timeout=OLLAMA_TIMEOUT,
    )
    r.raise_for_status()
    payload = r.json()
    return (payload.get("response") or "").strip()

def analyze_fixture_with_ollama(db, fixture_id: int, mode: str = "prematch", perspectives: list[str] | None = None, model: str | None = None):
    dataset = get_fixture_dataset(db, fixture_id)
    if not dataset.get("ok"):
        return dataset
    return _json_safe({
        "ok": True,
        "fixture_id": fixture_id,
        "mode": mode,
        "ollama_model": "deterministic-analysis",
        "analysis": deterministic_analysis(dataset, mode, perspectives),
    })

def chat_fixture_with_ollama(db, fixture_id: int, question: str, mode: str = "prematch", model: str | None = None, history: list[dict[str, str]] | None = None):
    dataset = get_fixture_dataset(db, fixture_id)
    if not dataset.get("ok"):
        return dataset
    context = build_llm_context(dataset, mode=mode, include_external=True)
    chosen_model = choose_ollama_model(model)
    prompt = build_chat_prompt(context, mode, question, history)
    answer = ollama_generate_text(chosen_model, prompt)
    return _json_safe({
        "ok": True,
        "fixture_id": fixture_id,
        "mode": mode,
        "ollama_model": chosen_model,
        "answer": answer,
    })
