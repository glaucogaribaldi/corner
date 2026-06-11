import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

def init_db():
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS competitions (
            id SERIAL PRIMARY KEY,
            api_league_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            season INTEGER NOT NULL,
            coverage JSONB,
            UNIQUE(api_league_id, season)
        );
        """))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS matches (
            id SERIAL PRIMARY KEY,
            fixture_id INTEGER UNIQUE NOT NULL,
            api_league_id INTEGER,
            season INTEGER,
            fixture_date TIMESTAMP,
            status_short TEXT,
            status_long TEXT,
            home_team_id INTEGER,
            home_team_name TEXT,
            away_team_id INTEGER,
            away_team_name TEXT,
            home_goals INTEGER,
            away_goals INTEGER,
            raw JSONB
        );
        """))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS match_team_stats (
            id SERIAL PRIMARY KEY,
            fixture_id INTEGER NOT NULL,
            team_id INTEGER,
            team_name TEXT,
            shots_on_goal INTEGER,
            shots_off_goal INTEGER,
            total_shots INTEGER,
            blocked_shots INTEGER,
            shots_insidebox INTEGER,
            shots_outsidebox INTEGER,
            fouls INTEGER,
            corner_kicks INTEGER,
            offsides INTEGER,
            ball_possession INTEGER,
            yellow_cards INTEGER,
            red_cards INTEGER,
            goalkeeper_saves INTEGER,
            total_passes INTEGER,
            passes_accurate INTEGER,
            passes_percent INTEGER,
            raw JSONB,
            UNIQUE(fixture_id, team_id)
        );
        """))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            fixture_id INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            over12_corners FLOAT,
            over14_corners FLOAT,
            expected_corners FLOAT,
            shots_range_low INTEGER,
            shots_range_high INTEGER,
            sot_range_low INTEGER,
            sot_range_high INTEGER,
            thermometer TEXT,
            explanation TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS ingest_runs (
            id SERIAL PRIMARY KEY,
            api_league_id INTEGER,
            season INTEGER,
            fixtures_found INTEGER,
            fixtures_ingested INTEGER,
            stats_ingested INTEGER,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """))
