import os
import requests

BASE_URL = "https://v3.football.api-sports.io"

class ApiFootballClient:
    def __init__(self):
        key = os.getenv("APIFOOTBALL_KEY")
        if not key:
            raise RuntimeError("APIFOOTBALL_KEY mancante")
        self.headers = {"x-apisports-key": key}

    def get(self, path, params=None):
        url = f"{BASE_URL}{path}"
        r = requests.get(url, headers=self.headers, params=params or {}, timeout=30)
        r.raise_for_status()
        return r.json()

    def status(self):
        return self.get("/status")

    def leagues(self, search=None):
        params = {}
        if search:
            params["search"] = search
        return self.get("/leagues", params)

    def fixtures(self, league, season):
        return self.get("/fixtures", {"league": league, "season": season})

    def fixture_statistics(self, fixture_id):
        return self.get("/fixtures/statistics", {"fixture": fixture_id})

    def fixture_events(self, fixture_id):
        return self.get("/fixtures/events", {"fixture": fixture_id})

    def fixture_lineups(self, fixture_id):
        return self.get("/fixtures/lineups", {"fixture": fixture_id})

    def fixture_players(self, fixture_id):
        return self.get("/fixtures/players", {"fixture": fixture_id})

    def predictions(self, fixture_id):
        return self.get("/predictions", {"fixture": fixture_id})
