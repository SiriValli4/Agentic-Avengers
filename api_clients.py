"""
api_clients.py — External API helpers for the Trip Planner AI Agent

Covers:
  - Nominatim (OpenStreetMap geocoding)
  - Overpass API (live Points of Interest)

Both services are free and require a descriptive User-Agent header
per OpenStreetMap's Acceptable Use Policy.
"""

import time
import requests

# ── Shared config ─────────────────────────────────────────────────────────────

# IMPORTANT: OpenStreetMap requires a meaningful User-Agent that identifies
# your app and contact info. Using "python-requests" or similar is not allowed.
USER_AGENT = "TripPlannerAI/1.0 (Infosys Springboard Capstone; contact: student@example.com)"

NOMINATIM_BASE = "https://nominatim.openstreetmap.org"
OVERPASS_BASE  = "https://overpass-api.de/api/interpreter"

HEADERS = {"User-Agent": USER_AGENT}

# ── Nominatim helpers ─────────────────────────────────────────────────────────

def geocode_city(city: str) -> dict | None:
    """
    Convert a city name to (lat, lon) coordinates using Nominatim.

    Returns a dict with keys: display_name, lat, lon
    Returns None if not found or on error.
    """
    params = {
        "q": city,
        "format": "json",
        "limit": 1,
    }
    try:
        response = requests.get(
            f"{NOMINATIM_BASE}/search",
            params=params,
            headers=HEADERS,
            timeout=10,
        )
        response.raise_for_status()
        results = response.json()
        if not results:
            return None
        r = results[0]
        return {
            "display_name": r["display_name"],
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
        }
    except requests.RequestException as e:
        print(f"[Nominatim] Error: {e}")
        return None


def test_nominatim() -> tuple[bool, str]:
    """Quick connectivity check — geocodes 'Paris' and returns (ok, message)."""
    result = geocode_city("Paris, France")
    if result:
        return True, f"Connected (Paris → {result['lat']:.4f}, {result['lon']:.4f})"
    return False, "Could not reach Nominatim"


# ── Overpass helpers ──────────────────────────────────────────────────────────

# Category → Overpass tag mapping
POI_CATEGORIES = {
    "restaurant":   'amenity"="restaurant',
    "cafe":         'amenity"="cafe',
    "museum":       'tourism"="museum',
    "hotel":        'tourism"="hotel',
    "attraction":   'tourism"="attraction',
    "park":         'leisure"="park',
    "viewpoint":    'tourism"="viewpoint',
}


def search_pois(
    lat: float,
    lon: float,
    category: str = "attraction",
    radius_m: int = 2000,
    limit: int = 20,
) -> list[dict]:
    """
    Search for Points of Interest near (lat, lon) using the Overpass API.

    Args:
        lat, lon    : Centre coordinates
        category    : One of the keys in POI_CATEGORIES
        radius_m    : Search radius in metres (default 2 km)
        limit       : Max results to return

    Returns a list of dicts with keys: name, lat, lon, tags
    """
    tag = POI_CATEGORIES.get(category, 'tourism"="attraction')

    # Overpass QL query — searches nodes, ways, and relations
    query = f"""
    [out:json][timeout:25];
    (
      node["{tag}](around:{radius_m},{lat},{lon});
      way["{tag}](around:{radius_m},{lat},{lon});
      relation["{tag}](around:{radius_m},{lat},{lon});
    );
    out center {limit};
    """

    try:
        response = requests.post(
            OVERPASS_BASE,
            data={"data": query},
            headers=HEADERS,
            timeout=30,
        )
        response.raise_for_status()
        elements = response.json().get("elements", [])

        pois = []
        for el in elements:
            # Ways/relations expose coords under "center"
            if el["type"] == "node":
                elat, elon = el.get("lat"), el.get("lon")
            else:
                center = el.get("center", {})
                elat, elon = center.get("lat"), center.get("lon")

            tags = el.get("tags", {})
            name = tags.get("name") or tags.get("name:en") or "Unnamed"

            if elat and elon:
                pois.append({
                    "name": name,
                    "lat": elat,
                    "lon": elon,
                    "tags": tags,
                })

        return pois

    except requests.RequestException as e:
        print(f"[Overpass] Error: {e}")
        return []


def test_overpass() -> tuple[bool, str]:
    """
    Quick connectivity check — searches for attractions near Paris city centre.
    Returns (ok, message).
    """
    # Paris approximate centre
    pois = search_pois(lat=48.8566, lon=2.3522, category="attraction", radius_m=1000, limit=3)
    if pois:
        return True, f"Connected ({len(pois)} attractions found near Paris)"
    # Empty result could mean no data, not necessarily a failure — try a basic HTTP check
    try:
        r = requests.get(OVERPASS_BASE, params={"data": "[out:json];node(1);out;"}, timeout=10)
        r.raise_for_status()
        return True, "Connected (no nearby results but API reachable)"
    except Exception as e:
        return False, f"Could not reach Overpass: {e}"
