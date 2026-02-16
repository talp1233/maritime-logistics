"""
Core constants for the Greek Maritime Intelligence Platform.

Beaufort scale conversions, known regulatory thresholds,
route definitions, and vessel classifications.
"""

# ---------------------------------------------------------------------------
# Beaufort Scale — official WMO conversion table
# Maps Beaufort number -> (min_knots, max_knots, description)
# ---------------------------------------------------------------------------
BEAUFORT_SCALE = {
    0:  (0,   1,   "Calm"),
    1:  (1,   3,   "Light air"),
    2:  (4,   6,   "Light breeze"),
    3:  (7,   10,  "Gentle breeze"),
    4:  (11,  16,  "Moderate breeze"),
    5:  (17,  21,  "Fresh breeze"),
    6:  (22,  27,  "Strong breeze"),
    7:  (28,  33,  "Near gale"),
    8:  (34,  40,  "Gale"),
    9:  (41,  47,  "Strong gale"),
    10: (48,  55,  "Storm"),
    11: (56,  63,  "Violent storm"),
    12: (64,  999, "Hurricane force"),
}


def knots_to_beaufort(knots: float) -> int:
    """Convert wind speed in knots to Beaufort number."""
    result = 0
    for bf, (lo, hi, _) in BEAUFORT_SCALE.items():
        if knots >= lo:
            result = bf
    return result


def kmh_to_knots(kmh: float) -> float:
    """Convert km/h to knots."""
    return kmh / 1.852


def ms_to_knots(ms: float) -> float:
    """Convert m/s to knots."""
    return ms * 1.94384


# ---------------------------------------------------------------------------
# Greek Coast Guard sailing ban thresholds (απαγορευτικό απόπλου)
# Source: Hellenic Coast Guard standard practice
# ---------------------------------------------------------------------------
# These are the *general* thresholds. Actual decisions can vary by port
# authority and specific conditions.
SAILING_BAN_THRESHOLDS = {
    "high_speed":    6,   # Beaufort — high-speed craft (ταχύπλοα)
    "conventional":  8,   # Beaufort — conventional vessels
    "small_craft":   5,   # Beaufort — small open-deck vessels
}

# ---------------------------------------------------------------------------
# Vessel type classification
# ---------------------------------------------------------------------------
VESSEL_TYPES = {
    "HIGH_SPEED":    "high_speed",
    "CONVENTIONAL":  "conventional",
    "CATAMARAN":     "high_speed",     # catamarans follow high-speed rules
    "SMALL":         "small_craft",
}

# ---------------------------------------------------------------------------
# Key Greek ferry routes — (origin, destination, distance_nm, sea_area)
# Distances are approximate nautical miles.
# Sea areas correspond to Greek meteorological marine forecast zones.
# ---------------------------------------------------------------------------
ROUTES = {
    "PIR-SYR": {
        "origin": "Piraeus",
        "destination": "Syros",
        "distance_nm": 83,
        "sea_area": "Central Aegean",
        "exposed": True,          # open-sea crossing
    },
    "PIR-MYK": {
        "origin": "Piraeus",
        "destination": "Mykonos",
        "distance_nm": 94,
        "sea_area": "Central Aegean",
        "exposed": True,
    },
    "PIR-NAX": {
        "origin": "Piraeus",
        "destination": "Naxos",
        "distance_nm": 103,
        "sea_area": "Central Aegean",
        "exposed": True,
    },
    "PIR-SAN": {
        "origin": "Piraeus",
        "destination": "Santorini",
        "distance_nm": 128,
        "sea_area": "South Aegean",
        "exposed": True,
    },
    "PIR-HER": {
        "origin": "Piraeus",
        "destination": "Heraklion",
        "distance_nm": 174,
        "sea_area": "Cretan Sea",
        "exposed": True,
    },
    "PIR-CHN": {
        "origin": "Piraeus",
        "destination": "Chania",
        "distance_nm": 163,
        "sea_area": "Cretan Sea",
        "exposed": True,
    },
    "RAF-MYK": {
        "origin": "Rafina",
        "destination": "Mykonos",
        "distance_nm": 75,
        "sea_area": "Central Aegean",
        "exposed": True,
    },
    "RAF-AND": {
        "origin": "Rafina",
        "destination": "Andros",
        "distance_nm": 37,
        "sea_area": "Central Aegean",
        "exposed": False,         # relatively sheltered
    },
    "RAF-TIN": {
        "origin": "Rafina",
        "destination": "Tinos",
        "distance_nm": 65,
        "sea_area": "Central Aegean",
        "exposed": True,
    },
    "LAV-CHI": {
        "origin": "Lavrio",
        "destination": "Chios",
        "distance_nm": 120,
        "sea_area": "East Aegean",
        "exposed": True,
    },
    "PIR-MIT": {
        "origin": "Piraeus",
        "destination": "Milos",
        "distance_nm": 87,
        "sea_area": "West Cyclades",
        "exposed": True,
    },
}

# ---------------------------------------------------------------------------
# Key ports with metadata
# ---------------------------------------------------------------------------
PORTS = {
    "PIR": {
        "name": "Piraeus",
        "lat": 37.9475,
        "lon": 23.6372,
        "authority": "OLP S.A.",
        "type": "major",
    },
    "RAF": {
        "name": "Rafina",
        "lat": 38.0228,
        "lon": 24.0097,
        "authority": "Rafina Port Authority",
        "type": "secondary",
    },
    "HER": {
        "name": "Heraklion",
        "lat": 35.3387,
        "lon": 25.1442,
        "authority": "Heraklion Port Authority",
        "type": "major",
    },
    "SAN": {
        "name": "Santorini (Athinios)",
        "lat": 36.3932,
        "lon": 25.4254,
        "authority": "Santorini Port Authority",
        "type": "island",
    },
    "MYK": {
        "name": "Mykonos",
        "lat": 37.4467,
        "lon": 25.3289,
        "authority": "Mykonos Port Authority",
        "type": "island",
    },
    "NAX": {
        "name": "Naxos",
        "lat": 37.1066,
        "lon": 25.3756,
        "authority": "Naxos Port Authority",
        "type": "island",
    },
    "SYR": {
        "name": "Syros (Ermoupoli)",
        "lat": 37.4400,
        "lon": 24.9461,
        "authority": "Syros Port Authority",
        "type": "island",
    },
    "CHN": {
        "name": "Chania (Souda)",
        "lat": 35.4878,
        "lon": 24.0764,
        "authority": "Chania Port Authority",
        "type": "secondary",
    },
    "LAV": {
        "name": "Lavrio",
        "lat": 37.7133,
        "lon": 24.0561,
        "authority": "Lavrio Port Authority",
        "type": "secondary",
    },
    "AND": {
        "name": "Andros (Gavrio)",
        "lat": 37.8800,
        "lon": 24.7314,
        "authority": "Andros Port Authority",
        "type": "island",
    },
    "TIN": {
        "name": "Tinos",
        "lat": 37.5317,
        "lon": 25.1631,
        "authority": "Tinos Port Authority",
        "type": "island",
    },
    "CHI": {
        "name": "Chios",
        "lat": 38.3722,
        "lon": 26.1358,
        "authority": "Chios Port Authority",
        "type": "island",
    },
    "MIT": {
        "name": "Milos (Adamas)",
        "lat": 36.7261,
        "lon": 24.4417,
        "authority": "Milos Port Authority",
        "type": "island",
    },
}

# ---------------------------------------------------------------------------
# Major Greek ferry operators
# ---------------------------------------------------------------------------
OPERATORS = {
    "BLUESTAR": {
        "name": "Blue Star Ferries",
        "parent": "Attica Group",
        "fleet_type": "conventional",
        "key_routes": ["PIR-SYR", "PIR-NAX", "PIR-SAN", "PIR-HER"],
    },
    "HELLENIC": {
        "name": "Hellenic Seaways",
        "parent": "Attica Group",
        "fleet_type": "both",       # operates both conventional and high-speed
        "key_routes": ["PIR-MYK", "PIR-SAN", "RAF-MYK"],
    },
    "SEAJETS": {
        "name": "SeaJets",
        "parent": "SeaJets",
        "fleet_type": "high_speed",
        "key_routes": ["PIR-MYK", "PIR-SAN", "RAF-MYK", "PIR-NAX"],
    },
    "MINOAN": {
        "name": "Minoan Lines",
        "parent": "Grimaldi Group",
        "fleet_type": "conventional",
        "key_routes": ["PIR-HER"],
    },
    "ANEK": {
        "name": "ANEK Lines",
        "parent": "ANEK Lines",
        "fleet_type": "conventional",
        "key_routes": ["PIR-CHN", "PIR-HER"],
    },
    "GOLDENSTAR": {
        "name": "Golden Star Ferries",
        "parent": "Golden Star",
        "fleet_type": "both",
        "key_routes": ["RAF-MYK", "RAF-AND", "RAF-TIN"],
    },
    "FASTFERRIES": {
        "name": "Fast Ferries",
        "parent": "Fast Ferries",
        "fleet_type": "conventional",
        "key_routes": ["RAF-AND", "RAF-TIN"],
    },
}
