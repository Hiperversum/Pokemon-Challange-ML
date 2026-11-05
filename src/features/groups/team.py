"""Static features describing the six Pokemon of Player 1's team."""

import numpy as np

from ...utility.config.config_constants import STATS
from ..core.pokedex import base_total, clean_types


####################################################################################
# TEAM: how strong is Player 1
####################################################################################

def team_features(battle):
    """Aggregate base stats, items and types of Player 1's six Pokemon."""
    team = battle["p1_team_details"]
    feats = {}

    #---------------------------
    # STATS: mean, max and min of each of the 6 base stats (18 features)
    #---------------------------

    for stat in STATS:
        # The value of this stat for each of the 6 Pokemon of the team
        values = np.array([p[f"base_{stat}"] for p in team], dtype=float)
        feats[f"p1_team_{stat}_mean"] = values.mean()
        feats[f"p1_team_{stat}_max"] = values.max()
        feats[f"p1_team_{stat}_min"] = values.min()

    #---------------------------
    # TOTALS: the sum of the 6 stats of each Pokemon (3 features)
    #---------------------------

    totals = np.array([base_total(p) for p in team], dtype=float)
    feats["p1_team_total_mean"] = totals.mean()
    feats["p1_team_total_min"] = totals.min()
    # How different the 6 Pokemon are from each other (small = a team of similar Pokemon)
    feats["p1_team_total_std"] = totals.std()

    #---------------------------
    # LEVELS (2 features)
    #---------------------------

    levels = np.array([p["level"] for p in team], dtype=float)
    feats["p1_team_level_mean"] = levels.mean()
    feats["p1_team_low_level"] = int((levels < 100).sum())

    #---------------------------
    # ITEMS (2 features)
    #---------------------------

    # A Pokemon without item gets the placeholder "noitem", so there are no empty values
    items = [p.get("item") or "noitem" for p in team]
    feats["p1_team_leftovers"] = items.count("leftovers")
    feats["p1_team_no_item"] = sum(i in ("noitem", "", "none") for i in items)

    #---------------------------
    # TYPES (1 feature)
    #---------------------------

    # All the types of the team in one list, then the repeated ones are removed with set()
    all_types = [t for p in team for t in clean_types(p["types"])]
    feats["p1_team_distinct_types"] = len(set(all_types))
    return feats

