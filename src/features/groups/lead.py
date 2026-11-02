"""Static features comparing the lead of Player 2 with the lead of Player 1."""

from ... import config
from ..core.pokedex import base_total


####################################################################################
# LEAD: who opens the battle better
####################################################################################

def lead_features(battle):
    """Describe Player 2's lead and compare it with Player 1's first team slot."""
    # The only Pokemon of Player 2 that we know before the battle, and the one Player 1 opens with
    p2_lead = battle["p2_lead_details"]
    p1_lead = battle["p1_team_details"][0]
    feats = {}

    #---------------------------
    # STATS: the lead of Player 2 and the difference with the lead of Player 1 (12 features)
    #---------------------------

    for stat in config.STATS:
        feats[f"p2_lead_{stat}"] = p2_lead[f"base_{stat}"]
        # Positive = the lead of Player 1 is better in this stat
        feats[f"lead_{stat}_diff"] = p1_lead[f"base_{stat}"] - p2_lead[f"base_{stat}"]

    #---------------------------
    # TOTAL AND LEVEL (3 features)
    #---------------------------

    feats["p2_lead_total"] = base_total(p2_lead)
    feats["lead_total_diff"] = base_total(p1_lead) - base_total(p2_lead)
    feats["lead_level_diff"] = p1_lead["level"] - p2_lead["level"]
    return feats
