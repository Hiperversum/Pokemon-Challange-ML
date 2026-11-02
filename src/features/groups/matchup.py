"""Type matchup features between Player 1's team and Player 2's lead."""

import numpy as np

from ..core.pokedex import clean_types


####################################################################################
# MATCHUP: do the types of Player 1 favour it against the lead of Player 2
####################################################################################

def matchup_features(battle, type_chart):
    """Measure how well each Player 1 Pokemon matches up against Player 2's lead.

    Uses the Pokemon's own types as a proxy for their attacking moves (STAB).
    """
    # The types of the only Pokemon of Player 2 that we know before the battle
    lead_types = clean_types(battle["p2_lead_details"]["types"])

    #---------------------------
    # ONE VALUE FOR EACH OF THE 6 POKEMON OF PLAYER 1
    #---------------------------

    offense, defense = [], []
    for pokemon in battle["p1_team_details"]:
        own_types = clean_types(pokemon["types"])
        # How hard this Pokemon can hit the lead of Player 2 (with its best type)
        offense.append(type_chart.best_multiplier(own_types, lead_types))
        # How hard the lead of Player 2 can hit this Pokemon (with its best type)
        defense.append(type_chart.best_multiplier(lead_types, own_types))

    offense = np.array(offense)
    defense = np.array(defense)

    #---------------------------
    # THE 6 FEATURES: summarise the two lists of 6 values
    #---------------------------

    return {
        "p1_vs_lead_offense_mean": offense.mean(),
        "p1_vs_lead_offense_max": offense.max(),
        # How many Pokemon of Player 1 hit the lead with a super effective (more than x1) attack
        "p1_vs_lead_super_effective": int((offense > 1).sum()),
        "lead_vs_p1_offense_mean": defense.mean(),
        # How many Pokemon of Player 1 are hit by the lead with a super effective attack
        "lead_vs_p1_super_effective": int((defense > 1).sum()),
        # Hit given minus hit taken, averaged over the 6 Pokemon: positive = types favour Player 1
        "p1_vs_lead_matchup": (offense - defense).mean(),
    }
