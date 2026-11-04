"""Turn a list of battles into a feature matrix."""

import pandas as pd

from ...utility.config.config_constants import ID_COLUMN, TARGET
from ..groups.bags import BagOfTokens
from ..groups.endstate import EndState
from ..groups.matchup import matchup_features
from ..groups.lead import lead_features
from ..groups.team import team_features
from ..groups.timeline import timeline_features


####################################################################################
# FEATURE BUILDER
####################################################################################

class FeatureBuilder:
    """Combine all feature groups into one row per battle.

    Each group can be switched off to measure its contribution (ablation).
    Call fit() on the training battles before transform().
    """

    GROUPS = ("team", "lead", "matchup", "timeline", "bags", "endstate")

    #---------------------------
    # CREATE: keep the tools and choose the groups
    #---------------------------

    def __init__(self, pokedex, type_chart, groups=None, bag_min_count=20):
        # The two lookup tables used by the feature groups
        self.pokedex = pokedex
        self.type_chart = type_chart

        # The groups to compute: all of them unless a list is given (ablation)
        self.groups = tuple(groups) if groups else self.GROUPS

        # The groups that still have to learn from the training battles: fit() will do it
        self.bags = BagOfTokens(min_count=bag_min_count)
        self.endstate = EndState()

    #---------------------------
    # FIT: learn from the training battles
    #---------------------------

    def fit(self, battles):
        """Learn the parts that depend on the data (the bag and species vocabularies)."""
        if "bags" in self.groups:
            self.bags.fit(battles)
        if "endstate" in self.groups:
            self.endstate.fit(battles)
        return self

    #---------------------------
    # TRANSFORM: from the battles to the table
    #---------------------------

    def battle_features(self, battle):
        """Return the dictionary of features for a single battle."""
        # Start from an empty dictionary and add the features of each group that is switched on
        feats = {}
        if "team" in self.groups:
            feats.update(team_features(battle))
        if "lead" in self.groups:
            feats.update(lead_features(battle))
        if "matchup" in self.groups:
            feats.update(matchup_features(battle, self.type_chart))
        if "timeline" in self.groups:
            feats.update(timeline_features(battle, self.pokedex, self.type_chart))
        if "bags" in self.groups:
            feats.update(self.bags.features(battle))
        if "endstate" in self.groups:
            feats.update(self.endstate.features(battle, self.pokedex))
        return feats

    def transform(self, battles):
        """Return a DataFrame indexed by battle_id with one column per feature."""
        # One dictionary of features per battle, each with its battle_id
        rows = []
        for battle in battles:
            feats = self.battle_features(battle)
            feats[ID_COLUMN] = battle[ID_COLUMN]
            rows.append(feats)

        # The list of dictionaries becomes a table: one row per battle, one column per feature
        return pd.DataFrame(rows).set_index(ID_COLUMN).fillna(0.0)


####################################################################################
# TARGET
####################################################################################

def extract_target(battles):
    """Return the player_won labels as a Series indexed by battle_id."""
    return pd.Series(
        [int(b[TARGET]) for b in battles],
        index=[b[ID_COLUMN] for b in battles],
        name=TARGET,
    )
