"""Turn a list of battles into a feature matrix."""

import pandas as pd

from ... import config
from ..groups.bags import BagOfTokens
from ..groups.endstate import EndState
from ..groups.simulation import BattleSimulation
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

    GROUPS = ("team", "lead", "matchup", "timeline", "bags", "endstate", "simulation")

    #---------------------------
    # CREATE: keep the tools and choose the groups
    #---------------------------

    def __init__(self, pokedex, type_chart, groups=None, bag_min_count=20):
        # The two lookup tables used by the feature groups
        self.pokedex = pokedex
        self.type_chart = type_chart

        # The groups to compute: all of them unless a list is given (ablation)
        self.groups = tuple(groups) if groups else self.GROUPS

        # The bags still have no vocabulary: fit() will learn it
        self.bags = BagOfTokens(min_count=bag_min_count)
        # The species vocabulary of the end-state group is also learned in fit()
        self.endstate = EndState()
        # The simulation learns species stats, types and usual attacks in fit()
        self.simulation = BattleSimulation(type_chart)

    #---------------------------
    # FIT: learn from the training battles
    #---------------------------

    def fit(self, battles):
        """Learn the parts that depend on the data (the bag and species vocabularies)."""
        if "bags" in self.groups:
            self.bags.fit(battles)
        if "endstate" in self.groups:
            self.endstate.fit(battles)
        if "simulation" in self.groups:
            self.simulation.fit(battles)
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
        if "simulation" in self.groups:
            feats.update(self.simulation.features(battle))
        return feats

    def transform(self, battles):
        """Return a DataFrame indexed by battle_id with one column per feature."""
        # One dictionary of features per battle, each with its battle_id
        rows = []
        for battle in battles:
            feats = self.battle_features(battle)
            feats[config.ID_COLUMN] = battle[config.ID_COLUMN]
            rows.append(feats)

        # The list of dictionaries becomes a table: one row per battle, one column per feature
        return pd.DataFrame(rows).set_index(config.ID_COLUMN).fillna(0.0)


####################################################################################
# TARGET
####################################################################################

def extract_target(battles):
    """Return the player_won labels as a Series indexed by battle_id."""
    return pd.Series(
        [int(b[config.TARGET]) for b in battles],
        index=[b[config.ID_COLUMN] for b in battles],
        name=config.TARGET,
    )
