"""Species lookup table built only from the competition data.

Player 1 teams and Player 2 leads list base stats and types for every
species they contain. Collecting them gives a small Pokedex that lets us
describe the Player 2 Pokemon revealed during the timeline.
"""

from ...utility.config.config_constants import STATS


####################################################################################
# TYPES
####################################################################################

def clean_types(types):
    """Remove the "notype" placeholder used for single-typed Pokemon."""
    return [t for t in types if t and t != "notype"]


####################################################################################
# BASE STATS
####################################################################################

def base_total(pokemon):
    """Return the sum of the six base stats of a Pokemon."""
    return sum(pokemon[f"base_{s}"] for s in STATS)


####################################################################################
# POKEDEX
####################################################################################

class Pokedex:
    """Map species name -> base stats and types."""

    #---------------------------
    # CREATE: an empty Pokedex
    #---------------------------

    def __init__(self):
        self._entries = {}

    #---------------------------
    # FILL: read the battles and save one entry for every species
    #---------------------------

    @classmethod
    def from_battles(cls, *battle_lists):
        """Build the Pokedex from one or more lists of battles."""
        # Start from an empty Pokedex
        pokedex = cls()

        # Go through every battle of every list we received (for us: only the train)
        for battles in battle_lists:
            for battle in battles:
                # The 6 Pokemon of Player 1 and the lead of Player 2 show their stats and types
                for pokemon in battle["p1_team_details"]:
                    pokedex.add(pokemon)
                pokedex.add(battle["p2_lead_details"])
        return pokedex

    def add(self, pokemon):
        """Store the base stats and types of a Pokemon if not already known."""
        # A species always has the same stats, so the first time we see it is enough
        name = pokemon["name"]
        if name in self._entries:
            return

        # The entry of the species: its 6 base stats and its types
        entry = {f"base_{s}": pokemon[f"base_{s}"] for s in STATS}
        entry["types"] = clean_types(pokemon["types"])
        self._entries[name] = entry

    #---------------------------
    # CONSULT: look up a species
    #---------------------------

    def get(self, name):
        """Return the entry of a species, or None if it was never seen."""
        return self._entries.get(name)

    def types(self, name):
        """Return the list of types of a species (empty if unknown)."""
        entry = self.get(name)
        return entry["types"] if entry else []

    def stat(self, name, stat, default=0.0):
        """Return one base stat of a species, or a default if unknown."""
        entry = self.get(name)
        return entry[f"base_{stat}"] if entry else default
