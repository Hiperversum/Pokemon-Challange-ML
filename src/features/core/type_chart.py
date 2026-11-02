"""Type effectiveness lookups based on the supplied Gen 2 type chart."""


####################################################################################
# TYPE CHART
####################################################################################

class TypeChart:
    """Damage multipliers between attacking and defending types."""

    #---------------------------
    # CREATE: turn the csv table into a dictionary
    #---------------------------

    def __init__(self, chart_df):
        # {("fire", "bug"): 2.0, ("normal", "ghost"): 0.0, ...}: one entry for each row of the csv
        self._multipliers = {
            (row.attacking_type, row.defending_type): float(row.multiplier)
            for row in chart_df.itertuples(index=False)
        }

    #---------------------------
    # CONSULT: one attack type against the defending types
    #---------------------------

    def multiplier(self, attack_type, defend_types):
        """Return the multiplier of an attack type against a list of defending types.

        Pairs missing from the chart are neutral (1.0).
        """
        # With two defending types the two multipliers are multiplied: fire vs bug/steel = 2 * 2 = 4
        result = 1.0
        for defend_type in defend_types:
            result *= self._multipliers.get((attack_type, defend_type), 1.0)
        return result

    def best_multiplier(self, attack_types, defend_types):
        """Return the best multiplier among several attack types (e.g. a Pokemon's own types)."""
        if not attack_types:
            return 1.0
        # A Pokemon can have two types: we assume it attacks with the better one
        return max(self.multiplier(t, defend_types) for t in attack_types)

    #---------------------------
    # CONSULT: who is favoured in a matchup
    #---------------------------

    def matchup_score(self, types_a, types_b):
        """Return how much A is favoured over B: best(A->B) minus best(B->A)."""
        # Positive = A hits B harder than B hits A; negative = the opposite; 0 = balanced
        return self.best_multiplier(types_a, types_b) - self.best_multiplier(types_b, types_a)
