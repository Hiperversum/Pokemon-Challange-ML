"""Bag-of-tokens features: which species are involved and which moves are used.

Each battle becomes a vector of counts over a vocabulary (as in a
bag-of-words). The vocabulary is learned on the training battles only and
rare tokens are dropped to limit overfitting.
"""

from collections import Counter


####################################################################################
# TOKENS: the "words" of a battle
####################################################################################

def battle_tokens(battle):
    """Return the token counts of a battle (species and moves per player)."""
    tokens = Counter()

    #---------------------------
    # SPECIES OF PLAYER 1: the 6 Pokemon of its team
    #---------------------------

    for pokemon in battle["p1_team_details"]:
        tokens[f"p1_species_{pokemon['name']}"] += 1

    #---------------------------
    # SPECIES OF PLAYER 2: only the ones that entered the field in the 30 turns
    #---------------------------

    # A set keeps each name once, even if the Pokemon was on the field for many turns
    p2_revealed = {turn["p2_pokemon_state"]["name"] for turn in battle["battle_timeline"]}
    for name in p2_revealed:
        tokens[f"p2_species_{name}"] += 1

    #---------------------------
    # MOVES: how many times each player used each move
    #---------------------------

    for turn in battle["battle_timeline"]:
        for side in ("p1", "p2"):
            move = turn.get(f"{side}_move_details")
            # A turn without move (a switch, for example) is None and has no token
            if move:
                tokens[f"{side}_move_{move['name']}"] += 1
    return tokens

####################################################################################
# BAG OF TOKENS: the vocabulary and the counts
####################################################################################

class BagOfTokens:
    """Learn a token vocabulary on training battles and count tokens per battle."""

    #---------------------------
    # CREATE: the rule and an empty vocabulary
    #---------------------------

    def __init__(self, min_count=20):
        # A token is kept only if it appears in at least this many battles
        self.min_count = min_count
        # The list of tokens kept: empty until fit() is called
        self.vocabulary = []

    #---------------------------
    # FIT: learn the vocabulary from the training battles
    #---------------------------

    def fit(self, battles):
        """Keep the tokens that appear in at least min_count training battles."""
        # In how many battles does each token appear (a token counts once per battle)
        document_freq = Counter()
        for battle in battles:
            document_freq.update(battle_tokens(battle).keys())

        # Keep the frequent ones, in alphabetical order so the columns are always in the same order
        self.vocabulary = sorted(t for t, c in document_freq.items() if c >= self.min_count)
        return self

    #---------------------------
    # FEATURES: count the vocabulary tokens in one battle
    #---------------------------

    def features(self, battle):
        """Return the counts of the vocabulary tokens for one battle."""
        tokens = battle_tokens(battle)
        # One feature for each token of the vocabulary: its count, or 0 if it is not in this battle
        return {f"bag_{t}": tokens.get(t, 0) for t in self.vocabulary}
