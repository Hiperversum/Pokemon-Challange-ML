"""End-state features: a detailed picture of the battle after the 30 observed turns.

The timeline group already gives KOs, HP and statuses at turn 30. This group adds what it misses:
- how healthy the Pokemon still alive are, and which ones are asleep, frozen or almost KO;
- the Player 2 Pokemon resting on the bench with low HP: Player 2 faints are recorded much
  less often than Player 1 faints (0.5 vs 1.7 per battle), and those Pokemon behave like hidden KOs;
- how the HP difference changed during the 30 turns (trend, last turns);
- boosts and effects of the Pokemon on the field, spikes;
- the HP left of each frequent species (a healthy Snorlax is worth more than another healthy Pokemon).
"""

from collections import Counter

import numpy as np

from ...utility.config.config_constants import TEAM_SIZE

SIDES = ("p1", "p2")
END_EFFECTS = ("substitute", "leechseed", "confusion", "perish", "trapped")
BENCH_THRESHOLDS = (0.2, 0.35, 0.5)
# HP of a Pokemon with a status is worth a bit less: a frozen Pokemon is almost lost (Gen 2)
STATUS_WEIGHT = {"frz": 0.4, "slp": 0.8, "par": 0.9, "tox": 0.9}
# A benched Player 2 Pokemon at or below this HP is treated as a probable hidden KO
HIDDEN_KO_HP = 0.35
# Spikes take 1/8 of the HP of the Pokemon that comes in
SPIKES_DAMAGE = 0.13


####################################################################################
# HELPER
####################################################################################

def opponent(side):
    """Return the other side ("p1" <-> "p2")."""
    return "p2" if side == "p1" else "p1"


####################################################################################
# REPLAY: walk the 30 turns and remember the last state of every Pokemon
####################################################################################

def replay(battle):
    """Return the last known HP/status of every Pokemon and a few running records."""
    # Player 1's team is fully known (unseen = full HP); Player 2 only through what we see
    hp = {"p1": {p["name"]: 1.0 for p in battle["p1_team_details"]}, "p2": {}}
    status = {"p1": {p["name"]: "nostatus" for p in battle["p1_team_details"]}, "p2": {}}
    last = {}                     # last state (name, boosts, effects) of the active Pokemon of each side
    spikes = {"p1": 0, "p2": 0}   # 1 if this side has spikes laid on the opponent's field
    history = []                  # (HP difference, KO difference, p1 HP, p2 HP) after each turn

    for turn in battle["battle_timeline"]:
        # The state of the two Pokemon on the field and the moves of this turn
        for side in SIDES:
            state = turn[f"{side}_pokemon_state"]
            hp[side][state["name"]] = state["hp_pct"]
            status[side][state["name"]] = state["status"]
            last[side] = state
            move = turn.get(f"{side}_move_details")
            if move:
                if move["name"] == "spikes":
                    spikes[side] = 1
                # Rapid Spin removes the spikes from the field of the user
                if move["name"] == "rapidspin":
                    spikes[opponent(side)] = 0

        # The score after this turn: Player 2 Pokemon never seen count as full HP
        p1_hp = sum(hp["p1"].values())
        p2_hp = sum(hp["p2"].values()) + TEAM_SIZE - len(hp["p2"])
        p1_ko = sum(s == "fnt" for s in status["p1"].values())
        p2_ko = sum(s == "fnt" for s in status["p2"].values())
        history.append((p1_hp - p2_hp, p2_ko - p1_ko, p1_hp, p2_hp))

    return hp, status, last, spikes, np.array(history)


####################################################################################
# EXITS: how each benched Pokemon left the field for the last time (to find hidden KOs)
####################################################################################

def last_exits(battle, side):
    """For each benched, not fainted Pokemon of `side`: (last HP, kind of its last exit).

    A Player 2 faint is written almost only when a Player 1 Pokemon faints in the same turn:
    otherwise the row already shows the replacement and the fainted Pokemon just disappears.
    Kind of exit: did the opponent attack, was the newcomer hit (more than spikes), was the
    opponent hit, and how much HP the Pokemon had left (4 levels).
    """
    opp, timeline = opponent(side), battle["battle_timeline"]
    names = [turn[f"{side}_pokemon_state"]["name"] for turn in timeline]
    exits = {}
    for name, row in {n: i for i, n in enumerate(names)}.items():
        state = timeline[row][f"{side}_pokemon_state"]
        if state["status"] == "fnt" or row == len(timeline) - 1:
            continue
        newcomer = timeline[row + 1][f"{side}_pokemon_state"]
        newcomer_hp = next((t[f"{side}_pokemon_state"]["hp_pct"] for t in reversed(timeline[:row + 1])
                            if t[f"{side}_pokemon_state"]["name"] == newcomer["name"]), 1.0)
        old_opp, new_opp = timeline[row][f"{opp}_pokemon_state"], timeline[row + 1][f"{opp}_pokemon_state"]
        move = timeline[row + 1].get(f"{opp}_move_details")
        exits[name] = (state["hp_pct"], (
            bool(move) and move["category"] != "status",
            newcomer["hp_pct"] < newcomer_hp - SPIKES_DAMAGE,
            new_opp["name"] == old_opp["name"] and new_opp["hp_pct"] < old_opp["hp_pct"],
            min(int(state["hp_pct"] * 4), 3),
        ))
    return exits


####################################################################################
# END STATE
####################################################################################

class EndState:
    """Learn the species vocabulary on the training battles, then describe each battle's end."""

    #---------------------------
    # CREATE: the rule and an empty vocabulary
    #---------------------------

    def __init__(self, min_count=100):
        # A species gets its own "HP left" column only if it is seen in this many battles
        self.min_count = min_count
        self.species = []
        # Share of hidden KOs among the Player 2 exits of each kind (learned in fit)
        self.hidden_ko_share = {}

    #---------------------------
    # FIT: learn the frequent species from the training battles
    #---------------------------

    def fit(self, battles):
        """Keep the species that appear (in P1's team or on P2's field) in enough battles."""
        counts = Counter()
        for battle in battles:
            names = {p["name"] for p in battle["p1_team_details"]}
            names |= {t["p2_pokemon_state"]["name"] for t in battle["battle_timeline"]}
            counts.update(names)
        # Alphabetical order, so the columns are always in the same order
        self.species = sorted(n for n, c in counts.items() if c >= self.min_count)

        # Hidden KOs: Player 1 faints are always written, so Player 1 shows how often each kind of
        # exit happens to an alive Pokemon; the extra Player 2 exits of the same kind are hidden KOs
        exits = {side: Counter() for side in SIDES}
        for battle in battles:
            for side in SIDES:
                exits[side].update(kind for _, kind in last_exits(battle, side).values())
        self.hidden_ko_share = {kind: max(0.0, (n - exits["p1"][kind]) / n) for kind, n in exits["p2"].items()}
        return self

    #---------------------------
    # FEATURES: the state of one battle at turn 30
    #---------------------------

    def features(self, battle, pokedex):
        """Return the end-state features of one battle."""
        hp, status, last, spikes, history = replay(battle)
        feats = {}

        for side in SIDES:
            alive = [n for n in hp[side] if status[side][n] != "fnt"]
            active = last[side]["name"]

            # How many Player 2 Pokemon we saw alive (for Player 1 it copies the KOs)
            if side == "p2":
                feats["end_p2_alive_known"] = len(alive)

            # How healthy the alive Pokemon are
            feats[f"end_{side}_near_dead"] = sum(hp[side][n] <= 0.25 for n in alive)
            feats[f"end_{side}_healthy"] = sum(hp[side][n] > 0.5 and status[side][n] == "nostatus"
                                               for n in alive)
            # HP weighted by the status (only Player 2: for Player 1 it copies the HP left)
            if side == "p2":
                feats["end_p2_eff_hp"] = sum(hp["p2"][n] * STATUS_WEIGHT.get(status["p2"][n], 1.0)
                                             for n in alive)

            # Benched Pokemon (alive, not on the field) with low HP
            bench = [hp[side][n] for n in alive if n != active]
            for th in BENCH_THRESHOLDS:
                feats[f"end_{side}_bench_low{th}"] = sum(h <= th for h in bench)
            if side == "p2":
                feats["end_p2_bench_hp"] = sum(bench)

            # The Pokemon on the field: boosts and effects
            boosts = last[side]["boosts"].values()
            feats[f"end_{side}_boost_max"] = max(boosts)
            feats[f"end_{side}_boost_min"] = min(boosts)
            for effect in END_EFFECTS:
                feats[f"end_{side}_{effect}"] = int(effect in last[side]["effects"])
            feats[f"end_{side}_spikes_laid"] = spikes[side]

            # HP left of each frequent species (0 if fainted; small offset so "alive at 0 HP" != absent)
            for name in self.species:
                if name in hp[side]:
                    alive_hp = 0.0 if status[side][name] == "fnt" else hp[side][name] + 0.01
                    feats[f"species_{side}_{name}"] = alive_hp

        #---------------------------
        # PLAYER 1 vs PLAYER 2
        #---------------------------

        feats["end_near_dead_diff"] = feats["end_p1_near_dead"] - feats["end_p2_near_dead"]
        # Asleep or frozen: Pokemon that cannot act for a while
        disabled = {side: sum(status[side][n] in ("slp", "frz") for n in hp[side]) for side in SIDES}
        feats["end_disabled_diff"] = disabled["p1"] - disabled["p2"]

        # Player 2's HP without the probable hidden KOs (benched with low HP)
        unrevealed = TEAM_SIZE - len(hp["p2"])
        p2_active = last["p2"]["name"]
        p2_hp_adjusted = sum(
            h for n, h in hp["p2"].items()
            if status["p2"][n] != "fnt" and (h > HIDDEN_KO_HP or n == p2_active)
        ) + unrevealed
        feats["end_p2_hp_adjusted"] = p2_hp_adjusted
        feats["end_hp_adjusted_diff"] = sum(hp["p1"].values()) - p2_hp_adjusted

        # Expected hidden KOs: each benched Player 2 Pokemon is weighted by the share of hidden KOs
        # of its kind of exit
        hidden = {name: self.hidden_ko_share.get(kind, 0.0) for name, (_, kind) in last_exits(battle, "p2").items()}
        feats["hidden_ko_estimate"] = sum(hidden.values())
        feats["ko_diff_estimated"] = (sum(s == "fnt" for s in status["p2"].values()) + feats["hidden_ko_estimate"]
                                      - sum(s == "fnt" for s in status["p1"].values()))
        p2_hp_estimated = sum(h * (1 - hidden.get(n, 0.0)) for n, h in hp["p2"].items()
                              if status["p2"][n] != "fnt") + unrevealed
        feats["hp_diff_estimated"] = sum(hp["p1"].values()) - p2_hp_estimated

        #---------------------------
        # TREND OF THE HP DIFFERENCE DURING THE 30 TURNS
        #---------------------------

        diff = history[:, 0]
        feats["trend_hp_diff_mean"] = diff.mean()
        feats["trend_hp_diff_min"] = diff.min()
        feats["trend_hp_diff_max"] = diff.max()
        feats["trend_hp_diff_t25"] = diff[min(24, len(diff) - 1)]
        feats["trend_hp_diff_t28"] = diff[min(27, len(diff) - 1)]

        # Slope: is Player 1 gaining or losing ground (last 10 turns, all turns)
        last10 = diff[-10:]
        feats["trend_slope_last10"] = np.polyfit(np.arange(len(last10)), last10, 1)[0]
        feats["trend_slope_all"] = np.polyfit(np.arange(len(diff)), diff, 1)[0]

        # KOs and HP lost in the last 10 turns
        start = max(len(history) - 11, 0)
        feats["trend_ko_last10"] = history[-1, 1] - history[start, 1]
        feats["trend_p1_hp_lost_last10"] = history[start, 2] - history[-1, 2]
        feats["trend_p2_hp_lost_last10"] = history[start, 3] - history[-1, 3]

        #---------------------------
        # THE TWO POKEMON ON THE FIELD
        #---------------------------

        p1_active, p2_active = last["p1"]["name"], last["p2"]["name"]
        feats["end_active_hp_diff"] = hp["p1"][p1_active] - hp["p2"][p2_active]
        feats["end_active_p1_faster"] = int(pokedex.stat(p1_active, "spe") > pokedex.stat(p2_active, "spe"))
        return feats
