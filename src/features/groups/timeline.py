"""Dynamic features extracted from the first 30 turns of a battle."""

import numpy as np

from ... import config

SIDES = ("p1", "p2")
SNAPSHOT_TURNS = (10, 20)
TRACKED_EFFECTS = ("substitute", "leechseed", "confusion", "trapped", "perish", "protect")

####################################################################################
# HELPER
####################################################################################

def opponent(side):
    """Return the other side ("p1" <-> "p2")."""
    return "p2" if side == "p1" else "p1"


####################################################################################
# SIDE STATE: the notebook of one player while we read the turns
####################################################################################

class SideState:
    """Running state of one player while walking through the timeline."""

    #---------------------------
    # CREATE: an empty notebook
    #---------------------------

    def __init__(self):
        self.hp = {}              # last known hp_pct of each revealed Pokemon
        self.status = {}          # last known status of each revealed Pokemon
        self.active = None        # name of the Pokemon currently on the field
        self.active_boosts = {}   # stat boosts of the active Pokemon
        self.first_faint_turn = None
        # The counters that grow turn after turn
        self.counts = {
            "hp_lost": 0.0, 
            "hp_healed": 0.0, 
            "switches": 0, 
            "no_move": 0,
            "physical": 0, 
            "special": 0, 
            "status_moves": 0, 
            "power_sum": 0.0,
            "super_effective": 0, 
            "resisted": 0, 
            "immune": 0, 
            "stab": 0,
            "priority": 0, 
            "status_inflicted": 0, 
            "boost_turns": 0,
        }
        # One counter for each tracked effect: how many turns the Pokemon on the field had it
        for effect in TRACKED_EFFECTS:
            self.counts[f"effect_{effect}"] = 0

    #---------------------------
    # READ: questions we can ask the notebook at the end
    #---------------------------

    def fainted(self):
        """Number of revealed Pokemon that have fainted."""
        return sum(s == "fnt" for s in self.status.values())

    def hp_remaining(self):
        """Total HP fraction left, counting unrevealed Pokemon at full health."""
        # A Pokemon we never saw is probably still in perfect shape: it counts as 1.0
        unrevealed = config.TEAM_SIZE - len(self.hp)
        return sum(self.hp.values()) + max(unrevealed, 0)

    #---------------------------
    # UPDATE: write in the notebook what happened in one turn
    #---------------------------

    def update_pokemon(self, state, turn):
        """Apply the observed state of the active Pokemon for this turn."""
        
        name = state["name"]
        # HP: compare with the last known value (1.0 the first time we see the Pokemon)
        prev_hp = self.hp.get(name, 1.0)
        new_hp = float(state["hp_pct"])
        if new_hp < prev_hp:
            self.counts["hp_lost"] += prev_hp - new_hp
        else:
            self.counts["hp_healed"] += new_hp - prev_hp
        self.hp[name] = new_hp

        # Status: count it only when the Pokemon goes from "nostatus" to a real status
        prev_status = self.status.get(name, "nostatus")
        new_status = state["status"]
        if prev_status == "nostatus" and new_status not in ("nostatus", "fnt"):
            self.counts["status_inflicted"] += 1
        # Remember the turn of the first KO
        if new_status == "fnt" and self.first_faint_turn is None:
            self.first_faint_turn = turn
        self.status[name] = new_status

        # Switch: a different Pokemon on the field than the turn before
        if self.active is not None and self.active != name:
            self.counts["switches"] += 1
        self.active = name

        # Boosts: count the turns with at least one positive boost
        self.active_boosts = state.get("boosts", {})
        if any(v > 0 for v in self.active_boosts.values()):
            self.counts["boost_turns"] += 1

        # Effects: count the turns with each tracked effect (confusion, substitute, ...)
        for effect in state.get("effects", []):
            if effect in TRACKED_EFFECTS:
                self.counts[f"effect_{effect}"] += 1


####################################################################################
# MOVES: what kind of attack was used
####################################################################################

def record_move(attacker, move, attacker_name, defender_name, pokedex, type_chart):
    """Update the move counters of the attacking side."""
    counts = attacker.counts

    # No move this turn (a voluntary switch, or the Pokemon could not act)
    if move is None:
        counts["no_move"] += 1
        return
    if move.get("priority", 0) > 0:
        counts["priority"] += 1

    # A status move does no direct damage: count it and stop
    if move["category"] == "status":
        counts["status_moves"] += 1
        return

    # A damaging move: physical or special, and its power
    counts[move["category"]] += 1
    counts["power_sum"] += move.get("base_power", 0) or 0

    # STAB: the move has the same type as the Pokemon that uses it
    if move["type"] in pokedex.types(attacker_name):
        counts["stab"] += 1

    # Effectiveness against the Pokemon in front (the type chart), if we know its types
    defender_types = pokedex.types(defender_name)
    if defender_types:
        mult = type_chart.multiplier(move["type"], defender_types)
        if mult == 0:
            counts["immune"] += 1
        elif mult > 1:
            counts["super_effective"] += 1
        elif mult < 1:
            counts["resisted"] += 1

####################################################################################
# STRENGTH LEFT: how much power the team has left
####################################################################################

def remaining_strength(side_state, team_totals, default_total):
    """Sum of base-stat totals of non-fainted Pokemon, weighted by their HP."""
    # Each revealed Pokemon: its remaining HP times its total strength
    strength = 0.0
    for name, hp in side_state.hp.items():
        strength += hp * team_totals.get(name, default_total)

    # The Pokemon never seen count as healthy, with an average strength
    unrevealed = config.TEAM_SIZE - len(side_state.hp)
    strength += max(unrevealed, 0) * default_total
    return strength

####################################################################################
# ACTIVE POKEMON: who is on the field at turn 30
####################################################################################

def active_features(states, pokedex, type_chart):
    """Compare the two Pokemon on the field at the end of the observed turns."""
    p1, p2 = states["p1"], states["p2"]
    feats = {}

    # State of the Pokemon on the field, for each side
    for side in SIDES:
        st = states[side]
        feats[f"{side}_active_hp"] = st.hp.get(st.active, 0.0)
        status = st.status.get(st.active, "nostatus")
        feats[f"{side}_active_statused"] = int(status not in ("nostatus", "fnt"))
        feats[f"{side}_active_boost_sum"] = sum(st.active_boosts.values())

    # The two Pokemon compared: type advantage, speed and total strength
    p1_types = pokedex.types(p1.active)
    p2_types = pokedex.types(p2.active)
    feats["active_matchup"] = type_chart.matchup_score(p1_types, p2_types)
    feats["active_speed_diff"] = pokedex.stat(p1.active, "spe") - pokedex.stat(p2.active, "spe")
    p1_total = sum(pokedex.stat(p1.active, s) for s in config.STATS)
    p2_total = sum(pokedex.stat(p2.active, s) for s in config.STATS)
    feats["active_total_diff"] = p1_total - p2_total
    return feats

####################################################################################
# DIFFERENCES: Player 1 minus Player 2
####################################################################################

def difference_features(feats):
    """Player 1 minus Player 2 versions of the most important counters."""
    keys = [
        "hp_remaining", "hp_lost", "hp_healed", "switches", "super_effective",
        "status_inflicted", "boost_turns", "strength_left", "mean_power", "stab",
        "alive_hp_mean", "revealed",
    ]
    diffs = {f"{k}_diff": feats[f"p1_{k}"] - feats[f"p2_{k}"] for k in keys}
    # Positive means Player 1 has knocked out more Pokemon than it has lost.
    diffs["fainted_diff"] = feats["p2_fainted"] - feats["p1_fainted"]
    diffs["statused_diff"] = (
        sum(feats[f"p2_end_{s}"] for s in config.STATUSES)
        - sum(feats[f"p1_end_{s}"] for s in config.STATUSES)
    )
    return diffs

####################################################################################
# TIMELINE: read the 30 turns and build all the features
####################################################################################

def timeline_features(battle, pokedex, type_chart):
    """Walk the observed turns and summarise how the battle is going."""

    #---------------------------
    # WALK THE TURNS: fill the two notebooks
    #---------------------------

    states = {side: SideState() for side in SIDES}
    snapshots = {}

    for turn in battle["battle_timeline"]:
        # First the Pokemon on the field of both sides...
        for side in SIDES:
            states[side].update_pokemon(turn[f"{side}_pokemon_state"], turn["turn"])
        # ...then the moves, so each move is compared with the Pokemon that is really in front
        for side in SIDES:
            record_move(
                states[side], turn.get(f"{side}_move_details"),
                states[side].active, states[opponent(side)].active,
                pokedex, type_chart,
            )
        # A "photo" of the battle at turns 10 and 20: KOs and HP left of each side
        if turn["turn"] in SNAPSHOT_TURNS:
            snapshots[turn["turn"]] = {
                side: (states[side].fainted(), states[side].hp_remaining()) for side in SIDES
            }

    #---------------------------
    # FEATURES OF EACH SIDE: read the notebooks (33 features each)
    #---------------------------

    n_turns = max(len(battle["battle_timeline"]), 1)
    feats = {}
    for side in SIDES:
        st = states[side]
        feats[f"{side}_fainted"] = st.fainted()
        feats[f"{side}_revealed"] = len(st.hp)
        feats[f"{side}_hp_remaining"] = st.hp_remaining()
        alive_hp = [hp for name, hp in st.hp.items() if st.status.get(name) != "fnt"]
        feats[f"{side}_alive_hp_mean"] = np.mean(alive_hp) if alive_hp else 0.0
        # How many Pokemon have each status at the end
        for status in config.STATUSES:
            feats[f"{side}_end_{status}"] = sum(s == status for s in st.status.values())
        # No KO at all: the first KO "happens" after the last turn
        feats[f"{side}_first_faint_turn"] = st.first_faint_turn or (n_turns + 1)
        # The counters of the notebook
        for key, value in st.counts.items():
            feats[f"{side}_{key}"] = value
        damaging = st.counts["physical"] + st.counts["special"]
        feats[f"{side}_mean_power"] = st.counts["power_sum"] / damaging if damaging else 0.0

    #---------------------------
    # STRENGTH LEFT (2 features)
    #---------------------------

    # Base-stat strength left on each side (Player 2's hidden members use the
    # average of Player 1's team as a neutral guess).
    p1_totals = {p["name"]: sum(p[f"base_{s}"] for s in config.STATS) for p in battle["p1_team_details"]}
    default_total = float(np.mean(list(p1_totals.values())))
    p2_totals = {
        name: sum(pokedex.stat(name, s, default_total / 6) for s in config.STATS)
        for name in states["p2"].hp
    }
    feats["p1_strength_left"] = remaining_strength(states["p1"], p1_totals, default_total)
    feats["p2_strength_left"] = remaining_strength(states["p2"], p2_totals, default_total)

    #---------------------------
    # SNAPSHOTS (4 features)
    #---------------------------

    for turn_no, snap in snapshots.items():
        feats[f"fainted_diff_t{turn_no}"] = snap["p2"][0] - snap["p1"][0]
        feats[f"hp_diff_t{turn_no}"] = snap["p1"][1] - snap["p2"][1]

    #---------------------------
    # ACTIVE POKEMON AND DIFFERENCES (9 + 14 features)
    #---------------------------

    feats.update(active_features(states, pokedex, type_chart))
    feats.update(difference_features(feats))
    return feats
