# Pokemon Battle Prediction - FDS Kaggle Challenge

Prediction of the winner of Gen 2 OU Pokemon battles from an incomplete
observation of the match (Player 1 team, Player 2 lead and the first 30 turns).
Project for *Fundamentals of Data Science*, Sapienza University of Rome, 2026-2027.

Competition: <https://www.kaggle.com/competitions/fds-pokemon-battles-prediction-2026>

Only classical machine learning models are used (no neural networks) and only
the competition data plus the supplied Gen 2 type chart.

## Repository structure

```
Pokemon-Challange-ML/
├── main.py                    entry point: python main.py (runs what config.json5 says)
├── config.json5               settings: what to run, commands, dataset, output, validation
├── data/                      competition files (not versioned, see below)
│   └── models/
│       ├── tuned_params.json  best hyperparameters found by Optuna (versioned)
│       └── ensembles.json     type and members of each ensemble (versioned)
├── notebooks/
│   └── kaggle_submission.ipynb  notebook run on Kaggle (clones this repo)
├── scripts/
│   ├── evaluation/
│   │   ├── compare_models.py    k-fold comparison of all base models
│   │   ├── compare_tuned.py     k-fold comparison of tuned models and ensembles
│   │   ├── feature_ablation.py  contribution of each feature group
│   │   └── tune_model.py        Optuna search of the hyperparameters of one model
│   ├── experiments/
│   │   ├── pca.py               PCA on all features or on the bags only
│   │   └── polynomial.py        polynomial features for logistic regression
│   └── submission/
│       └── make_submission.py   fit on full train set and write a submission
└── src/
    ├── features/
    │   ├── load_dataset.py    load_dataset: files -> table of features
    │   ├── core/
    │   │   ├── feature_builder.py  assembles all feature groups into one row per battle
    │   │   ├── pokedex.py          species stats and types taken from the battles
    │   │   └── type_chart.py       type effectiveness lookups
    │   └── groups/
    │       ├── team.py        Player 1 team base stats, items and types
    │       ├── lead.py        Player 2 lead against Player 1 lead
    │       ├── matchup.py     type matchups against Player 2's lead
    │       ├── timeline.py    state of the battle after 30 turns
    │       ├── bags.py        species and moves counts (bag of tokens)
    │       └── endstate.py    detailed end state, hidden KOs, HP trend, HP left per species
    ├── models/
    │   ├── core/
    │   │   ├── model_builder.py   catalog of the models, build_model, tuned models and ensembles
    │   │   └── preprocessing.py   StandardScaler (and PCA) in front of a model
    │   ├── classifiers/       one file per model: logistic_regression, knn, decision_tree,
    │   │                      random_forest, xgboost, ensemble (voting and stacking)
    │   └── submission.py      fit on the train set, predict the test, write the csv
    ├── evaluation/
    │   ├── validation.py      stratified k-fold cross-validation
    │   └── tuning.py          Optuna search spaces and search
    └── utility/
        ├── config/
        │   ├── config_loader.py     read config.json5 into a dictionary
        │   ├── config_constants.py  values of config.json5 and constants of the game
        │   ├── config_paths.py      data and output folders
        │   └── config_run.py        start the command; model, models, trials for the scripts
        ├── data/
        │   └── data_loader.py     read jsonl and csv files
        ├── model/
        │   └── load_params.py     read and update data/models/tuned_params.json
        └── print/
            └── messages.py        messages shown on the terminal
```

## Setup

```bash
pip install -r requirements.txt
```

Download the competition files into `data/`:

```bash
kaggle competitions download fds-pokemon-battles-prediction-2026 -p data
cd data && unzip fds-pokemon-battles-prediction-2026.zip
```

Another data folder can be written in `config.json5` (`data_dir`). On Kaggle the
files are found automatically under `/kaggle/input`.

## Usage

What to run is chosen in `config.json5`, section `run`, then:

```bash
python main.py
```

```json5
run: {
    command: "submit",        // what to do (the list of commands is written above it)
    model: "voting_best",     // one model: submit, tune, ablation
    models: [],               // several models: compare, compare-tuned, pca ([] = all)
    trials: 50,               // Optuna trials of tune
},
```

Each command reads only the values it needs. The names of all the models are listed
in the same file, section "MODEL NAMES".

Commands: `compare`, `compare-tuned`, `ablation`, `tune`, `pca`, `polynomial`,
`submit`. The same file also holds the data folder, file names,
seed and number of folds. Everything starts from `main.py`: each script reads the values it needs (model,
models, trials) from this file.

## Method overview

1. **Features** (one row per battle)
   - *team*: mean/max/min base stats of Player 1's team, items, type variety.
   - *lead*: Player 2's lead stats and difference with Player 1's first slot.
   - *matchup*: type advantage of each Player 1 Pokemon against Player 2's lead.
   - *timeline*: fainted Pokemon, remaining HP, statuses, damage dealt and
     healed, switches, move categories, super effective hits, boosts and
     snapshots at turns 10 and 20; differences Player 1 - Player 2.
   - *bags*: counts of species and moves seen in the battle.
   - *endstate*: health of the Pokemon still alive, low-HP Player 2 Pokemon on the
     bench treated as hidden KOs, trend of the HP difference during the 30 turns,
     HP left of each frequent species.
2. **Validation**: stratified 5-fold cross-validation on `train.jsonl`; the
   public leaderboard is only used as a sanity check.
3. **Models**: logistic regression (L1/L2), kNN, decision tree, random forest and
   XGBoost, then soft voting and stacking ensembles of logistic regression, kNN,
   random forest and XGBoost.
