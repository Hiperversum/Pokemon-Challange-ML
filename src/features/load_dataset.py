"""From the competition files to the tables of features used by the models."""

from ..utility.data.data_loader import load_battles, load_type_chart
from ..utility.print.messages import print_info, timed_step
from .core.feature_builder import FeatureBuilder, extract_target
from .core.pokedex import Pokedex
from .core.type_chart import TypeChart


####################################################################################
# LOAD THE DATASET
####################################################################################

def load_dataset(groups=None, data_dir=None):
    """Load train and test battles and return (X_train, y_train, X_test).

    Everything that is learned from the data (the Pokedex and the bag vocabulary) comes
    from train.jsonl only: the competition rules allow test.jsonl only for the predictions.
    """
    # Read the battles (one dictionary each) and build the Pokedex and the type chart
    with timed_step("Reading the battles, the Pokedex and the type chart"):
        train = load_battles("train", data_dir)
        test = load_battles("test", data_dir)
        pokedex = Pokedex.from_battles(train)
        type_chart = TypeChart(load_type_chart(data_dir))
    print_info("train / test battles", f"{len(train)} / {len(test)}")

    # Turn every battle into a row of numbers; the vocabulary is learned on the train only
    with timed_step("Building the features"):
        builder = FeatureBuilder(pokedex, type_chart, groups=groups).fit(train)
        X_train = builder.transform(train)
        # The test table must have the same columns, in the same order, as the train table
        X_test = builder.transform(test).reindex(columns=X_train.columns, fill_value=0.0)
        y_train = extract_target(train).loc[X_train.index]
    print_info("features per battle", X_train.shape[1])
    return X_train, y_train, X_test
