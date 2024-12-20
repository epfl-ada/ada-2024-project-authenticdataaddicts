import pandas as pd
import numpy as np
import networkx as nx
from sklearn.linear_model import LinearRegression


def propensity_matching(data: pd.DataFrame, effect: str, treated: pd.Series, max_propensity_difference: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform propensity matching on the data.
    

    Args:
        data (pd.DataFrame): The data to compute the effect on.
        effect (str): The column with the effect.
        treated (pd.Series): The treated group. Is expected to be a boolean series.
        max_propensity_difference (float): The maximum propensity difference.

    Returns:
        tuple[np.ndarray, np.ndarray]: The indices of the matched treated and control groups.
    """

    # Fit a linear regression model to predict the effect
    X = data.drop(effect, axis=1)
    y = data[effect]

    model = LinearRegression()
    model.fit(X, y)

    propensity = model.predict(X)

    treated_propensity = propensity[treated]
    control_propensity = propensity[~treated]

    treated_index = data[treated].index
    control_index = data[~treated].index

    # Compute the difference between the treated and control group
    difference = np.abs(treated_propensity.reshape(-1, 1) - control_propensity.reshape(1, -1))

    # Find the edges
    edges = []
    for i, j in np.argwhere(difference < max_propensity_difference):
        edges.append((treated_index[i], control_index[j]))

    # Create a graph
    G = nx.Graph()
    G.add_edges_from(edges)

    # Find min weight matching
    matching = nx.min_weight_matching(G)

    # Get all matched indices
    first_indices, second_indices = zip(*matching)
    indices = np.concatenate([first_indices, second_indices])

    # Convert indices to boolean mask
    matched = np.zeros(len(data), dtype=bool)
    matched[indices] = True

    # Get treated and control indices of the matched pairs
    treated_indices = X[treated & matched].index
    control_indices = X[~treated & matched].index

    return treated_indices, control_indices


def ate_categorical_values(data: pd.DataFrame, categorical: list[str], effect: str, effect_values: pd.Series) -> pd.DataFrame:
    """Computes the average treatment effect of categorical values

    Args:
        data (pd.DataFrame): The data.
        categorical (list[str]): The categorical column names.
        effect (str): The effect to study.
        effect_values (pd.Series): The actual un-standardized effect values.

    Returns:
        pd.DataFrame: The average treatment effects.
    """
    effects = []
    for category in categorical:
        treated = data[category].astype(bool)
        df = data.drop(category, axis=1)
        
        treated_indices, control_indices = propensity_matching(df, effect, treated, 0.005)

        treated_mean = effect_values[treated_indices].mean()
        control_mean = effect_values[control_indices].mean()
        ate = round(treated_mean - control_mean, 4)

        effects.append((category, ate, treated_mean, control_mean, len(treated_indices)))

    effects = pd.DataFrame(effects, columns=['treatment', 'ATE', 'treatment_mean', 'control_mean', 'size'])
    effects = effects.set_index('treatment')

    return effects