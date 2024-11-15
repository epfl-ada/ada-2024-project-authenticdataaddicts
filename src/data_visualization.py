import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def histogram_plots(
    data: pd.DataFrame,
    columns: list[str],
    titles: list[str],
    labels: list[str],
    bins: list[int] | int = 50,
    log_scale: list[bool] | bool = False,
    kdes: list[bool] | bool = True,
    hue: str | None = None,
    axes: list[plt.Axes] = None,
):
    """Plot histograms for the specified columns.

    Args:
        data (pd.DataFrame): The dataset to plot.
        columns (list[str]): The columns form the dataset to plot.
        titles (list[str]): The titles for each plot.
        labels (list[str]): The x labels for each plot.
        bins (list[int] | int, optional): The number of bins. Defaults to 50.
        log_scale (list[bool] | bool, optional): If true, uses a log scale. Defaults to False.
        kdes (list[bool] | bool, optional): If true, adds a kde curve. Defaults to True.
        hue (str | None, optional): Name of the column in the dataset. Defaults to None.
        axes (list[plt.Axes], optional): The axes to plot on. Defaults to None.
    """
    if axes is None:
        fig, axes = plt.subplots(1, len(columns), figsize=(6 * len(columns), 5))
    else:
        fig = None

    if not isinstance(bins, list):
        bins = [bins] * len(columns)

    if not isinstance(log_scale, list):
        log_scale = [log_scale] * len(columns)

    if not isinstance(kdes, list):
        kdes = [kdes] * len(columns)

    for i, col in enumerate(columns):
        sns.histplot(
            data=data,
            x=col,
            hue=hue,
            ax=axes[i],
            kde=kdes[i],
            bins=bins[i],
            log_scale=log_scale[i],
            legend=True if hue else False,
            stat="density",
        )
        axes[i].set_title(titles[i])
        axes[i].set_xlabel(labels[i])

    if fig:
        fig.tight_layout()


def histogram_actors(
    actors: pd.DataFrame,
    hue: str | None = None,
    axes: list[plt.Axes] = None,
):
    """Plot histograms for the actor data.

    Args:
        actors (pd.DataFrame): The actor dataset.
        hue (str | None, optional): Name of the column in the dataset. Defaults to None.
        axes (list[plt.Axes], optional): The axes to plot on. Defaults to None.
    """
    columns = ["actor_height", "actor_age_at_release", "actor_dob"]
    titles = [
        "Height of the actor",
        "Age of the actor at the release of the movie",
        "Date of birth of the actor",
    ]
    labels = ["Height (m)", "Age (years)", "Date of birth"]

    histogram_plots(actors, columns, titles, labels, bins=25, hue=hue, axes=axes)


def histogram_movies(
    movies: pd.DataFrame,
    hue: str | None = None,
    axes: list[plt.Axes] = None,
):
    """Plot histograms for the movie data.

    Args:
        movies (pd.DataFrame): The movie dataset.
        hue (str | None, optional): Name of the column in the dataset. Defaults to None.
        axes (list[plt.Axes], optional): The axes to plot on. Defaults to None.
    """
    columns = ["runtime", "box_office_revenue", "movie_release_date"]
    titles = [
        "Runtime of the movie",
        "Box office revenue of the movie",
        "Release date of the movie",
    ]
    labels = ["Runtime (min)", "Box office revenue (dollars, log scale)", "Release date"]
    log_scale = [False, True, False]

    histogram_plots(
        movies,
        columns,
        titles,
        labels,
        bins=50,
        log_scale=log_scale,
        hue=hue,
        axes=axes,
    )


def histogram_movie_ratings(
    movies: pd.DataFrame,
    hue: str | None = None,
    axes: list[plt.Axes] = None,
):
    """Plot histograms for the movie ratings.

    Args:
        movies (pd.DataFrame): The movie dataset.
        hue (str | None, optional): Name of the column in the dataset. Defaults to None.
        axes (list[plt.Axes], optional): The axes to plot on. Defaults to None.
    """
    columns = ["averageRating", "numVotes"]
    titles = ["Average rating of the movie", "Number of votes for the movie"]
    labels = ["Average rating", "Number of votes (log scale)"]
    log_scale = [False, True]
    bins = [range(11), 50]
    kdes = [False, True]

    histogram_plots(
        movies,
        columns,
        titles,
        labels,
        bins=bins,
        log_scale=log_scale,
        kdes=kdes,
        hue=hue,
        axes=axes,
    )


def count_plots(
    data: pd.DataFrame,
    columns: list[str],
    titles: list[str],
    labels: list[str],
    cutoffs: list[int] | int | None = None,
    horizontal: list[bool] | bool = True,
    transforms=None,
    hue: str | None = None,
    axes: list[plt.Axes] = None,
):
    """Plot count plots for the specified columns.

    Args:
        data (pd.DataFrame): The dataset to plot.
        columns (list[str]): The columns form the dataset to plot.
        titles (list[str]): The titles for each plot.
        labels (list[str]): The x labels for each plot.
        cutoffs (list[int] | int | None, optional): The cutoffs. Defaults to None.
        horizontal (list[bool] | bool, optional): If true, plots horizontal bars. Defaults to True.
        transforms ([type], optional): [description]. Defaults to None.
        hue (str | None, optional): Name of the column in the dataset. Defaults to None.
        axes (list[plt.Axes], optional): The axes to plot on. Defaults to None.
    """
    if axes is None:
        fig, axes = plt.subplots(1, len(columns), figsize=(6 * len(columns), 5))
    else:
        fig = None

    if not isinstance(cutoffs, list):
        cutoffs = [cutoffs] * len(columns)

    if not isinstance(transforms, list):
        transforms = [transforms] * len(columns)

    if not isinstance(horizontal, list):
        horizontal = [horizontal] * len(columns)

    for i, col in enumerate(columns):
        transform = transforms[i]
        if transform:
            col_data = transform(data[col])
            data_copy = pd.merge(
                col_data, data, left_index=True, right_index=True, suffixes=("_x", "")
            )
            col = f"{col}_x"
        else:
            data_copy = data

        col_counts = data_copy[col].value_counts()

        cutoff = cutoffs[i]
        if cutoff:
            col_counts = col_counts[:cutoff]
            data_copy = data_copy[data_copy[col].isin(col_counts.index)]

        if horizontal[i]:
            sns.countplot(
                data=data_copy, y=col, hue=hue, ax=axes[i], order=col_counts.index
            )
            axes[i].set_title(titles[i])
            axes[i].set_ylabel(labels[i])
        else:
            sns.countplot(
                data=data_copy, x=col, hue=hue, ax=axes[i], order=col_counts.index
            )
            axes[i].set_title(titles[i])
            axes[i].set_xlabel(labels[i])

    if fig:
        fig.tight_layout()


def count_actors(
    actors: pd.DataFrame,
    hue: str | None = None,
    axes: list[plt.Axes] = None,
):
    """Plot count plots for the actor data.

    Args:
        actors (pd.DataFrame): The actor dataset.
        hue (str | None, optional): Name of the column in the dataset. Defaults to None.
        axes (list[plt.Axes], optional): The axes to plot on. Defaults to None.
    """
    columns = ["actor_gender", "actor_ethnicity_label"]
    titles = ["Actor gender distribution", "Most common ethnicities"]
    labels = ["Gender", "Ethnicity"]
    cutoffs = [None, 20]
    horizontal = [False, True]

    count_plots(
        actors,
        columns,
        titles,
        labels,
        cutoffs,
        horizontal=horizontal,
        hue=hue,
        axes=axes,
    )


def count_movies(
    movies: pd.DataFrame,
    hue: str | None = None,
    axes: list[plt.Axes] = None,
):
    """Plot count plots for the movie data.

    Args:
        movies (pd.DataFrame): The movie dataset.
        hue (str | None, optional): Name of the column in the dataset. Defaults to None.
        axes (list[plt.Axes], optional): The axes to plot on. Defaults to None.
    """
    columns = ["genres", "languages", "countries"]
    titles = ["Most common genres", "Most common languages", "Most common countries"]
    labels = ["Genres", "Languages", "Countries"]
    cutoffs = [20, 10, 10]
    transforms = lambda col: col.apply(eval).explode()

    count_plots(
        movies,
        columns,
        titles,
        labels,
        cutoffs,
        transforms=transforms,
        hue=hue,
        axes=axes,
    )
