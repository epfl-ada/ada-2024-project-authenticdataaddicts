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

def inflation_plots(movie_inflation_data, top = False):
    """Plot with the average box office revenue over time (adjusted & unadjusted) 
    for all movies & for top 10 movies

    Args:
        movie_inflation_data (pd.DataFrame): The inflation_dataset
        top (bool, optional): If True, plots for the top 10 movies. False if nothing specified.
    """
    #Group by release year and calculate the average box office revenue
    if top: #For top 10 movies
        avg_box_office = movie_inflation_data.groupby('release_year').apply(
            lambda x: x.nlargest(10, 'box_office_revenue')
        ).reset_index(drop=True)
        avg_box_office = avg_box_office.groupby('release_year')[['adjusted_box_office', 'box_office_revenue']].mean()
    else: #For all movies
        avg_box_office = movie_inflation_data.groupby('release_year')[['adjusted_box_office', 'box_office_revenue']].mean()

    #Convert box office to millions of dollars for readability
    avg_box_office['adjusted_box_office'] = avg_box_office['adjusted_box_office'] / 1e6
    avg_box_office['box_office_revenue'] = avg_box_office['box_office_revenue'] / 1e6

    #Plot the average box office revenue over time (adjusted & unadjuste)
    fig, ax = plt.subplots(figsize=(15, 6))

    sns.scatterplot(data=avg_box_office, x=avg_box_office.index, y='adjusted_box_office', label="Adjusted Box Office", color='blue')
    sns.scatterplot(data=avg_box_office, x=avg_box_office.index, y='box_office_revenue', label="Unadjusted Box Office", color='red')

    ax.set_title("Average Box Office of Movies over Time")
    ax.set_xlabel("Movie release Year")
    ax.set_ylabel("Average Box Office Revenue (in millions dollars)")
    ax.grid(axis='y')
    ax.legend()

    plt.show()


def ate_barplot(rating_effects: pd.DataFrame, revenue_effects: pd.DataFrame, effect: str):
    """Creates a bar plot showing the average rating and revenue effects.

    Args:
        rating_effects (pd.DataFrame): The average rating effects.
        revenue_effects (pd.DataFrame): The box office revenue effects.
        effect (str): The effect's name.
    """

    rating_country_effect = rating_effects.sort_values('ATE', ascending=False)
    revenue_country_effect = revenue_effects.loc[rating_country_effect.index]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.barplot(rating_country_effect.reset_index(names=['index']), y='index', x='ATE', ax=axes[0])
    axes[0].set_ylabel(effect)
    axes[0].set_title(f'Average effect of {effect.lower()} on average rating')

    sns.barplot(revenue_country_effect.reset_index(names=['index']), y='index', x='ATE', ax=axes[1])
    axes[1].set_ylabel(effect)
    axes[1].set_title(f'Average effect of {effect.lower()} on log adjusted box office revenue')

    fig.tight_layout()
    
    
def box_office_by_genre_barplot(movies: pd.DataFrame, column_bo = 'adjusted_box_office', cutoff = 10):
    """Creates a bar plot showing the (adjusted) box office revenue by genre for the top 10 genres.
    
    Args:
        movies (pd.DataFrame): The movie dataset with box office revenue and genres.
        column_bo (str): column name (either 'box_office_revenue' or 'adjusted_box_office'). Default to adjusted.
        cutoff (int): cutoff to show only the higher box offices. Default to 10.

    """
    # Some movies have multiple genres, so we need to look at all of them
    genres_exploded = movies["genres"].apply(eval).explode()
    genres_revenue_df = pd.DataFrame({
        "genre": genres_exploded,
        column_bo: movies.loc[genres_exploded.index, column_bo]
    })

    # Grouping by the genres to compare box office
    genre_revenue_aggregated = genres_revenue_df.groupby("genre")[column_bo].sum().reset_index()
    top_genres = genre_revenue_aggregated.sort_values(by=column_bo, ascending=False).head(cutoff)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=column_bo, y='genre', data=top_genres)
    adj_string = 'Adjusted ' if column_bo == 'adjusted_box_office' else ''
    plt.title('Top ' + str(cutoff) + ' Genres by ' + adj_string + 'Box Office Revenue')
    plt.xlabel('Total ' + adj_string + 'Box Office Revenue')
    plt.ylabel('Genre')
    plt.tight_layout()
    plt.show()