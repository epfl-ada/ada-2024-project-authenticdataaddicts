import pandas as pd
import numpy as np

def load_data(character_path='data/character.metadata.tsv', movie_path='data/movie.metadata.tsv', plot_path='data/plot_summaries.txt'):
    """
    Loads character and movie metadata from .tsv files. And plot summaries from .txt file

    Parameters:
    - character_path (str): The file path to the character metadata .tsv file.
    - movie_path (str): The file path to the movie metadata .tsv file.
    - plot_path (str): The file path to the plot summaries .txt file.

    Returns:
    - character_data (pd.DataFrame): DataFrame containing the character metadata 
    - movie_data (pd.DataFrame): DataFrame containing the movie metadata
    -plot_data (pd.DataFrame): DataFrame containing the plot summaries
    
    Example usage:
    >>> character_data, movie_data, plot_data = load_data()
    """
    movie_columns = [
        "wikipedia_movie_id", "freebase_movie_id", "movie_name", "movie_release_date",
        "box_office_revenue", "runtime", "languages", "countries", "genres"
    ]
    
    character_columns = [
        "wikipedia_movie_id", "freebase_movie_id", "movie_release_date", "character_name",
        "actor_dob", "actor_gender", "actor_height", "actor_ethnicity", "actor_name",
        "actor_age_at_release", "character_actor_map_id", "character_id", "actor_id"
    ]
    
    plot_columns = ["wikipedia_movie_id","summary"]
    
    character_data = pd.read_csv(character_path, sep='\t', header=None, names=character_columns)
    movie_data = pd.read_csv(movie_path, sep='\t', header=None, names=movie_columns)
    plot_data = pd.read_csv(plot_path, sep='\t', header=None, names=plot_columns)
   
    return character_data, movie_data, plot_data

# ==================== PREPROCESSING ====================

def standardize(df: pd.DataFrame):
    """Standardize data."""
    numeric_df = df.select_dtypes(include=[np.number]) # select only numeric columns
    
    mean = numeric_df.mean(axis=0, skipna=True)
    std = numeric_df.std(axis=0, skipna=True)
    
    standardized_numeric_df = (numeric_df - mean) / std

    standardized_df = df.copy()
    standardized_df[numeric_df.columns] = standardized_numeric_df
    
    return standardized_df

def replace_nans(df: pd.DataFrame, method='mean'):
    """Replace NaN with the specified method."""
    if method == 'mean':
        fill_values = df.mean(numeric_only=True, skipna=True)
    elif method == 'median':
        fill_values = df.median(numeric_only=True, skipna=True)
    else:
        raise ValueError("Method must be 'mean' or 'median'")
    return df.fillna(fill_values)
