import pandas as pd
import numpy as np

CHARACTER_DATA_PATH = 'data/character.metadata.tsv'
ACTOR_ETHNICITY_PATH = 'data/actor_ethnicity.csv'


def load_character_data():
    """Loads the character data and merges it with the actor ethnicity data."""
    character_columns = [
        "wikipedia_movie_id", "freebase_movie_id", "movie_release_date", "character_name",
        "actor_dob", "actor_gender", "actor_height", "actor_ethnicity", "actor_name",
        "actor_age_at_release", "character_actor_map_id", "character_id", "actor_id"
    ]
    
    character_data = pd.read_csv(CHARACTER_DATA_PATH, sep='\t', header=None, names=character_columns)
    actor_ethnicity = pd.read_csv(ACTOR_ETHNICITY_PATH, sep=';')

    # Merge the character data with the actor ethnicity data
    merged = pd.merge(character_data, actor_ethnicity, how='left', left_on='actor_ethnicity', right_on='Freebase ID')

    # Replace by the actual ethnicity label
    merged["actor_ethnicity_label"] = merged["Label"]

    # Drop the unnecessary columns
    merged = merged.drop(columns=["Freebase ID", "URL", "Label"])

    return merged


def load_data(movie_path='data/movie.metadata.tsv', plot_path='data/plot_summaries.txt'):
    """
    Loads character and movie metadata from .tsv files. And plot summaries from .txt file

    Parameters:
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
    
    plot_columns = ["wikipedia_movie_id","summary"]
    
    character_data = load_character_data()
    movie_data = pd.read_csv(movie_path, sep='\t', header=None, names=movie_columns)
    plot_data = pd.read_csv(plot_path, sep='\t', header=None, names=plot_columns)
   
    return character_data, movie_data, plot_data

# ==================== DATA COMPLETION ====================

def merge_for_completion(df1, df2, merge_cols_1, merge_cols_2, target_col, merge_strategy='prioritize_first'):
    """
    Merges two datasets based on the specified columns, merging only the target column from the second dataset.

    Parameters:
    ----------
    df1 : pd.DataFrame
        The first dataset (DataFrame) to merge.
    
    df2 : pd.DataFrame
        The second dataset (DataFrame) to merge.
    
    merge_cols_1 : list
        The list of columns in `df1` to merge on.
    
    merge_cols_2 : list
        The list of columns in `df2` to merge on.
    
    target_col : str
        The column to calculate the missing value percentage increase.
    
    merge_strategy : str, optional
        Defines the merge strategy for handling `target_col` after the merge. 
        Options are:
        - 'prioritize_first': Prioritize values from the first dataset (`df1`).
        - 'mean': Take the mean of values from both datasets.
        - 'add_column': Here we just add a column, so no need to have merge strategy
    
    Returns:
    -------
    pd.DataFrame
        A merged DataFrame with only the `target_col` from `df2` and other columns from `df1`.
    
    float
        The number of missing before the merging
    
    float
        The number of missing after the merging
    """

    # Copy of the datasets to avoid modifying the originals
    df1_copy = df1.copy()
    df2_copy = df2.copy()

    missing_before = 0
    missing_after = 0
    
    # Drop duplicates based on the specified columns (to avoid the duplication when merging)
    df1_copy = df1_copy.drop_duplicates(subset=merge_cols_1)
    df2_copy = df2_copy.drop_duplicates(subset=merge_cols_2)

    if merge_strategy != "add_column":
        # Calculate missing values before the merge
        missing_before = df1_copy[target_col].isna().mean()
    
    # Perform the merge on the target column only (other columns remain unchanged)
    merged_df = pd.merge(df1_copy, 
                         df2_copy[merge_cols_2 + [target_col]], 
                         how='left', 
                         left_on=merge_cols_1, 
                         right_on=merge_cols_2, 
                         suffixes=('', '_from_second'))
    
    # Handle missing values based on the specified merge strategy
    if merge_strategy == 'prioritize_first':
        merged_df[target_col] = merged_df[target_col].fillna(merged_df[f'{target_col}_from_second'])
    elif merge_strategy == 'mean':
        # Take the mean of both columns, ignoring NaN values by default
        merged_df[target_col] = merged_df[[target_col, f'{target_col}_from_second']].mean(axis=1)
    
    if merge_strategy != "add_column":
        # Drop the extra column from the second dataset
        merged_df.drop([f'{target_col}_from_second'], axis=1, inplace=True)
        
        # Drop the additional merge columns from df2 if desired (optional step)
        for col in merge_cols_2:
            if f"{col}_from_second" in merged_df.columns:
                merged_df.drop(columns=[f"{col}_from_second"], inplace=True)
        
        # Calculate missing values after the merge
        missing_after = merged_df[target_col].isna().mean()
    
    return merged_df, missing_before, missing_after




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

def extract_values(data, clean_func=None):
    if not data or pd.isna(data):  
        return None
    try:
        # Convert the string to a dictionary by replacing single quotes with double quotes
        data_dict = eval(data.replace("'", "\""))
        
        # Extract all the values from the dictionary
        values = list(data_dict.values()) if data_dict else None
        
        if clean_func and values:
            values = [clean_func(value) for value in values]
        
        return values
    except:
        return None

# Cleaning function for languages to remove the word "Language"
def clean_language(value):
    # If "Language" is found in the value, only keep the first word
    return value.split()[0] if 'Language' in value else value