import pandas as pd

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

def extract_lead_actors(title_principals_path='data/title.principals.tsv'):
    """
    Extracts a list of lead actors and actresses from an IMDb title principals dataset.

    This function reads IMDb title principals data in chunks to avoid high memory usage, filtering for 
    lead actors and actresses. It selects entries based on the `category` and `ordering` columns:
    - `category`: Includes only entries labeled as 'actor' or 'actress'.
    - `ordering`: Includes only entries with a value of 1 or 2 (representing the first and second roles).

    Parameters:
    ----------
    title_principals_path : str, optional
        The file path to the IMDb title principals dataset (default is 'data/title.principals.tsv').

    Returns:
    -------
    list of DataFrame
        A list of pandas DataFrames, each containing rows for lead actors and actresses from 
        one chunk of the dataset. Each DataFrame contains the following columns:
        - `tconst`: The unique identifier for the title (e.g., movie or show).
        - `nconst`: The unique identifier for the name (e.g., actor or actress).
        - `ordering`: The order in which the name is listed for the title (1 or 2, indicating lead roles).
    """

    filtered_lead_actors = []

    # Process imdb_principals in chunks to reduce memory usage
    for chunk in pd.read_csv(title_principals_path, sep='\t', chunksize=100000):
        # Filter for lead actors (first and second-billed actor or actress)
        chunk_actors = chunk[
            (chunk['category'].isin(['actor', 'actress'])) & 
            (chunk['ordering'].isin([1, 2]))
        ][['tconst', 'nconst', 'ordering']]
        
        # Append the filtered chunk to the list
        filtered_lead_actors.append(chunk_actors)
    
    return filtered_lead_actors

def merge_lead_actors_and_ratings(movie_data, imdb_names, imdb_ratings, filtered_lead_actors):
    """
    Merges lead actor information into the movie dataset.

    This function processes and merges lead actor information from an IMDb dataset into a movie dataset.
    It filters and pivots data to get the primary names of the first- and second-billed actors for each title 
    and merges this information with movie data.

    Parameters:
    ----------
    movie_data : DataFrame
        A pandas DataFrame containing movie data.
    imdb_names : DataFrame
        A pandas DataFrame with IMDb name data.
    imdb_ratings : DataFrame
        A pandas DataFrame containing the ratings
    filtered_lead_actors : list of DataFrame
        A list of DataFrames containing filtered lead actor data.

    Returns:
    -------
    DataFrame
        A pandas DataFrame containing the original movie data merged with lead actor and ratings information. 
    """

    # Concatenate all filtered chunks into a single DataFrame
    lead_actors = pd.concat(filtered_lead_actors)

    # Rename columns and merge with imdb_names DataFrame to get actor names
    lead_actors = lead_actors.rename(columns={'tconst': 'imdb_id'})
    lead_actors = lead_actors.merge(imdb_names[['nconst', 'primaryName']], on='nconst', how='left')

    # Pivot to get separate columns for the first and second lead actors
    lead_actors = lead_actors.pivot(index='imdb_id', columns='ordering', values='primaryName').reset_index()
    lead_actors.columns = ['imdb_id', 'lead_actor_1', 'lead_actor_2']

    # Merge movies_wikidata_merged_imdbid with IMDb ratings
    imdb_merged_movie_data = pd.merge(movie_data, imdb_ratings, on='imdb_id', how='left')

    # Merge with lead actors data
    imdb_merged_movie_data_actors = imdb_merged_movie_data.merge(lead_actors, on='imdb_id', how='left')

    imdb_merged_movie_data_actors = imdb_merged_movie_data_actors.drop_duplicates(subset=['imdb_id'])

    return imdb_merged_movie_data_actors