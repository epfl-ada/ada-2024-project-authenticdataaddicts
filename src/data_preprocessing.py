import numpy as np
import pandas as pd
from src.data_fetching import fetch_freebase_labels
import os

def compute_reduction(df_before, df_after):
    """ 
    Compute the reduction in size between two datasets.
    
    Parameters:
    ----------
    df_before : pd.DataFrame
        The original dataset before filtering.
        
    df_after : pd.DataFrame
        The dataset after filtering.

    Returns:
    -------
    float
        The proportion of data removed, as a fraction of the original dataset size.
    """

    return 1 - df_after.shape[0]/df_before.shape[0]

def filter_attribute(df, target_col, v_min, v_max):
    """
    Keep only the dataframe rows with a valid range of the specified attribute.

    Parameters:
    ----------
    df : pd.DataFrame
        The dataset (DataFrame) to preprocess.
    
    target_col : str
        The column that contains the attributes we want to filter.
        
    v_min : float
        Minimum valid value for an attribute in the column.
    
    v_max : float
        Maximum valid value for an attribute in the column.
    
    Returns:
    -------
    pd.DataFrame
        A DataFrame with 'target_col' having only values between v_min and v_max.
    
    float
        The reduction in size of the dataset.
    """
    # Create query to keep values in target_col between v_min and v_max
    filter_query = target_col + ' > ' + str(v_min) + ' and ' + target_col + ' < ' + str(v_max)
    df_filtered = df.query(filter_query)
    reduction = compute_reduction(df, df_filtered)
    
    return df_filtered, reduction


def keep_only_non_nans(df, columns_list):
    """
    Keep only non NaN values for each specified column.

    Parameters:
    ----------
    df : pd.DataFrame
        The dataset (DataFrame) to preprocess.
    
    columns_list : list (of str)
        The list of columns names (str) where we want to remove NaNs.
    
    Returns:
    -------
    pd.DataFrame
        A DataFrame where all values in the specified columns are not NaNs.
    
    float
        The reduction in size of the dataset.
    """
    # Create a mask for non-NaN values across all specified columns
    not_na_mask = df[columns_list].notna().all(axis=1)
    
    # Apply the mask to clean the DataFrame
    df_cleaned = df[not_na_mask].copy()
    reduction = compute_reduction(df, df_cleaned)
    
    return df_cleaned, reduction
    

def standardize(df):
    """
    Standardize the numeric columns in the dataset.

    Parameters:
    ----------
    df : pd.DataFrame
        The dataset to be standardized.
    
    Returns:
    -------
    pd.DataFrame
        A DataFrame with standardized numeric columns.
    """

    # Keep only numerical
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate mean and standard deviation
    mean = numeric_df.mean(axis=0, skipna=True)
    std = numeric_df.std(axis=0, skipna=True)
    
    # Standardized
    standardized_numeric_df = (numeric_df - mean) / std

    standardized_df = df.copy()
    standardized_df[numeric_df.columns] = standardized_numeric_df
    
    return standardized_df

def replace_nans(df, method='mean'):
    """
    Replace NaN values in the dataset using the specified method.

    Parameters:
    ----------
    df : pd.DataFrame
        The dataset where NaN values are to be replaced.
    
    method : str, optional
        The method to use for filling NaNs: 'mean' or 'median'. Default is 'mean'.
    
    Returns:
    -------
    pd.DataFrame
        A DataFrame with NaNs replaced by the specified method.
    
    Raises:
    ------
    ValueError
        If the specified method is not 'mean' or 'median'.
    """

    if method == 'mean':
        fill_values = df.mean(numeric_only=True, skipna=True)
    elif method == 'median':
        fill_values = df.median(numeric_only=True, skipna=True)
    else:
        raise ValueError("Method must be 'mean' or 'median'")
    return df.fillna(fill_values)

def extract_values(data, clean_func=None):
    """
    Extract values from a string-represented dictionary, optionally applying a cleaning function.

    Parameters:
    ----------
    data : str or None
        The string representation of a dictionary to extract values from.
    
    clean_func : callable, optional
        A function to apply to each extracted value for cleaning. Default is None.
    
    Returns:
    -------
    list or None
        A list of values extracted from the dictionary, cleaned if clean_func is provided.
    """

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

def clean_language(value):
    """
    Clean language strings by removing the word 'Language'.

    Parameters:
    ----------
    value : str
        The string to clean.
    
    Returns:
    -------
    str
        The cleaned string with only the first word if 'Language' is present, else the original value.
    """

    # If "Language" is found in the value, only keep the first word
    return value.split()[0] if 'Language' in value else value

def map_ethnicities(character_data, ethnicities_file_path='data/actor_ethnicity.csv'):
    """
    Map ethnicity labels to character data using actor ethnicity information.

    This function checks if a local CSV file containing actor ethnicity labels exists.
    If the file is not found, it fetches ethnicity labels from Freebase using unique values 
    of 'actor_ethnicity' in the provided character data. It then merges these ethnicity labels
    with the character data, adding a new column with the actual ethnicity label.

    Parameters:
    ----------
    character_data : pd.DataFrame
        A DataFrame containing character data, including an 'actor_ethnicity' column 
        with Freebase IDs representing actor ethnicities.
    
    ethnicities_file_path : str, optional
        The file path to save or load the actor ethnicity labels CSV file. 
        Default is 'data/actor_ethnicity.csv'.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the character data with an additional 'actor_ethnicity_label' 
        column, which includes the mapped ethnicity labels. Unnecessary columns from the 
        merging process are dropped.
    """

    merged_character_data = character_data.copy()

    if os.path.exists(ethnicities_file_path):
        print('Ethnicity data already fetched.')
        actor_ethnicity_labels = pd.read_csv(ethnicities_file_path, sep=';')
    else:
        print('Fetching ethnicity data...')
        actor_ethnicity_labels = fetch_freebase_labels(merged_character_data['actor_ethnicity'].dropna().unique())
        actor_ethnicity_labels.to_csv(ethnicities_file_path, index=False, sep=';')

    # Merge the character data with the actor ethnicity data
    merged_character_data = pd.merge(merged_character_data, actor_ethnicity_labels, how='left', left_on='actor_ethnicity', right_on='Freebase ID')

    # Replace by the actual ethnicity label
    merged_character_data["actor_ethnicity_label"] = merged_character_data["Label"]

    # Drop the unnecessary columns
    merged_character_data = merged_character_data.drop(columns=["Freebase ID", "URL", "Label", "actor_ethnicity"])

    return merged_character_data

def preprocess_characters(character_data):

    merged_character_data = character_data.copy()

    # Convert dob and movie_release_date to date
    merged_character_data.loc[:, "actor_dob"] = pd.to_datetime(merged_character_data["actor_dob"], errors='coerce')
    merged_character_data.loc[:, "movie_release_date"] = pd.to_datetime(merged_character_data["movie_release_date"], errors='coerce')

    # Create a mask for cases where `actor_dob` and `movie_release_date` are present but `actor_age_at_release` is missing
    missing_age_cases = merged_character_data['actor_age_at_release'].isna() & \
                        merged_character_data['actor_dob'].notna() & \
                        merged_character_data['movie_release_date'].notna()

    # In this case we can calculate the actor age at release
    for i in merged_character_data[missing_age_cases].index:
        merged_character_data.loc[i,'actor_age_at_release']=merged_character_data.loc[i,'movie_release_date'].year - merged_character_data.loc[i,'actor_dob'].year

    print(f"Number of ages retrieved through calculation (using movie release data and actor dob): {merged_character_data.loc[missing_age_cases,:].shape[0]}")

    # Keep only non-NaN values for all columns
    columns_names = ['actor_height', 'actor_age_at_release', 'actor_gender', 
                    'actor_ethnicity_label', 'character_name', 'actor_dob']
    character_data_cleaned, reduction = keep_only_non_nans(merged_character_data, columns_names)

    print(f"Removing NaN reduced the dataset by: {reduction:.2%}")

    # Keep only valid heights (between 1.5 and 2.8 meters)
    character_data_valid_heights, reduction = filter_attribute(character_data_cleaned, 'actor_height', 1.5, 2.8)

    print(f"Removing invalid actor heights reduced that dataset by {reduction:.2%}.")

    # Keep only valid ages (between 0 and 100 years)
    character_data_valid_ages, reduction = filter_attribute(character_data_valid_heights, 'actor_age_at_release', 0, 100)

    print(f"Removing invalid actor ages reduced that dataset by {reduction:.2%}.")

    # Keep only ethnicity labels that are common
    min_occurrence = 10
    ethnicity_label_counts = character_data_valid_ages['actor_ethnicity_label'].value_counts()
    ethnicity_labels = ethnicity_label_counts[ethnicity_label_counts > min_occurrence]

    mask = character_data_valid_ages['actor_ethnicity_label'].isin(ethnicity_labels.index)
    character_data_valid = character_data_valid_ages[mask]

    reduction = compute_reduction(character_data_valid_ages, character_data_valid)

    print(f"Removing ethnicity labels which are uncommon reduced that dataset by {reduction:.2%}.")

    return character_data_valid

def preprocess_movies(movie_data):
    """
    Preprocess the movie dataset by cleaning, filtering, and enriching data.

    This function performs several preprocessing tasks on movie data, such as
    dropping duplicated columns, setting duplicate lead actors to NaN, removing
    movies without a lead actor, and filtering for valid release dates, runtimes,
    votes, and box office revenues.

    Parameters:
    ----------
    movie_data : pd.DataFrame
        A DataFrame containing movie data, with columns such as 'movie_release_date',
        'runtime', 'languages', 'countries', 'genres', 'lead_actor_1', 'lead_actor_2',
        'box_office_revenue', 'averageRating', and 'numVotes'.

    Returns:
    -------
    pd.DataFrame
        A cleaned and filtered DataFrame containing only movies with valid release 
        dates, runtimes, votes, and box office revenues.
    """

    movie_data_preprocessed = movie_data.copy()

    # Drop duplicated columns
    movie_data_extracted = movie_data_preprocessed.drop(columns=['title', 'release_date', 'release_date_from_second', 'title_from_second'])

    # Set lead_actor_2 to NaN where it is the same as lead_actor_1
    movie_data_extracted.loc[movie_data_extracted['lead_actor_1'] == movie_data_extracted['lead_actor_2'], 'lead_actor_2'] = pd.NA

    # Remove movies where we don't have lead actor
    movie_data_extracted = movie_data_extracted.dropna(subset=['lead_actor_1'])

    # Keep only non-NaN values for all columns (the other columns have no missing values)
    columns_names = ['movie_release_date', 'runtime', 'languages', 'countries', 
                    'genres', 'lead_actor_1', 'box_office_revenue', 'averageRating', 'lead_actor_2']
    movie_data_cleaned, reduction = keep_only_non_nans(movie_data_extracted, columns_names)

    print(f"Removing NaN reduced the dataset by: {reduction:.2%}")

    #Keep only movies released after 1940
    movie_data_valid_release_dates = movie_data_cleaned[movie_data_cleaned["movie_release_date"]>'1-1-1940']

    reduction = compute_reduction(movie_data_cleaned, movie_data_valid_release_dates)
    print(f"Removing movies released before 1940 reduced the dataset by: {reduction:.2%}")

    # Keep only movies that last between 1h and 200min
    movie_data_valid_runtime, reduction = filter_attribute(movie_data_valid_release_dates, 'runtime', 60, 200)
    print(f"Removing movies lasting less than 1h or more than 3 hours 20mins reduced the dataset by: {reduction:.2%}")

    #Keep only movies that have at least 500 votes
    movie_data_valid_votes = movie_data_valid_runtime[movie_data_valid_runtime["numVotes"]>500]

    reduction = compute_reduction(movie_data_valid_runtime, movie_data_valid_votes)
    print(f"Removing movies that have less than 500 votes: {reduction:.2%}")

    # Keep only movies that have more than 0 box office revenue
    movie_data_valid_revenue = movie_data_valid_votes[movie_data_valid_votes["box_office_revenue"]>0]

    reduction = compute_reduction(movie_data_valid_votes, movie_data_valid_revenue)
    print(f"Removing movies that have no box office revenue: {reduction:.2%}")

    return movie_data_valid_revenue

def extract_movies_with_lead_actors_data(movie_data_valid, character_data_valid):
    # Extract unique pairs of (freebase_movie_id, lead_actor) from the movie dataset
    lead_actor_pairs = pd.concat([
        movie_data_valid[['freebase_movie_id', 'lead_actor_1']].rename(columns={'lead_actor_1': 'actor_name'}),
        movie_data_valid[['freebase_movie_id', 'lead_actor_2']].rename(columns={'lead_actor_2': 'actor_name'})
    ])

    # Convert the DataFrame of pairs to a list of tuples for filtering
    lead_actor_pairs = list(lead_actor_pairs.itertuples(index=False, name=None))

    # Filter character_data to keep only rows where (freebase_movie_id, actor_name) matches the pairs in lead_actor_pairs
    lead_actor_data = character_data_valid[
        character_data_valid[['freebase_movie_id', 'actor_name']].apply(tuple, axis=1).isin(lead_actor_pairs)
    ]

    # Check for missing values in key columns
    print("Missing values in lead actor data:")
    print(lead_actor_data[['actor_name', 'actor_dob', 'actor_gender', 'actor_ethnicity_label', 'actor_height', 'actor_age_at_release']].isna().mean()*100)

    return lead_actor_data
