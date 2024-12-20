import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
import ast
from .data_loading import load_data
from .data_preprocessing import extract_movies_with_lead_actors_data, map_ethnicities, preprocess_characters, preprocess_movies, extract_values, ETHNICITY_MAPPING, GENRE_MAPPING
from .data_completion import extract_lead_actors, merge_for_completion, merge_lead_actors_and_ratings
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from .model_convert_to_onxx import save_scalers_as_json, export_model_to_onnx

def get_most_frequent_ethnicity(x):
    """
    Find the most frequent ethnicity for each movie (freebase_movie_id)
    """
    modes = pd.Series(x).mode()
    if not modes.empty:
        return modes[0]
    return None 

def calculate_character_stats(character_data, movie_id_column):
    """
    Calculates mean age and mean height of characters for each movie.
    """
    character_stats = character_data.groupby(movie_id_column).agg(
        mean_age=('actor_age_at_release', 'mean'),
        mean_height=('actor_height', 'mean')
    ).reset_index()

    character_stats = character_stats.rename(columns={
        'mean_age': 'mean_age_of_characters',
        'mean_height': 'mean_height_of_characters',
    })
    return character_stats

def swap_actors_if_missing(merged_data):
    """
    Swap lead actor 1 and 2 if one is not defined
    """

    # List of columns to swap between lead_actor_1 and lead_actor_2
    actor_columns = [
        'actor_gender_lead_actor1', 'actor_height_lead_actor1', 'actor_age_at_release_lead_actor1', 'actor_ethnicity_label_lead_actor1'
    ]

    # Check for NaN in lead_actor_1 columns and swap with lead_actor_2 if NaN
    for col in actor_columns:
        # Swap the values if lead_actor_1 has NaN and lead_actor_2 has a value
        mask = merged_data[col].isna() & merged_data[col.replace('lead_actor1', 'lead_actor2')].notna()
        merged_data.loc[mask, col] = merged_data.loc[mask, col.replace('lead_actor1', 'lead_actor2')]
        merged_data.loc[mask, col.replace('lead_actor1', 'lead_actor2')] = merged_data.loc[mask, col]

    # If lead_actor_2 has NaN, copy the lead_actor_1 values to lead_actor_2
    mask_lead2_nan = merged_data[col.replace('lead_actor1', 'lead_actor2')].isna()
    merged_data.loc[mask_lead2_nan, col.replace('lead_actor1', 'lead_actor2')] = merged_data.loc[mask_lead2_nan, col]

    return merged_data

def count_characters(character_data, movie_id_column):
    """
    Counts the number of characters for each movie.
    """
    character_count = character_data.groupby(movie_id_column)['character_name'].count().reset_index()
    character_count.rename(columns={'character_name': 'number_of_characters'}, inplace=True)
    return character_count

def most_frequent_ethnicity(character_data, movie_id_column):
    """
    Extract the most frequent etchnity in a movie
    """
    most_frequent_ethnicity = character_data.groupby(movie_id_column)['actor_ethnicity_label'].agg(get_most_frequent_ethnicity).reset_index()
    most_frequent_ethnicity = most_frequent_ethnicity.rename(columns={
        'actor_ethnicity_label': 'most_frequent_ethnicity'
    })
    most_frequent_ethnicity['most_frequent_ethnicity'] = most_frequent_ethnicity['most_frequent_ethnicity'].map(lambda x: ETHNICITY_MAPPING.get(x, "Other ethnicity"))

    return most_frequent_ethnicity

def process_genres(merged_data):
    """
    Maps genres into broader categories and applies one-hot encoding.
    """    
    def map_genre(genres):
        return [GENRE_MAPPING.get(g, "Other genre") for g in genres]
    
    # Map genres to categories
    grouped_genres = merged_data['genres'].map(map_genre)

    # Flatten genres and apply one-hot encoding
    flattened_genres = grouped_genres.apply(lambda x: ','.join(x) if isinstance(x, list) else x)
    genre_onehot = pd.get_dummies(flattened_genres.str.split(',').explode()).groupby(level=0).any()

    merged_data = pd.concat([merged_data, genre_onehot], axis=1)

    merged_data = merged_data.drop(['genres'], axis=1)

    return merged_data

def transform_actors_ethinicities(merged_data_filtered):
    """
    Transform ethnicities using the mapping defined in data_preprocessing.py
    """
    # Map Ethnicities in the Dataset to Broader Groups
    merged_data_filtered['grouped_ethnicity_lead_actor1'] = merged_data_filtered['actor_ethnicity_label_lead_actor1'].map(lambda x: ETHNICITY_MAPPING.get(x, "Other ethnicity"))
    merged_data_filtered['grouped_ethnicity_lead_actor2'] = merged_data_filtered['actor_ethnicity_label_lead_actor2'].map(lambda x: ETHNICITY_MAPPING.get(x, "Other ethnicity"))

    # Combine Ethnicities for One-Hot Encoding
    all_grouped_ethnicities = set(merged_data_filtered['grouped_ethnicity_lead_actor1']) | set(merged_data_filtered['grouped_ethnicity_lead_actor2'])

    leadactor1_ethnicities = pd.get_dummies(merged_data_filtered['grouped_ethnicity_lead_actor1']).add_suffix('_lead_actor1')
    leadactor2_ethnicities = pd.get_dummies(merged_data_filtered['grouped_ethnicity_lead_actor2']).add_suffix('_lead_actor2')

    # Step 5: Combine One-Hot Encoded Data Back into Original DataFrame
    merged_data_filtered = pd.concat([merged_data_filtered, leadactor1_ethnicities, leadactor2_ethnicities], axis=1)

    most_common_ethni = pd.get_dummies(merged_data_filtered['most_frequent_ethnicity']).add_suffix('_most_frequent')
    merged_data_filtered = pd.concat([merged_data_filtered, most_common_ethni], axis=1)

    merged_data_filtered = merged_data_filtered.drop(['grouped_ethnicity_lead_actor1', 'most_frequent_ethnicity', 'grouped_ethnicity_lead_actor2', 'actor_ethnicity_label_lead_actor1', 'actor_ethnicity_label_lead_actor2', ], axis=1)

    merged_data_filtered = merged_data_filtered.drop(['release_date'], axis=1)

    return merged_data_filtered

def prepare_dataset():
    """
    Prepare the datasets for rating prediction
    """

    # Load the datasets
    print("Loading the data...")
    character_data, movie_data, plot_data = load_data()
    movies_dataset = pd.read_csv('data/tmbd_movies.csv')
    imdb_ratings = pd.read_csv('data/title.ratings.tsv', sep='\t')
    imdb_names = pd.read_csv('data/name.basics.tsv', sep='\t')

    # Preprocess needed columns
    print("Preprocessing the columns of different datasets...")

    merged_character_data = map_ethnicities(character_data)
    character_data_valid = preprocess_characters(merged_character_data)
    
    movie_data['genres'] = movie_data['genres'].apply(lambda x: extract_values(x))  
    movie_data['movie_release_date'] = pd.to_datetime(movie_data['movie_release_date'], errors='coerce')
    movie_data = movie_data.dropna(subset=['movie_release_date'])
    movie_data["movie_name"] = movie_data["movie_name"].str.lower().str.strip()
    movie_data["movie_release_year"] = movie_data["movie_release_date"].dt.year

    movies_dataset['release_date'] = pd.to_datetime(movies_dataset['release_date'], errors='coerce')
    movies_dataset["title"] = movies_dataset["title"].str.lower().str.strip()
    movies_dataset["movie_release_year"] = movies_dataset["release_date"].dt.year

    imdb_ratings = imdb_ratings.rename(columns={'tconst': 'imdb_id'})
    imdb_ratings = imdb_ratings[['imdb_id', 'averageRating', 'numVotes']]

    filtered_lead_actors = extract_lead_actors()

    # Merge the datasets
    print("Merging the datasets...")

    movie_data_merged, _, _ = merge_for_completion(movie_data, movies_dataset, ["movie_name", "movie_release_year"], ["title", "movie_release_year"], "runtime", merge_strategy='prioritize_first')
    movie_data_merged, _, _ = merge_for_completion(movie_data_merged, movies_dataset, ["movie_name", "movie_release_date"], ["title", "release_date"], "imdb_id", merge_strategy='add_column')
    movie_data_merged = merge_lead_actors_and_ratings(movie_data_merged, imdb_names, imdb_ratings, filtered_lead_actors)

    # Preprocess the merged dataset
    print("Preprocessing merged dataset...")
    movie_data_merged = movie_data_merged.drop(['movie_name', 'box_office_revenue', 'languages', 'countries', 'movie_release_date', 'title', 'title_from_second', 'numVotes'], axis=1)
    not_na_mask = ~(movie_data_merged['lead_actor_1'].isna() & movie_data_merged['lead_actor_2'].isna() | movie_data_merged['averageRating'].isna() | movie_data_merged['genres'].isna() | movie_data_merged['imdb_id'].isna()) 
    movie_data_valid = movie_data_merged[not_na_mask].copy()

    # Create subset data
    print("Creating subset of data...")
    lead_actor_data = extract_movies_with_lead_actors_data(movie_data_valid, character_data_valid)
    subset_movie_with_full_data_on_lead_actors = movie_data_valid[movie_data_valid['freebase_movie_id'].isin(lead_actor_data['freebase_movie_id'])]
    subset_characters_with_lead_actor_data = character_data_valid[character_data_valid['freebase_movie_id'].isin(subset_movie_with_full_data_on_lead_actors['freebase_movie_id'])]
    subset_characters_with_lead_actor_data = subset_characters_with_lead_actor_data[['actor_name', 'actor_dob', 'actor_gender', 'actor_ethnicity_label', 'actor_height', 'actor_age_at_release', 'freebase_movie_id', 'character_name']]
    lead_actors_data_on_subset_movie = extract_movies_with_lead_actors_data(subset_movie_with_full_data_on_lead_actors, character_data_valid)

    # Add bugdet to subset movie
    print("Adding budget as a feature...")

    subset_movie_with_full_data_on_lead_actors = pd.merge(
        subset_movie_with_full_data_on_lead_actors, 
        movies_dataset[['imdb_id', 'budget']], 
        on='imdb_id', 
        how='left'
    )

    return lead_actors_data_on_subset_movie, subset_movie_with_full_data_on_lead_actors, subset_characters_with_lead_actor_data


def merge_actor_data(merged_data, lead_actor_data):
    """
    Merges lead actor data into the main dataset based on actor names and movie IDs.
    """
    # Add suffixes for clarity and merge data
    lead_actor_data_1 = lead_actor_data.copy().add_suffix('_lead_actor1')
    lead_actor_data_2 = lead_actor_data.copy().add_suffix('_lead_actor2')

    merged_data = merged_data.merge(
        lead_actor_data_1, how='left',
        left_on=['freebase_movie_id', 'lead_actor_1'],
        right_on=['freebase_movie_id_lead_actor1', 'actor_name_lead_actor1']
    )
    merged_data = merged_data.merge(
        lead_actor_data_2, how='left',
        left_on=['freebase_movie_id', 'lead_actor_2'],
        right_on=['freebase_movie_id_lead_actor2', 'actor_name_lead_actor2']
    )
    return merged_data


def random_forest_model():
    """
    Train a random forest model to predict rating of movies
    """

    lead_actors_data_on_subset_movie, subset_movie_with_full_data_on_lead_actors, characters_data_on_subset_movie = prepare_dataset()

    print("Converting features to datetime...")

    # We must convert dates to datetime
    lead_actors_data_on_subset_movie['actor_dob'] = pd.to_datetime(lead_actors_data_on_subset_movie['actor_dob'])
    characters_data_on_subset_movie['actor_dob'] = pd.to_datetime(characters_data_on_subset_movie['actor_dob'])

    print("Merging the actors data in movie...")

    # Merge the data of actors and movies
    merged_data = merge_actor_data(subset_movie_with_full_data_on_lead_actors, lead_actors_data_on_subset_movie)

    # Add a most frquent column
    print("Adding most frequent ethnicity column...")
    most_frequent_ethni = most_frequent_ethnicity(characters_data_on_subset_movie, 'freebase_movie_id')
    merged_data = merged_data.merge(most_frequent_ethni, on='freebase_movie_id', how='left')
   
    # Add the mean of height and age of the actors in the movie
    print("Adding characters stats ...")
    character_stats = calculate_character_stats(characters_data_on_subset_movie, 'freebase_movie_id')
    merged_data = merged_data.merge(character_stats, on='freebase_movie_id', how='left')

    # Add the count of actors
    character_count = count_characters(characters_data_on_subset_movie, 'freebase_movie_id')
    merged_data = merged_data.merge(character_count, on='freebase_movie_id', how='left')

    # Drop unused columns
    print("Dropping unused columns ...")
    merged_data = merged_data.drop(
        ["wikipedia_movie_id", "freebase_movie_id", 'movie_name', 'languages', 'countries', 'movie_release_date', 'numVotes',
        'imdb_id', 'freebase_movie_id_lead_actor2', 'wikipedia_movie_id_lead_actor1', 'box_office_revenue', 'actor_dob_lead_actor1', 'actor_dob_lead_actor2', 
        'actor_name_lead_actor1', 'movie_release_date_lead_actor1', 'character_actor_map_id_lead_actor1', 'character_name_lead_actor1', 'actor_name_lead_actor2', 'character_name_lead_actor2', 'lead_actor_1', 'lead_actor_2',
        'character_id_lead_actor1', 'actor_id_lead_actor1', 'freebase_movie_id_lead_actor1', 'wikipedia_movie_id_lead_actor2', 'movie_release_date_lead_actor2',
        'character_actor_map_id_lead_actor2', 'character_id_lead_actor2', 'actor_id_lead_actor2'], axis=1, errors='ignore')

    # Swap the actors if missing
    print("Swapping actors if needed ...")

    merged_data = swap_actors_if_missing(merged_data)

    print("Processing movie genres...")

    merged_data = process_genres(merged_data)

    print("Transforming actor etchinicites to grouped ethnicities...")
    merged_data = transform_actors_ethinicities(merged_data)

    # Encode gender
    print("Encoding genders...")
    merged_data['actor_gender_lead_actor1'] = merged_data['actor_gender_lead_actor1'].map({'M': 1, 'F': 0})
    merged_data['actor_gender_lead_actor2'] = merged_data['actor_gender_lead_actor2'].map({'M': 1, 'F': 0})

    # Setup our datasets
    print("Creating train/test sets...")

    X_rating = merged_data.drop(columns=["averageRating"])
    y_rating = merged_data["averageRating"]

    # Split to train/test
    X_rating_train, X_rating_test, y_rating_train, y_rating_test = train_test_split(X_rating, y_rating, test_size=0.2, random_state=42)

    # Standardize
    numerical_columns_rating = X_rating_train.select_dtypes(include=['float64', 'int64', 'int32']).columns

    # Initialize the scaler
    scaler = StandardScaler()
    scaler_y = StandardScaler()

    X_rating_train[numerical_columns_rating] = scaler.fit_transform(X_rating_train[numerical_columns_rating])
    X_rating_test[numerical_columns_rating] = scaler.transform(X_rating_test[numerical_columns_rating])

    y_rating_train = scaler_y.fit_transform(y_rating_train.values.reshape(-1, 1))
    y_rating_test = scaler_y.transform(y_rating_test.values.reshape(-1, 1))

    y_rating_train = y_rating_train.ravel()
    y_rating_test = y_rating_test.ravel()

    # Training the RF
    model_rating_rf = RandomForestRegressor(n_estimators=500, bootstrap=True, n_jobs=-1, random_state=42)
    model_rating_rf.fit(X_rating_train, y_rating_train)
    y_rating_pred_rf = model_rating_rf.predict(X_rating_test)

    rating_r2_rf = r2_score(y_rating_test, y_rating_pred_rf)
    rating_mse_rf = mean_squared_error(y_rating_test, y_rating_pred_rf)

    print(f'Rating Model (Random Forest) R^2: {rating_r2_rf:.4f}')
    print(f'Rating Model (Random Forest) MSE: {rating_mse_rf:.4f}')

    # Saving model to onxx (for JS)
    export_model_to_onnx(model_rating_rf, X_rating_train)
    save_scalers_as_json(scaler, scaler_y)

    return model_rating_rf, rating_r2_rf, rating_mse_rf, X_rating_train