import pandas as pd

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
