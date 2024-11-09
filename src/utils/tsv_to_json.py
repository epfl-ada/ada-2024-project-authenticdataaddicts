

import pandas as pd

def tsv_to_json(tsv_file, json_file, columns = [], has_header=False):
    """
    Converts a TSV file to a JSON file.

    Parameters:
    - tsv_file (str): Path to input file
    - json_file (str): Path to output file
    - columns (list): List of column names, do not specify if TSV file has header
    - has_header (bool): True if TSV file has header, False otherwise
    """
    if has_header:
        df = pd.read_csv(tsv_file, sep='\t')
    else:
        df = pd.read_csv(tsv_file, sep='\t', header=None)
        df.columns = columns
    
    # Convert to JSON
    df.to_json(json_file, orient='records', lines=False, indent=4)
    
    print("Success")

#Convert our TSV files to JSON: 

#CMU database:
tsv_to_json('data/movie.metadata.tsv', 'data/movies_CMU.json', columns=["wikiId", "freebaseId", "title", "releaseDate", "revenue", "runtime", "languages", "countries", "genres"])
tsv_to_json('data/character.metadata.tsv', 'data/characters_CMU.json', columns=["wikiId", "freebaseId", "releaseDate", "characterName", "actorDateOfBirth", 
"actorGender", "actorHeight", "actorEthnicity", "actorName", "actorAgeAtMovieRelease", "freebaseCharacterActorMapID", "freebaseCharacterID", "freebaseActorID"])

#IMDb database:
tsv_to_json('data/title.principals.tsv', 'data/principals_IMDb.json', has_header = True)
tsv_to_json('data/title.ratings.tsv', 'data/ratings_IMDb.json', has_header = True)
