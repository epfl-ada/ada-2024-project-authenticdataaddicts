import requests
import os
import pandas as pd
import numpy as np
import gzip
import shutil
import kagglehub
import tarfile

# --- WIKIDATA ---

WIKIDATA_URL = 'https://query.wikidata.org/sparql'
BATCH_SIZE = 50

def get_wikidata_query(ids: list[str]) -> str:
    """Generates a SPARQL query to fetch the labels for the given freebase IDs.

    Args:
        ids (list[str]): The list of freebase IDs

    Returns:
        str: The SPARQL query
    """
    query = "SELECT ?s ?sLabel ?freebaseID\nWHERE {\n  VALUES ?freebaseID {\n"
    query += '\n'.join(map(lambda x: f'    "{x}"', ids))
    query +='\n  }\n  ?s wdt:P646 ?freebaseID .\n  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }\n}\n'
    
    return query

def fetch_freebase_labels(ids: list[str]) -> pd.DataFrame:
    """Fetches the freebase labels for the given freebase IDs.

    Args:
        ids (list[str]): The list of freebase IDs

    Returns:
        pd.DataFrame: The DataFrame containing the freebase labels, freebase IDs and URLs
    """
    batches = np.array_split(ids, len(ids) // BATCH_SIZE)
    results = []

    for batch in batches:
        query = get_wikidata_query(batch)
        try:
            request = requests.get(WIKIDATA_URL, params = {'format': 'json', 'query': query})
            json = request.json()
            data = json['results']['bindings']
            data = map(lambda x: {'Freebase ID': x['freebaseID']['value'], 'URL': x['s']['value'], 'Label': x['sLabel']['value']}, data)
            results.extend(data)
        except Exception as e:
            print('Error fetching data from Wikidata')
            print(results)
            print(e)

    df = pd.DataFrame(results)

    return df

# --- IMDB ---

def download_imdb(filename):
    """Download specified IMDb datasets and save it to the data folder if it doesn't already exist.

    Args: 
        filename (str): Name of the IMDb dataset to download.

    Returns: None
    """
    file_path = f"./data/{filename}"

    # Check if the file already exists
    if os.path.exists(file_path):
        print(f"{filename} already exists. Skipping download.")
        return

    url = f"https://datasets.imdbws.com/{filename}.gz"
    r = requests.get(url)

    # Download and extract the file if not already present
    with open(f"./data/{filename}.gz", "wb") as f:
        f.write(r.content)

    with gzip.open(f"./data/{filename}.gz", "rb") as fz:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(fz, f)

    os.remove(f"./data/{filename}.gz")
    print(f"{filename} has been downloaded and extracted.")

# --- TMDB --- 

def download_tmdb():
    """Download TMDb dataset from Kaggle and save it to the data folder if it doesn't already exist.

    Args: None

    Returns: None
    """
    file_path = "./data/tmbd_movies.csv"

    # Check if the file already exists
    if os.path.exists(file_path):
        print("TMDb dataset already exists. Skipping download.")
        return

    path = kagglehub.dataset_download("asaniczka/tmdb-movies-dataset-2023-930k-movies")
    shutil.copy(path + "/TMDB_movie_dataset_v11.csv", file_path)
    print("TMDb dataset has been downloaded.")

# --- CMU ---

def download_cmu(filename="MovieSummaries.tar.gz"):
    """Download CMU dataset (MovieSummaries) and save it to the data folder if the required files don't already exist.

    Args: 
        filename (str): Name of the CMU dataset file to download. Default is "MovieSummaries.tar.gz".

    Returns: None
    """
    required_files = [
        './data/movie.metadata.tsv',
        './data/plot_summaries.txt',
        './data/character.metadata.tsv'
    ]
    
    # Check if all required files exist
    if all(os.path.exists(file) for file in required_files):
        print("All required files of CMU are already present. Skipping download.")
        return

    file_path = f"./data/{filename}"

    # Download the dataset if not all required files exist
    url = "https://www.cs.cmu.edu/~ark/personas/data/" + filename
    r = requests.get(url)

    # Check if the download was successful
    if r.status_code == 200:
        # Save the downloaded .tar.gz file
        with open(file_path, "wb") as f:
            f.write(r.content)
        print(f"{filename} has been downloaded.")

        # Extract the tar.gz file into the data folder without subfolders
        with tarfile.open(file_path, "r:gz") as tar:
            for member in tar.getmembers():
                # Extract each file directly into the data folder
                member.name = os.path.basename(member.name)
                tar.extract(member, path="./data/")
            print(f"Files have been extracted directly into './data/'.")

        os.remove(file_path)
        print(f"{filename} remains in the data folder after extraction.")
    else:
        print(f"Failed to download {filename}. Status code: {r.status_code}")