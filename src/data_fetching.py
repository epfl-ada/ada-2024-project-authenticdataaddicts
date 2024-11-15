import requests
import os
import pandas as pd
import numpy as np
import gzip
import shutil
import kagglehub

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
    """Download specifiet IMDb datasets and saves it to the data folder

    Args: 
        filename (str): Name of the IMDb dataset to download

    Returns: None
    """
     
    url = "https://datasets.imdbws.com/" + filename + ".gz"
    r = requests.get(url)
    #r.encoding = "utf-8"

    with open("./data/"+filename+".gz", "wb") as f:
        f.write(r.content)
    with gzip.open("./data/"+filename+".gz", "rb") as fz:
        with open("./data/"+filename, "wb") as f:
           shutil.copyfileobj(fz, f)

    os.remove("./data/"+filename+".gz")


# --- TMDB --- 

def download_tmdb():
    """Download TMDb dataset from Kaggle and saves it to the data folder

    Args: None

    Returns: None
    """
    path = kagglehub.dataset_download("asaniczka/tmdb-movies-dataset-2023-930k-movies")
    shutil.copy(path+"/TMDB_movie_dataset_v11.csv", "./data/tmbd_movies.csv")