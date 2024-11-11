"""
Helpers to fetch freebase labels from Wikidata.

Executing this script will fetch the freebase labels for the actor ethnicity's 
in the character data and save it to a csv file. The script will only fetch
the data if the file does not already exist.
"""

import requests
import os

import pandas as pd
import numpy as np

from data_utils import load_data

URL = 'https://query.wikidata.org/sparql'
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
            request = requests.get(URL, params = {'format': 'json', 'query': query})
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

if __name__ == '__main__':
    path = 'data/actor_ethnicity.csv'

    if os.path.exists(path):
        print('Data already exists')
    else: 
        print('Loading data...')
        character_data, _, _ = load_data()

        print('Fetching data...')
        ids = character_data['actor_ethnicity'].dropna().unique()
        df = fetch_freebase_labels(ids)

        print('Saving data...')
        df.to_csv(path, index=False, sep=';')
