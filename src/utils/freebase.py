import requests
import os

import pandas as pd
import numpy as np

from data_utils import load_data

URL = 'https://query.wikidata.org/sparql'
BATCH_SIZE = 50

def get_wikidata_query(ids: list[str]) -> str:
    query = "SELECT ?s ?sLabel ?freebaseID\nWHERE {\n  VALUES ?freebaseID {\n"
    query += '\n'.join(map(lambda x: f'    "{x}"', ids))
    query +='\n  }\n  ?s wdt:P646 ?freebaseID .\n  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }\n}\n'
    
    return query

def fetch_freebase_labels(ids: list[str]) -> pd.DataFrame:
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
        df = fetch_freebase_labels(ids[:50])

        print('Saving data...')
        df.to_csv(path, index=False, sep=';')

