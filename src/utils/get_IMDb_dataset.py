import requests
import gzip
import shutil
import os
def get_IMDb_dataset(filename):
    url = "https://datasets.imdbws.com/" + filename + ".gz"
    r = requests.get(url)
    #r.encoding = "utf-8"

    with open("./data/"+filename+".gz", "wb") as f:
        f.write(r.content)
    with gzip.open("./data/"+filename+".gz", "rb") as fz:
        with open("./data/"+filename, "wb") as f:
           shutil.copyfileobj(fz, f)

    os.remove("./data/"+filename+".gz")