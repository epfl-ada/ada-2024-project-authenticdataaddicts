import kagglehub
import shutil

def import_TMDb():
    """Download TMDb dataset from Kaggle and saves it to the data folder

    Args: None

    Returns: None
    """
    path = kagglehub.dataset_download("asaniczka/tmdb-movies-dataset-2023-930k-movies")
    shutil.copy(path+"/TMDB_movie_dataset_v11.csv", "./data/tmbd_movies.csv")