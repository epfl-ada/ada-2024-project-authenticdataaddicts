import kagglehub
import shutil

def import_TMDb():
    path = kagglehub.dataset_download("asaniczka/tmdb-movies-dataset-2023-930k-movies")
    shutil.copy(path+"/TMDB_movie_dataset_v11.csv", "./data/tmbd_movies.csv")