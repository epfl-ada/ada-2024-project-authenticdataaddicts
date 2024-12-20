# ada-2024-project-authenticdataaddicts

[Project website](https://mina-tang.github.io/ada-2024-project-authenticdataaddicts/)

## Recipe for Success: Uncovering the actors behind a hit movie

### Abstract
While a movie storyline is often seen as its core, other factors like cast or genres may influence its success. This project examines what leads to a movies' success, with an emphasis on the effect of lead actors.
We begin by studying intrinsic characterics of movies (genre, runtime, language, etc.) to get a general sense of the data we are handling. 
We then investigate the lead actors' physical and demographic characteristic and whether they are correlated to a movie's success. We make sure to compare the differences between lead actors and all actors. By also analyzing these traits across different genres, we aim to identify patterns linking specific actor attributes to successful movies. We also investigate whether the country of origin of a movie is correlated with its sucess. Movie success is defined by two metrics: box office performance and critical reception. Since high-grossing movies may not always be well-received, and vice versa, separating performance from reception could yield new insight. Ultimately, our goal is not only to determine whether actor attributes influence a movie’s success, but also identify characteristics of movies themselves that may affect its success. 

### Research questions
- What genres and movie length lead to the most successful movies?
- Is there a correlation between lead actors' physical and demographic characteristics and the success of the movie they star in?
- If there is indeed a correlation as mentioned in the above question, does it vary depending on the genre of the movie
- Are movies genres that are more successful in the box office also more successful in terms of ratings?

### Additional datasets
In addition to the CMU dataset, we also used:
#### [The IMDb non-commercial dataset (principles, ratings & names)](https://datasets.imdbws.com)
It gives us the lead actors, critic ratings, and names corresponding to the unique actor ID, respectively.
#### [TMDb Kaggle dataset](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies?resource=download)
This is a comprehensive database of over one million movies and contains informations such as movie's rating and box office performance. It complements the CMU and IMDb datasets when a movie's rating or box office is missing. 
#### [US Bureau of Labor Inflation Statistics](https://data.bls.gov/timeseries/CUUR0000SA0L1E?output_view=pct_12mths)
This gives us the US inflation numbers dating back to 1958. It can be used in order to calculate inflation adjusted box office results. (To access the data, click on the provided link, then More Formatting Options, and choose 1957-2012 with annual data, and save in a comma delimited file.)

### Methods
#### Task 1: Getting to know the data
This first crucial step allows us to determine whether we have all the necessary informations in the CMU dataset, or whether it must be complemented with outside datasets. We noticed a non-negligeable amount of missing box office results, and there are no information on lead actors or movie ratings. Hence, we decided to use the TMDb dataset to complement our box office results, and the IMDb datasets to obtain movie ratings as well as information on lead actors. 
#### Task 2: Data pre-processing
We first needed to merge the datasets. We did this and split it into 5 datasets:
- full_movie_data_preprocessed: the data of all our movies after preprocessing (handling NaNs, outliers, etc.), including merged ratings, box office, inflation data, etc. 
- full_character_data_preprocessed: similar to the first one but for character data.
- subset_movies: contains only movies with information on at least one main actor.
- subset_characters: filtered to contain only characters for which (all) information is available.
- lead_actors: contains the lead actors and their corresponding information.
#### Task 3: Deeper Analysis
We conducted a deeper analysis of the preprocessed data, to see if we could already find some interesting patterns relevant to our research questions. We did the following sub-tasks:
  - 3.1: *General deeper analysis and visualisation* of the preprocessed data.
  - 3.2: *Comparison between lead actors and all actors.*
    - 3.2.1: *Gender distribution of actors*
    - 3.2.2: *Main ethnicities*
    - 3.2.3: *Height and age*
  - 3.3: *Regression analysis of actor and movie attributes*.
  - 3.4: *Box office and inflation*: adjust the box office on inflation, compare old and recent movies.
  - 3.5: *Rating vs revenue analysis depending on genre*.

#### Task 4: Prediction
Thanks to all the data gathered earlier, we implemented a model that predicts the rating of a movie, given its attributes and cast. This is done by implementing a random forest on the chosen features. 


#### Task 5: Datastory
We implement a data story that can be seen (here)[https://mina-tang.github.io/ada-2024-project-authenticdataaddicts/]


## Setup and usage

### Data story
As previously mentioned, here is the link to our (Recipe for Success)[https://mina-tang.github.io/ada-2024-project-authenticdataaddicts/].

### Files and Folders Description
* `results.ipynb`: the main notebook with all our results.
* `data/`: folder containing all datasets (after loading them). See the dedicated README for more details.
* `data/preprocessed`: folder containing the preprocessed data (to directly do the analysis).
* `src/`: folder containing the utility functions for the different steps of the project:
    - `data_completion.py`: for merging and completing our datasets;
    - `data_evaluation.py`: for propensity matching
    - `data_fetching.py`: to fetch and query some of the datasets;
    - `data_loading.py`: to load the CMU movie dataset;
    - `data_preprocessing`: useful tools for preprocessing;
    - `data_visualization.py`: useful tools for data visualization.
    - `model_convert_to_onxx.py`: ???
    - `model_random_forest.py`: to implement the prediction model.

### Get started
1. Clone this github repository.
2. Make sure to have all libraries from the requirements.txt [file](requirements.txt) installed. If not, run the following command in your terminal:
```
pip install -r requirements.txt
```
3. All data not in the repo should be directly downloadable via the second cell of the main notebook (so no need to download them separately).
4. To reproduce our results, run the jupyter notebook `results.ipynb`. The first part (before Deep Analysis) could take a few minutes (due to the large size of some of the datasets).

### Contributors
Ezra Baup (estelle.baup@epfl.ch)
Colin Berger (colin.berger@epfl.ch)
Florian Comte (florian.comte@epfl.ch)
David Gauch (david.gauch@epfl.ch)
Mina Tang (mina.tang@epfl.ch)
