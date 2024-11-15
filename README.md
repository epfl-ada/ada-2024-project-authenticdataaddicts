# ada-2024-project-authenticdataaddicts

## Recipe for Success: Uncovering the actors behind a hit movie

### Abstract
This project examines what leads to a movies' success, with an emphasis on the effect of actors. We begin by studying intrinsic characterics of movies, such as genre and runtime, and how they correlate to a movie's success in order to establish general patterns. We then investigate how lead actors characteristics (height, ethnicity, gender, age) might influence a movie's success. By analyzing these traits across different genres, we aim to identify patterns linking specific actor attributes to successful movies. Data is gathered from CMU’s movie database, the IMDb non-commercial and the TMDb Kaggle datasets. Movie success is defined by two metrics: box office performance and critical reception. Since high-grossing movies may not always be well-received, and vice versa, separating performance from reception could yield new insight. Ultimately, our goal is to determine how actor attributes influence a movie’s success, providing valuable understanding for casting decisions and broader industry trends.

### Research questions
- What genres and movie length lead to the most successful movies?
- What correlation is there between a lead actor's physical and demographic characteristics and the success of the movie?
- Do the type of actors who star in critically aclaimed movies different than those who star in box office hits? 
- How does the genre of a movie influence our results? 
- Do the relationships between actor characteristics and movie success change depending on time period?

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
  - 3.3: *Bow office and inflation*: adjust the box office on inflation, compare old and recent movies.
  - 3.4: *Low vs high rating/revenue analysis*.

These are future sub-tasks we wish to implement to complete the deep analysis.
  - 3.5: *Effect of gender*: does having lead actors of the same gender help or hurt a movie's performance? We could first do a histogram for the average ratings/box office results for movies where both lead actors are the same gender, and those where both lead actors are different genders. Second, we would do a pie chart with these two categories for the top movies across different genres.
  - 3.6: *Ethnicity and height*. To visualize this, we can use stacks (for height, we would use ranges rather than an exact number). We can compare the stack of the average actor with the stack of the lead actors of the most successful movies. 
  - 3.7: *To be good or to be popular?* Analyzing the distribution of movie ratings compared to box office results. Here, we want to analyze the distribution of movies ratings compared to box office results. Are successful movies in one category necessarily successful in the other? In order to find out, we want to implement a scatterplot heatmap with one axis representing movie ratings and the other box office results. This is a great visualisation tool to see where most movies place on the 2D map.

### Proposed timeline:
15.11: Tasks 1, 2, 3.1-3.4
29.11: Homework 2
TODO
20.12: Deadline P3

### Organization with the team (internal milestones until P3)
