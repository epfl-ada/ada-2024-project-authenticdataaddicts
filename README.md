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
We conducted a deeper analysis of the preprocessed data, to see if we could already find some interesting patterns relevant to our research questions.
#### Task 4: Implementation and putting it all together
Here we want to expand on what we did in Task 3, either by perfecting the graphs and models we made or by going deeper. In order to this, we will implement the following sub-tasks:
##### Box office and Inflation
We established that the USA was by far the number 1 filmmaker, as such, adjusted box office is indexed on US inflation and given in dollars. We want to see how old movies compare to new movies when levelling the playing field. This can be done through scatterplots of both the average box office return, adjusted for inflation, and the average box office of the ten highest grossing movies in a given year. This is because we may not have information about very old obscure and unsuccessfull movies. By adjusting for inflation, we ensure that our data is not unfairly biaised towards recent releases. 
##### The Effects of Gender
Does having lead actors of the same gender help or hurt a movie's performance? In order to determine this with two vizualizations:
- First, a histogram for the average ratings/box office results for movies where both lead actors are the same gender, and those where both lead actors are different genders.
- Second, a pie chart with these two categories for the top movies across different genres.
##### Ethnicity and Height
To visualize this, we can use stacks (for height, we would us ranges rather than an exact number). We can compare the stack of the average actor with the stack of the lead actors of the most successful movies. 
##### To be Good or to be Popular? 
Here, we want to analyze the distribution of movies ratings compared to box office results. Are successful movies in one category necessarily successful in the other? In order to find out, we want to implement a scatterplot heatmap with one axis representing movie ratings and the other box office results. This is a great visualisation tool to see where most movies place on the 2D map.

### Proposed timeline:
15.11: Tasks 1-4
29.11: Homework 2
TODO
20.12: Deadline P3

### Organization with the team (internal milestones until P3)
