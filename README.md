# ada-2024-project-authenticdataaddicts

## Super Amazing Title Here

### Abstract
This project examines what leads to a movies' sucess. We will first study the intrinsic characterics of the movie itself, such as its genre and its length, and how they corelate to a movie's success. We will then analyze the role of the lead actors to deterine how the actors' characteristics (height, ethnicity, gender, age) might influence a movie's success. By analyzing these traits across different genres, we aim to identify patterns linking specific actor attributes to successful movies. Data is gathered from CMU’s movie database, the IMDb non-commercial dataset, and the TMDb Kaggle dataset. Movie success is defined by two metrics: box office performance and critical reception. Since high-grossing movies may not always be well-received, and vice versa, separating performance from reception could yield to new insight. Ultimately, our goal is to determine how actor attributes influence a movie’s success, providing valuable insights for casting decisions and broader industry trends.

### Research questions
- What genres and movie length lead to the most successful movies?
- What correlation is there between a lead actor's physical and demographic characteristics and the success of the movie?
- Do the type of actors who star in critically aclaimed movies different than those who star in box office hits? 
- How does the genre of a movie influence our results? 
- Do the relationships between actor characteristics and movie success change depending on time period?

### Additional datasets
In addition to the CMU dataset, we also used:
#### [The IMDb non-commercial dataset (principles, ratings & names)](https://datasets.imdbws.com)
These datasets give us the lead actors for each movie as well as critic ratings, respectively. 
#### [TMDb Kaggle dataset](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies?resource=download)
This is a comprehensive database of over one million movies and contains information movies, including a movie's rating and box office performance. 

### Method
#### Getting to know the data
This first step is crucial, as it allows us to determine whether we have all the necessary information in our dataset, or whether it must be complemented with outside datasets. In the CMU dataset, we have many missing box office results ((pourcentage ici)) , and we do not have information on lead actors or movie ratings. As such, we decided to utilise the TMDb dataset to complement our box office results, and the IMDb datasets in order to obtain movie ratings as well as information on lead actors. 
#### Data pre-processing
The first step we needed to do is to merge the datasets. 

### Timeline:

### Organization with the team (internal milestones until P3)
