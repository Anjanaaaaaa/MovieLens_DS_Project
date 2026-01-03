# %% [markdown]
# #  MovieLens Case Study: Movie Talkies – Classics

# %% [markdown]
# ## Business Context
# MovieLens operates in the internet and entertainment domain and provides an online database of films, TV series, and streaming content.
# 
# The company plans to launch a special edition titled **"Movie Talkies: Classic"**, focusing on movies that are over a decade old.
# 
# ## Objective
# As a Data Scientist, the objective is to analyze user ratings, movie information, and user demographics to understand:
# - Audience preferences
# - Popular genres
# - Rating trends for classic movies

# %% [markdown]
# ### Step 1: Import packages

# %%
# importing necessary libraries
import numpy as np
import pandas as pd

# importing matplotlib for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# setting up the plotting style
sns.set_style("whitegrid")
%matplotlib inline

# %% [markdown]
# ### Step 2: Read datasets

# %%
# loading movie dataset
movie=pd.read_csv('movie.csv')
movie.head()

# %%
# loading ratings dataset
ratings=pd.read_csv('ratings.csv')
ratings.head()

# %%
# loading user dataset
user=pd.read_csv('user.csv')
user.head()

# %% [markdown]
# ### Step 3: Overview of the Datasets

# %%
# shape of datasets
print('Movie:', movie.shape, 'Ratings:', ratings.shape, 'User:', user.shape)

# %%
# info of movie dataset
print(movie.info())

# %%
# info of user dataset
print(user.info())

# %%
# info of ratings dataset
print(ratings.info())

# %% [markdown]
# Observations:
# - The above dataset have no missing values in it.
# - Dtypes are correctly mentioned for respective features.
# - To note: release date column is currently of type object. This column should ideally be converted to datetime for time-based analysis. Conversion will be in the data preprocessing stage.

# %%
# summary statistics of datasets
print(user.describe())
print(ratings.describe())

# %% [markdown]
# Observations:
# - Age range: 7 to 73 means a wide demographic coverage
# - Average rating ≈ 3.53 means users generally rate movies positively
# - Median rating = 4 means slight positive skew

# %% [markdown]
# ### Step 4: Data insights

# %%
# Exploratory Data Analysis (EDA) on Ratings Dataset

# total number of ratings
print('Total number of ratings:', len(ratings))

# number of unique users who rated movies
print('Number of unique users who rated movies:', ratings['user id'].nunique())

# number of unique movies that received ratings)
print('Number of unique movies that received ratings:', ratings['movie id'].nunique())

# minimum and maximum rating values
print('Minimum rating value:', ratings['rating'].min())
print('Maximum rating value:', ratings['rating'].max())

# %% [markdown]
# Rating Overview Insights:
# - The dataset contains 100,000 ratings provided by 943 unique users.
# - A total of 1,682 unique movies have received ratings.
# - Ratings range from 1 to 5, indicating full utilization of the rating scale.

# %%
# Movie popularity (how many ratings each movie gets)
ratings_per_movie = ratings['movie id'].value_counts()
ratings_per_movie.head(10)

# %% [markdown]
# Observations:
# - Rating activity is unevenly distributed across movies, with a small number of movies receiving a large proportion of total ratings.
# - This suggests that user engagement is concentrated on certain movies based on rating frequency.
# - Further analysis is required to understand whether popular movies also receive higher average ratings.

# %%
# average rating per movie
avg_rating_per_movie = ratings.groupby('movie id')['rating'].mean()
print(avg_rating_per_movie.head())

# top 10 highest-rated movies (overall)
top_10_highest_rated = avg_rating_per_movie.sort_values(ascending=False).head(10)
print(top_10_highest_rated)

# %% [markdown]
# ### Step 5: Bias Handling

# %%
# filtering movies with at least 50 ratings
filtered_movies = ratings_per_movie[ratings_per_movie >= 50]
print(filtered_movies)

# From average ratings of all movies, we will select only those movies whose IDs appear in filtered_movies(>=50).
# average ratings only for those movies with at least 50 ratings
filtered_avg_ratings = avg_rating_per_movie[filtered_movies.index]
print(filtered_avg_ratings)

# %%
# from those, find the top 10 highest-rated movies
top_10_movies = filtered_avg_ratings.sort_values(ascending=False).head(10)
print(top_10_movies)

# %% [markdown]
# Hence, the above are the top 10 highest-rated movies among those that have received at least 50 ratings, making them more reliable in terms of user feedback.

# %%
# attaching Movie Titles

movie_titles = movie[['movie id', 'movie title']]
print(movie_titles.head())

# %%
top_10_movies_df = (top_10_movies.reset_index())
top_10_movies_df = top_10_movies_df.merge(movie_titles, on='movie id')
print(top_10_movies_df)

# %%
top_10_movies_df[['movie title', 'rating']]

# %% [markdown]
# After filtering out sparsely rated movies, the following titles emerged as the highest-rated classics. These movies not only received strong average ratings but also had sufficient audience engagement, making the results reliable.
# 
# The list includes critically acclaimed classics such as *Schindler's List*, *The Shawshank Redemption*, *Casablanca*, and *12 Angry Men*, indicating a strong preference among users for timeless storytelling and high-quality cinema.

# %% [markdown]
# ### Step 6: Visualization

# %% [markdown]
# #### Ratings per movie

# %%
import matplotlib.pyplot as plt

ratings_per_movie.hist(bins=50)
plt.title('Distribution of Ratings per Movie')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Movies')
plt.show()

# %% [markdown]
# The histogram shows a highly skewed distribution of the number of ratings per movie. Most movies have received very few ratings, while only a small subset of movies are rated by a large number of users.
# This long-tail pattern highlights the presence of popularity bias in the dataset. To ensure reliability in subsequent analysis, movies with fewer than 50 ratings were excluded when identifying top-rated movies.

# %% [markdown]
# #### Audience Analysis

# %%
# Plot age distribution of users
plt.figure(figsize=(8, 5))
plt.hist(user['age'], bins=10)
plt.xlabel('Age')
plt.ylabel('Number of Users')
plt.title('Age Distribution of MovieLens Users')
plt.show()

# %% [markdown]
# Observation:
# The age distribution shows that the majority of users fall within the 20–50 age range. This indicates that most of the audience engaging with classic movies consists of young to middle-aged adults, suggesting these movies continue to appeal strongly beyond their original release period.

# %% [markdown]
# #### Gender Analysis

# %%
# merge ratings with user data
ratings_users = ratings.merge(user, on='user id')

# average rating by gender
avg_rating_by_gender = ratings_users.groupby('gender')['rating'].mean()
avg_rating_by_gender

# %%
# visualization
avg_rating_by_gender.plot(kind='bar', title='Average Rating by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Rating')
plt.show()

# %% [markdown]
# Observation: The average ratings given by male and female users are very similar, indicating no significant gender-based bias in rating behavior. Both genders show comparable engagement with classic movies.

# %% [markdown]
# #### Occupation-wise User Engagement

# %%
# Number of ratings by occupation
ratings_by_occupation = ratings_users['occupation'].value_counts()
ratings_by_occupation.head(10)

# %%
# visualization
ratings_by_occupation.head(10).plot(kind='bar', figsize=(8,5))
plt.xlabel('Occupation')
plt.ylabel('Number of Ratings')
plt.title('Top 10 Occupations by Number of Ratings')
plt.show()

# %% [markdown]
# Observation: Users from occupations such as students, educators, engineers, and programmers contribute the highest number of ratings. This indicates that working professionals and students form a major segment of the audience consuming classic movies.

# %% [markdown]
# ### Conclusion:
# 
# The MovieLens dataset analysis provides valuable insights into both classic movies and the audiences engaging with them. The platform is primarily used by young to middle-aged adults, with balanced participation across genders. A small subset of movies attracts the majority of ratings, highlighting strong popularity bias in user engagement.
# 
# By filtering out sparsely rated movies, reliable top-rated classic movies were identified, including critically acclaimed titles such as *Schindler's List*, *The Shawshank Redemption*, and *12 Angry Men*. These findings help MovieLens understand audience preferences and engagement patterns, enabling better content curation and recommendation strategies.


