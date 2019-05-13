# Movies recomender based on previous watched movies

# Import the libraries
import pandas as pd

# import the datasets
movies_data = pd.read_csv("ratings.tsv", sep = '\t', names = ['usr_id', 'item_id', 'rating', 'timestamp'])
movies_data.head()

movies_titles = pd.read_csv("Movie_Id_Titles.csv")
movies_titles.head()

# merge the two datasets to get main dataset
dataset = pd.merge(movies_data, movies_titles, on = 'item_id')
dataset.describe()

# the average rating and nomber of rating for each movie
ratings = pd.DataFrame(dataset.groupby('title')['rating'].mean())
ratings['number_of_ratings'] = pd.DataFrame(dataset.groupby('title')['rating'].count())
#ratings.head()

# plot histograms
import matplotlib.pyplot as plt
ratings['rating'].hist(bins = 50)
ratings['number_of_ratings'].hist(bins = 100)

# plot the relationship between the rating of a movie and the number of ratings
import seaborn 
seaborn.jointplot(x = 'rating', y = 'number_of_ratings', data = ratings)

ratings.describe()

# movies matrix
movies_matrix = dataset.pivot_table(columns = 'title', index = 'usr_id', values = 'rating') 
#movies_matrix.head()

# the most rated movies
ratings.sort_values(by = 'number_of_ratings', ascending = False).head(10)

# Let's work on only two movies (for simplicity) Air Force One and Contact movies

# 1- the two movies ratings dataframes
AFO_rating = movies_matrix['Air Force One (1997)']
Contact_rating = movies_matrix['Contact (1997)']

# 2- find correlations of AFO and Contact
similar_to_AFO = movies_matrix.corrwith(AFO_rating)
similar_to_contact = movies_matrix.corrwith(Contact_rating)

# 3- transform correlation results into dataframes and drop NAN 
AFO_corr = pd.DataFrame(similar_to_AFO, columns = ['correlation'])
AFO_corr.dropna(inplace = True)

Contact_corr = pd.DataFrame(similar_to_contact, columns = ['correlation'])
Contact_corr.dropna(inplace = True)

# 4- join nomber of ratings to define a threshold on the number of ratings
AFO_corr = AFO_corr.join(ratings['number_of_ratings'])
Contact_corr = Contact_corr.join(ratings['number_of_ratings'])

# 5- find the movies similar to AFO and Contact to recommend them
AFO_corr[AFO_corr['number_of_ratings'] > 100].sort_values(by = 'correlation', ascending = False).head(10)

Contact_corr[Contact_corr['number_of_ratings'] > 100].sort_values(by = 'correlation', ascending = False).head(10)





