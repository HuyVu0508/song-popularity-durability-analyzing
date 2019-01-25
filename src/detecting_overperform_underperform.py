# This file is used to analyze data, learning function to approxiamate song's popualarity

import numpy as np   
import pickle 
import matplotlib.pyplot as plt


# Reading data
with open('./data/Youtube_all_data_df.pickle', 'rb') as f:  # Python 3: open(..., 'wb')
    Youtube_all_data_df = pickle.load(f)      
    f.close()       
Youtube_all_data_df = Youtube_all_data_df[Youtube_all_data_df['views']>0]

# In[]
# Run Linear Regression
years = 2018 - Youtube_all_data_df['year']   
popularity = Youtube_all_data_df['views']
peakPos = Youtube_all_data_df['peakPos']           
peakPos_numpy = np.reshape(peakPos.values,(-1,1))
year_numpy = np.reshape(years.values,(-1,1))        
X_train = np.concatenate( (peakPos_numpy, year_numpy), axis = 1 )
popularity_numpy = np.reshape( popularity.values ,(-1,1) ) 
popularity_numpy = np.log(popularity_numpy  + 0.00001)
Y_train = popularity_numpy      
#from sklearn.linear_model import LinearRegression
#model.fit(X_train, Y_train)
from sklearn.linear_model import HuberRegressor
model = HuberRegressor()
model.fit(X_train, np.reshape(Y_train,[len(Y_train),]))
model.coef_   
model.intercept_     


# In[]
# Model adjustment
# Original
#adjusted_coef = np.reshape(model.coef_, (1,2))
#adjusted_intercept = model.intercept_
# Experimenting
adjusted_coef = np.reshape([-0.05003947, -0.16015677], (1,2))
adjusted_intercept = 21.399992 - 0.2


# In[]
# Testing Linear Regression
# Plot 1 - fix year range, x-axis is peakPos
year_range = [1970, 1975]
peakPos_range = [1,100]
# Plot 1 - fix peakPos, x-axis is year
#year_range = [1960, 2018]
#peakPos_range = [1, 8]
songs_in_rank = Youtube_all_data_df.copy()
selected_samples = (songs_in_rank['peakPos'] >= peakPos_range[0]) & (songs_in_rank['peakPos'] <= peakPos_range[1]) & (songs_in_rank['year'] >= year_range[0]) & (songs_in_rank['year'] <= year_range[1])
# Select specific songs
sum(selected_samples)
selected_samples = selected_samples 
peakPos = songs_in_rank['peakPos'][selected_samples]   
years = 2018 - songs_in_rank['year'][selected_samples]     
popularity = songs_in_rank['views'][selected_samples]  
peakPos_numpy = np.reshape(peakPos.values,(-1,1))
year_numpy = np.reshape(years.values,(-1,1))        
X_val = np.concatenate( (peakPos_numpy, year_numpy), axis = 1 )
popularity_numpy = np.reshape( popularity.values,(-1,1) ) 
Y_val = np.log(popularity_numpy  + 0.00001) 
Y_prediction = np.matmul(X_val, np.transpose(adjusted_coef)) + adjusted_intercept
ratio_over = 1/0.325 # Defining thresholf for over-perform songs
ratio_under = 1/0.25  # Defining thresholf for udnder-perform songs
over_underperform_condition = (Y_val>(Y_prediction*(1 + 1/ratio_over))) | (Y_val<(Y_prediction*(1 - 1/ratio_under)))
Y_over_underperform = Y_val[over_underperform_condition]
print(sum(Y_val>(Y_prediction*(1 + 1/ratio_over))))
print(sum(Y_val<(Y_prediction*(1 - 1/ratio_under))))
# Plot on to figures
# X-axis is ranks
fig = plt.figure(figsize=(10,10))
plt.scatter(peakPos_numpy,(Y_val),color='blue',s=20, label="True Popularity")  
plt.scatter(peakPos_numpy,(Y_prediction),color='red',s=20, label="Predicted Popularity")  
#plt.scatter(peakPos_numpy[over_underperform_condition],(Y_over_underperform),color='green',s=20, label="Over/Under-perform")  
plt.legend(fontsize=20)
plt.ylim([0,25])
plt.xticks( fontsize = 15)
plt.yticks(fontsize = 15)
#plt.title("Over/Under-perform songs through ranks - log", fontsize=20, fontweight='bold')
# X-axis is years
fig = plt.figure(figsize=(10,10))
plt.scatter(2018 - year_numpy,(Y_val),color='blue',s=20, label="True Popularity")  
plt.scatter(2018 - year_numpy,(Y_prediction),color='red',s=20, label="Predicted Popularity")  
plt.scatter(2018 - year_numpy[over_underperform_condition],(Y_over_underperform),color='green',s=20, label="Over/Under-perform")  
plt.legend(fontsize=20)
plt.ylim([0,25])
plt.xticks( fontsize = 15)
plt.yticks(fontsize = 15)
#plt.title("Over/Under-perform songs through years  rank - log", fontsize=20, fontweight='bold')
## True popularity (power-law distribution)
## X-axis is ranks
#fig = plt.figure(figsize=(10,10))
#plt.scatter(peakPos_numpy,(Y_val),color='blue',s=20, label="True Popularity")  
#plt.legend(fontsize=20, loc = 1)
## plt.ylim([0,np.power(10,8)*15])
##plt.title("Log-Popularity of all Billboard Top100 against through rankings", fontsize=20, fontweight='bold')
#plt.xticks( fontsize = 10)
#plt.yticks(fontsize = 10)
## X-axis is years
#fig = plt.figure(figsize=(10,10))
#plt.scatter(2018 - year_numpy,(Y_val),color='blue',s=20, label="True Popularity")  
#plt.legend(fontsize=20)
## plt.ylim([0,np.power(10,8)*15])
##plt.title("Log-Popularity of songs ranking [80-100] against years [1960-2000]", pad  = 30,fontsize=20, fontweight='bold')
#plt.xticks( fontsize = 15)
#plt.yticks(fontsize = 15)
#plt.ylim([0,25])


# In[]
# Choosing out list of over/under-perform songs
over_perform_songs = pd.DataFrame([])
under_perform_songs = pd.DataFrame([])

songs_in_rank = Youtube_all_data_df.copy()
selected_index = (songs_in_rank['peakPos'] >= peakPos_range[0]) & (songs_in_rank['peakPos'] <= peakPos_range[1]) & (songs_in_rank['year'] >= year_range[0]) & (songs_in_rank['year'] <= year_range[1])
selected_songs = songs_in_rank[selected_index].copy()
selected_songs['predicted_views'] = np.round(Y_prediction,2)
selected_songs = selected_songs.drop(columns=['title & artist', 'scraped', 'rank'])
selected_songs['predicted_true_ratio'] = np.round(Y_val/Y_prediction,2)

over_perform_songs_index = Y_val>(Y_prediction*(1 + 1/ratio_over))
under_perform_songs_index = Y_val<(Y_prediction*(1 - 1/ratio_under))
over_perform_songs = over_perform_songs.append(selected_songs[over_perform_songs_index].copy())
under_perform_songs = under_perform_songs.append(selected_songs[under_perform_songs_index].copy())

# Some analysis of over/under-perform songs
np.array(np.unique(over_perform_songs['peakPos'], return_counts=True)).T
np.array(np.unique(under_perform_songs['peakPos'], return_counts=True)).T
np.array(np.unique(np.round(over_perform_songs['year'],0), return_counts=True)).T
np.array(np.unique(np.round(under_perform_songs['year'],0), return_counts=True)).T



# In[]
#### Anaylzing big artists

# Artist having the most Billboard songs
Youtube_all_data_df.groupby('artist').count().sort_values(['title'], ascending=False).head(10)
Youtube_all_data_df[Youtube_all_data_df['artist']=="The Beatles"]

# Analyzing selected big artists
#year_range = [1962, 1972]
#peakPos_range = [1,50]
#artist = "The Beatles"
#year_range = [2005, 2015]
#peakPos_range = [1,100]
#artist = "Glee Cast"
year_range = [1960, 2000]
peakPos_range = [1,100]
artist = "The Rolling Stones"
#year_range = [1960, 2010]
#peakPos_range = [1,50]
#artist = "Michael Jackson"
#title =
songs_in_rank = Youtube_all_data_df.copy()
selected_samples = (songs_in_rank['peakPos'] >= peakPos_range[0]) & (songs_in_rank['peakPos'] <= peakPos_range[1]) & (songs_in_rank['year'] >= year_range[0]) & (songs_in_rank['year'] <= year_range[1])
selected_samples_big_artist = selected_samples & (songs_in_rank['artist'] == artist)
print(sum(selected_samples_big_artist))
#selected_samples = selected_samples & (songs_in_rank['artist'] == artist) & (songs_in_rank['title'] == title)

# All songs
peakPos = songs_in_rank['peakPos'][selected_samples]   
years = 2018 - songs_in_rank['year'][selected_samples]     
popularity = songs_in_rank['views'][selected_samples]  
peakPos_numpy = np.reshape(peakPos.values,(-1,1))
year_numpy = np.reshape(years.values,(-1,1))        
X_val = np.concatenate( (peakPos_numpy, year_numpy), axis = 1 )
popularity_numpy = np.reshape( popularity.values,(-1,1) ) 
Y_val = np.log(popularity_numpy  + 0.00001) 
# Big artist
peakPos_big_artist = songs_in_rank['peakPos'][selected_samples_big_artist]   
years_big_artist = 2018 - songs_in_rank['year'][selected_samples_big_artist]     
popularity_big_artist = songs_in_rank['views'][selected_samples_big_artist]  
peakPos_numpy_big_artist = np.reshape(peakPos_big_artist.values,(-1,1))
year_numpy_big_artist = np.reshape(years_big_artist.values,(-1,1))        
X_val_big_artist = np.concatenate( (peakPos_numpy_big_artist, year_numpy_big_artist), axis = 1 )
popularity_numpy_big_artist = np.reshape( popularity_big_artist.values,(-1,1) ) 
Y_val_big_artist = np.log(popularity_numpy_big_artist  + 0.00001) 

## Plot on to figures
# X-axis is years
fig = plt.figure(figsize=(10,10))
plt.scatter(2018 - year_numpy,(Y_val),color='orange',s=20, label="All songs")  
plt.scatter(2018 - year_numpy_big_artist,(Y_val_big_artist),color='blue',s=20, label=artist)  
#plt.title("Songs from big artists - log", fontsize=20, fontweight='bold')
plt.legend(fontsize=20, loc = 4)
plt.ylim([0,25])
plt.xticks( fontsize = 15)
plt.yticks(fontsize = 15)
















