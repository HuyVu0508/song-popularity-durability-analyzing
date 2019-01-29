# Scraping Youtube data

import numpy as np
import pickle


####################################    
##### Youtube Data    
partion = np.array(range(5000,27001,1000))
for i in partion:
    with open('./data/Youtube Data/Old Data/Youtube_view_all_songs_' + str(i) + '.pickle','rb') as f:  # Python 3: open(..., 'rb')
        all_top_songs_list = pickle.load(f)    
        print('Partion '+ str(i) + "/" + str(len(all_top_songs_list)))    


# Loop through all Youtube data    
with open('./data/Youtube Data/Youtube_view_all_songs_' + str(i) + '.pickle','rb') as f:  # Python 3: open(..., 'rb')
    all_top_songs_list = pickle.load(f)    
    print('Partion '+ str(i) + "/" + str(len(all_top_songs_list))) 

with open('./data/Youtube Data/Old Data/Youtube_view_all_songs_15000.pickle','rb') as f:  # Python 3: open(..., 'rb')
    all_top_songs_list = pickle.load(f)    
    print(str(len(all_top_songs_list))) 
    all_top_songs_list[(len(all_top_songs_list)-3):(len(all_top_songs_list))]
    all_top_songs_list[1:3]
      
#Youtube_all_data = []        
for i in range(len(data_files)):
    with open('./data/Youtube Data/New Data/'+data_files[i]+'.pickle','rb') as f:  
        all_top_songs_list = pickle.load(f)    
        Youtube_all_data += all_top_songs_list
        print("Loading " + data_files[i] + " / "+ str(len(all_top_songs_list)))
    
    
# Turn into dataframe
import pandas as pd
Youtube_all_data_df = pd.DataFrame(Youtube_all_data)
Youtube_all_data_df['title & artist'] = Youtube_all_data_df['title'] + Youtube_all_data_df['artist']
Youtube_all_data_df = Youtube_all_data_df.drop_duplicates()    
# Get all_top_songs_list and turn into dataframe
with open('./data/all_top_songs_list.pickle','rb') as f: 
    all_top_songs_list = pickle.load(f)    
all_top_songs_list_df = pd.DataFrame(all_top_songs_list)    
all_top_songs_list_df['title & artist'] = all_top_songs_list_df['title'] + all_top_songs_list_df['artist']
# Loop through all_top_songs_list and update
songs_scraped_index = all_top_songs_list_df['title & artist'].isin(Youtube_all_data_df['title & artist'])    
all_top_songs_list_df['scraped'] =  songs_scraped_index  
# Select unscraped songs to put them into a list of dictionary
all_top_songs_list_unscraped_df = all_top_songs_list_df[~songs_scraped_index]    
all_top_songs_list_unscraped = list(all_top_songs_list_unscraped_df.T.to_dict().values())
# Save into pickle files
# Saving list
with open('./data/all_top_songs_list_unscraped.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(all_top_songs_list_unscraped, f)      
    f.close()     

    
with open('./data/Youtube_all_data_df.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(Youtube_all_data_df, f)      
    f.close()     



######################################
##### Try plotting something interesting
with open('./data/Youtube_all_data_df.pickle', 'rb') as f:  # Python 3: open(..., 'wb')
    Youtube_all_data_df = pickle.load(f)      
    f.close()       
Youtube_all_data_df = Youtube_all_data_df.rename(index=str, columns={"view_count": "views"})
Youtube_all_data_df = Youtube_all_data_df[Youtube_all_data_df['views']>0]    
Youtube_all_data_df['year'].groupby("year").count()
    

# Scatter plot 
import matplotlib.pyplot as plt   
rank_range = [1,100]
songs_in_rank = Youtube_all_data_df[(Youtube_all_data_df['peakPos']>=rank_range[0])
 & (Youtube_all_data_df['peakPos']<=rank_range[1])]
years = songs_in_rank['year']   
popularity = songs_in_rank['views']
peakPos = songs_in_rank['peakPos']
fig = plt.figure(figsize=(20,20))
plt.scatter(years,(popularity),edgecolors='none',s=20,c=(peakPos), vmin=0, vmax=100)
plt.colorbar()
plt.title('Ranks Peak: '+ str(rank_range[0]) + ":" + str(rank_range[1]))
    

fig = plt.figure(figsize=(20,20))   
for i in range(5):
    rank_range = [20*i,20*(i+1)]
    songs_in_rank = Youtube_all_data_df[(Youtube_all_data_df['peakPos']>=rank_range[0])
     & (Youtube_all_data_df['peakPos']<=rank_range[1])]
    years = songs_in_rank['year']   
    popularity = songs_in_rank['views']
    peakPos = songs_in_rank['peakPos']    
    ax = fig.add_subplot(3,2,i+1)
    ax.set_ylim([2.5, 22.5])
    ax.scatter(years,np.log(popularity),edgecolors='none',s=20,c=(peakPos), vmin=0, vmax=100)
    ax.set_title('Ranks Peak: '+ str(rank_range[0]) + ":" + str(rank_range[1]))
plt.colorbar()
    

####################################        
##### Testing by visualization
rank_range = [1,100]
songs_in_rank = Youtube_all_data_df[(Youtube_all_data_df['peakPos']>=rank_range[0])
 & (Youtube_all_data_df['peakPos']<=rank_range[1])]
songs_in_rank = songs_in_rank[(songs_in_rank["year"] >= 1960) & (songs_in_rank["year"] < 1980)]
years = songs_in_rank['year']   
popularity = songs_in_rank['views']
peakPos = songs_in_rank['peakPos']    
fig = plt.figure(figsize=(10,10))
plt.scatter(peakPos,np.log(popularity),edgecolors='none',s=20, label="Songs")
#plt.scatter(years,popularity,edgecolors='none',s=20,c=(peakPos), vmin=0, vmax=100, label="Songs")
plt.scatter(years,popularity,edgecolors='none',s=20,c=(peakPos), vmin=rank_range[0], vmax=rank_range[1], label="Songs")
#plt.scatter(years,popularity,edgecolors='none',s=20,c=(peakPos), vmin=0, vmax=100, cmap=plt.get_cmap('rainbow'))
plt.colorbar().set_label("Peak rank", fontsize=20)
plt.title('Ranks Peak '+ str(rank_range[0]) + " - " + str(rank_range[1]), fontsize=25)
plt.xlabel("Year", fontsize=20)
plt.ylabel("Listeners", fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tick_params(labelsize=14)


from popularity_functions import popularity_function_guessing
from matplotlib.pyplot import cm
color=cm.rainbow(np.linspace(0,1,rank_range[1]+1-rank_range[0]))
for i in range(rank_range[0],rank_range[1]+1):
    time_variable_val = np.arange(1960, 2015, 0.1)
    rank_variable_val = [i]*len(time_variable_val)
    
    popularity_predicted = popularity_function_guessing(rank_variable_val, time_variable_val)
    plt.plot(time_variable_val, popularity_predicted, color = color[i - rank_range[0]], linestyle = '-', label="Rank "+str(i))    
    if((rank_range[1]-rank_range[0]) < 10):
        plt.legend()
    
    
####################################        
##### Linear Regression
X_train = np.concatenate( (np.reshape(peakPos.values,(-1,1)), np.reshape(years.values,(-1,1))), axis = 1 )
Y_train = np.reshape( np.log(popularity.values),(-1,1) )        
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, Y_train)
model.coef_
model.intercept_        
