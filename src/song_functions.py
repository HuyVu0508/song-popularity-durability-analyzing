import matplotlib.pyplot as plt
import matplotlib
import datetime
import pickle
import re
import billboard
import requests
import bs4
from bs4 import BeautifulSoup as bs


#########################
# Function for searching a song from Billboard chart
def lookup_BillBoard(song_name, artist_name):
    
    # Getting back the objects:
    with open('./data/chart_history.pickle','rb') as f:  # Python 3: open(..., 'rb')
        chart_history = pickle.load(f)    
    
    # Searching from the beginning of the year    
    ranks_over_time = []
    weeks_over_time = []
    keys = list(chart_history.keys())
    for i in range(len(keys)): 
        # Searching for relevant year
        week = keys[i]
        date = datetime.datetime.strptime(week, '%Y-%m-%d')
        
        # Processing when in right year
        chart_week = chart_history[keys[i]]
        for j in range(len(chart_week[:])):
            song_j = chart_week[j]
            if (song_j.title == song_name) & (song_j.artist == artist_name):
                ranks_over_time.append(song_j.rank)
                weeks_over_time.append(date)        
    
#    plt.figure(figsize=(20,10))
#    plt.plot(weeks_over_time, ranks_over_time)
    
    BillBoard_dict = {"ranks_over_time": list(ranks_over_time), "weeks_over_time": list(weeks_over_time), "peak": min(ranks_over_time), "weeks": len(weeks_over_time)}
    return BillBoard_dict

#########################
# Scraping Billboard data
def BillBoard_scrape():
    # Scrape all data and save to variables
    start_date = '1976-02-27' 
    date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    chart_history = {}
    while (date < date.today()):
        
        # Extracting date
        date_string = date.strftime('%Y-%m-%d')
        print(date_string)
        
        # Add chart to chart_history
        chart = billboard.ChartData('hot-100', date_string) 
        chart_history.update({date_string: chart})
        
        # Updating date
        date = date + datetime.timedelta(days=7)
        
    # Save chart_history to file
    with open('./data/chart_history.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(chart_history, f)      
        f.close()               



#########################
# Fucntions retrieving data from one video    
def get_Youtube_video_data(video_page_url):
    video_data = {}
    response = requests.get(video_page_url)
    soup = bs4.BeautifulSoup(response.text, 'html.parser')
    
    # title
    video_data['title'] = soup.select('title')[0].get_text()
    
    # If not video (user account)
    if(len(soup.findAll('div', attrs={'class': 'watch-view-count'}))==0):
        video_data['views'] = 0
        return video_data
        
    # Else if true video (not user account)
    # view
    view = ''
    viewsText = soup.findAll('div', attrs={'class': 'watch-view-count'})[0].get_text().split(' ')[0]
    viewSplit = viewsText.split(',')
    for i in range(0, len(viewSplit)):
        view += viewSplit[i]
        
    video_data['views'] = int(view)
    
#    # like
#    like = ''
#    likeText = soup.findAll('button', attrs={'class': 'like-button-renderer-like-button'})[0].get_text()
#    likeSplit = likeText.split(',')
#    for i in range(0, len(likeSplit)):
#        like += likeSplit[i]
#    
#    video_data['like'] = int(like)
#    
#    # dislike
#    dislike = ''
#    dislikeText = soup.findAll('button', attrs={'class': 'like-button-renderer-dislike-button'})[0].get_text()
#    dislikeSplit = dislikeText.split(',')
#    for i in range(0, len(dislikeSplit)):
#        dislike += dislikeSplit[i]
#    
#    video_data['dislike'] = int(dislike)
    
#    print(video_data['title'])
#    print(video_data['views'])  
    return video_data    



#########################
# Scrabing Youtube function
def lookup_Youtube(song_name, artist_name):
    
    # Construct query link
    query_string = song_name + " " + artist_name
    query_link = query_string.replace(" ", "+")
    
    # Retrieving data
    youtube_base_link = "https://www.youtube.com/results?search_query="
    r = requests.get(youtube_base_link + query_link)   
    page = r.text
    soup=bs(page,'html.parser')   
    vids = soup.findAll('a',attrs={'class':'yt-uix-tile-link'})    
    
    # Loop through videos retrieved
    correct_vids = []
    num_of_correct_vids = 0
    maximum_video = 5
    for v in vids:
        
        # Check if video relevant to search query
        video_tile = v['title']
        if (song_name in video_tile) & (artist_name in video_tile) & (num_of_correct_vids<maximum_video):

            # Get video data
            tmp = 'https://www.youtube.com' + v['href']
            video_data = get_Youtube_video_data(tmp)            
            correct_vids.append(video_data)    
            num_of_correct_vids = num_of_correct_vids + 1     
            
            # Loop out if getting enough 10 videos
            if(num_of_correct_vids>=maximum_video):
                break
                
    # Song stats
    view_count = 0
    for i in range(num_of_correct_vids):
        view_count = view_count + correct_vids[i]['views']
    
    # Return num of views of song
    Youtube_dict = {"view_count": view_count}
    return Youtube_dict



#########################
# Searching Billboard data and Youtube data
def lookup_BillBoard_Youtube(song_name, artist_name):
    BillBoard_dict = lookup_BillBoard(song_name, artist_name)
    Youtube_dict = lookup_Youtube(song_name, artist_name)
    BillBoard_Youtube_dict = {'BillBoard_dict': BillBoard_dict, "Youtube_dict": Youtube_dict}
    
    peak = BillBoard_dict["peak"]
    weeks = BillBoard_dict["weeks"]
    view_count = Youtube_dict["view_count"]
    BillBoard_Youtube_dict = {"peak": peak, "weeks": weeks, "view_count": view_count}
    
    return BillBoard_Youtube_dict

#########################
# Combine list of top 100 BillBoard all time    
def BillBoard_chart_all_time(filePath):
    file = open(filePath,'rb')
    chart_history = pickle.load(file)
    unique = []
    data = []

    for i in range(len(chart_history)-1,-1,-1):    
        k = list(chart_history.keys())[i]
        v = list(chart_history.values())[i]
        year = k.split('-')[0]
        month = k.split('-')[1]
        day = k.split('-')[2]
    
        for j in range(len(v)-1,-1,-1):    
            song = v.entries[j]
            songData = {}
            songData['year'] = round(int(year)+ int(month)/12 + (int(day) - song.weeks*7)/366, 2)
            songData['title'] = song.title.strip()
            songData['artist'] = song.artist.strip()
            songData['rank'] = song.rank
            songData['weeks'] = song.weeks
            songData['peakPos'] = song.peakPos
            name = song.title.strip() + song.artist.strip()
            name = name.strip()
            
            if name not in unique:
                data.append(songData)
                unique.append(name)
         
    return data






