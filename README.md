# Do Popular Songs Endure?

## Introduction
Surprisingly, most of the popular enduring songs nowadays (as measured by current sales or airplay) were not actually ranked to the top of the charts when they first came out back in 1960s. The main objective of this project is to find the endurance of popular songs, and to formulate if songs which were popular at the time of their release actually remain their popularity over time. By investigation of a variety of datasets, the popularity of songs can be defined by various methods such as ranking functions which relate directly to number of views, Billboard rankings, etc, and other characteristics as well like number of sales, etc. Some other interesting factors that might contribute to the songs popularity can be explored more in depth with Million Songs Dataset for example, which include a rich features set about any particular songs such as its danceability, song_hotttness, analysis sample rate, types of genre, etc. Based on these features, we will build a model to predict the endurance of songs’ popularity over time from the time they were released.

## Dataset
### Songs’ Metadata – One Million Dataset
One Million Songs dataset contains many useful information about a song that if used appropriately, can predict the song’s durability through time. Some of the most important features include: tempo, loudness, durations, artists hotness. These features will be analyzed to know whether they have influence on song’s performance or not, and if it does, to which extent. 

### Past Songs’ Popularity – Billboard Rankings
The Billboard ranking datasets are chosen as records’ popularity at the time they were released. This is a very trustful source of data and reflect very well how popular the records were when they were released. For each record, the Billboard Top100 chart returns its ranking and also the time (in weeks) the record was released. 

### Current Songs’ Popularity - Youtube Views Count
We choose Youtube views count as our main popularity measurement. The main reasons are: Youtube is a very very popular channel for music listeners, and data collected from Youtube can be considered primary data, without any modification or formulaized by the webpage (unlike Spotify's popularity score).

## Building Model For Popularity Prediction
Our initial guess is that a song's current popularity is:
- Exponentially propotional to BillBoard ranking when it was released. The higher the rank was, the more popular now. 
- Exponentially inversely proportional with the time it was released. The older the song is, the less popular now.
We then take log of the equation to make it a linear equation, and then use linear regression to compute the fittest variables. 
The model fitted is illustrated in the image below:
![Optional Text](../master/illustrations/github_pic1.JPG)


## Detecting Over/Underperform Songs
We detect overperform/underperform songs by choosing out songs that have ground-truth popularity too higher or too lower than the predicted popularity. The image below illustrates how we point out overperform/underperform songs.
![Optional Text](../master/illustrations/github_pic2.JPG | width=48) 

## Analyzing Over/Underperform Songs
By studying songs that are classified as over/underperform songs as defined by our method in section 4, we find many interesting characteristics of these songs. Analyzing about 100 songs,  50 of them are overperform songs, the other are underperform songs, we find that songs’s durability can be explained by two main sets of factors:
- Special occasions, temporary trends. Example: famous cover of songs, death of artist, movies about artists released… These factors are unique for each song and therefore, are very hard to be used to predict durability of other songs.
- Song’s metadata. Example: tempo, genre, loudness or artist hotness,… These are features that can be shared between different songs and therefore, can be used to predict another song’s durability.
We will analyze each of these two sets of factors.

### Songs’ Durability Explained by Special Occasions, Temporary Trends
Studying over/underperform songs, we find there are some main special occasional/temporary trending reasons affecting songs’ durability:
-	Artist’s passing
- Movies about artist’s career
- Holidays occasions
- Famous cover of a record
### Songs’s Durability Explained by Metadata of Songs
Analyzing the One-Million Song Dataset, we test the effect on durability of each important feature of a song, including: Tempo, Duration,	Song hotness,	Loudness,	Artist hotness,	Genre.
We find out that the most important factor is artist hotness. This is the main factor deciding whether a song will be over/underperform. We will analyze as well as show examples for this factor’s influence on songs’ durability below.
On the other hand, other factors (tempo, duration, genre,…) are found to not have significant impact on a song’s performance through time. Indeed, as Figure 13 (in the report file) shows, there is not much differences in these features of an over/underperform song with a usual song.
 

## Conclusion
Our findings point out that the most important factors for a song’s performance is the “brand” of the artist. This observation can be illustrated by many examples. Other factors also very important are special occasion and temporary trending such as death of an artist, releasing of a movie relating to the artist. All of the factors combining together helps a song over/underperform comparing to other songs released at the same time, at the same specific Billboard ranking.  



## References
[1]    Million Song Dataset: https://labrosa.ee.columbia.edu/millionsong/
[2]    Billboard ranking: https://www.billboard.com/charts
[3]    Grammy Awards: https://www.kaggle.com/theriley106/grammyawardsinnumbers
[4]    S. Homan, “Popular music and cultural memory: Localised popular music histories and their significance for national music industries: data,” 2012.
[5]    Y. Kim, B. Suh, and K. Lee, “# nowplaying the future billboard: mining music listening behaviors of twitter users for hit song prediction,” in Proceedings of the first international workshop on Social media retrieval and analysis. ACM, 2014. pp. 51-56.
[6]    J. Berger, and G. Packard, “Are Atypical Things More Popular?”

































