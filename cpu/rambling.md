### Goal
Implement a k-means clustering algorithm on the dataset, clustering by genre.


### Process
1) k initial "means" are randomly generated within the data domain.
2) k clusters are created by associating every observation with the nearest mean.
3) The centroid of each of the k clusters becomes the new mean.
4) Repeat steps 2 and 3 until convergence.

#### Detailed Process
So, how the fuck do I actually implement k-means clustering here? K-means relies on distance. 
So, gotta create some way to measure distance. At a very basic level I could start with a one-
dimensional measure like danceability, then the implementation of kmeans is trivial, if a bit 
useless due to fitting 1m songs in a 0-1 range. The solution is to add more dimensions, and 
thankfully I got plenty of dimensions to work with. Expanding to higher dimensions using the 
various float and integer values is very easy, and can get me pretty far (probably).

Issue: handling string values. It's impossible, without implementing one-hot-encoding (or 
similar). And since these are relatively unique string values (names and such), that's not going 
to be very useful (one-hot-encoding is more for categories). That cuts out six of the columns. 
Year can just be an integer dimension, and bools work as 0/1 just fine.

That means I'll have 17 workable dimensions (or features or whatever). Track and disc number 
feel pretty useless in determining genre, bringing it down to 15 features that I feel are 
reasonable factors in guessing a song's genre. Should be enough for a decent kmeans clustering.

This also means I can cull a decent amount of data from the dataset. I'll keep the id for ref.



### Dataset
#### Dataset Features
- id:               unique value per track                  string
- name:             name of the track                       string
- album:            album title                             string
- album_id:         unique album ID                         string
- artists:          list of artist names                    list of strings
- artist_ids:       list of spotify artist IDs              list of strings
- track_number:     track number                            int
- disc_number:      disc number                             int
- explicit:         whether the song is explicit or not     bool
- danceability:     how suitable a track is for dancing     float (0-1)
- energy:           how intense and active a track is       float (0-1)
- key:              overall key of the track                int (0-11)
- loudness:         overall loudness of the track           float (-60-7.23) (dB)
- mode:             track mode, major (1) or minor (0)      bool
- speechiness:      proportion of spoken words in track     float (0-1)
- acousticness:     confidence that the track is acoustic   float (0-1)
- instrumentalness: proportion of instrumental parts        float (0-1)
- liveness:         confidence that the track is live       float (0-1)
- valence:          how positive a track sounds             float (0-1)
- tempo:            overall tempo of the track              float (0-249) (BPM)
- duration_ms:      duration of a track in ms               float (0-6.06m) (ms)
- time_signature:   overall time signature of a track       int (0-5)
- year:             release year of a track                 date (YYYY)
- release_date:     release date of a track                 date (DD/MM/YYYY)


The goal is to be able to group these songs into genres through these statistics.
Relevant statistics are:
- danceability (genres generally share danceability)
- energy
- loudness
- mode
- speechiness
- acousticness
- instrumentalness
- artist_ids, artists (artists generally stick to one genre)
- album_id, album (albums are generally one genre)
- liveness
- valence
- tempo
- duration_ms
- time_signature
- year
- release_date
- explicit (certain genres are more likely to be explicit)
- name (perhaps similar names are similar genres?)
- album (perhaps similar titles are similar genres?)

Likely irrelevant:
- id
- disc_number
- track_number

Overlaps:
- album, album_id: album could provide info on similarly-named albums, while album_id just groups
songs in an album together.
- artists, artist_ids: similar logic as above.
- year, release_date: same info, one's more precise. could likely just use year.


### to-do
- ensure input csvs for gpu and cpu implementations are the same (just use the established csv_parser file)
 - it seems like the established cpu implementation can handle the new file just fine :)
- check that the results from the cpu and gpu versions are the same.
 - okay it looks like the GPU version just skips explicitness entirely as a feature ?

#### differences:
- gpu implementation ignores the explicit feature
- gpu normalizes with min/max of each feature prior to doing the algorithm
- 