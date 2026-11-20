---
pretty_name: "Last.fm user artist song dataset"
tags:
- music
size_categories:
- 10M<n<100M
configs:
- config_name: default
  data_files:
  - split: train
    path: "train.gz.parquet"
  - split: valid
    path: "valid.gz.parquet"
  - split: test
    path: "test.gz.parquet"
---

## Dataset Description

This dataset is ideal for training a recommendation system that incorporates time and country information.

### Task Summary

A recommender system, or a recommendation system, is a subclass of information filtering system that provides suggestions for items that are most pertinent to a particular user. Recommender systems are particularly useful when an individual needs to choose an item from a potentially overwhelming number of items that a service may offer.

Typically, the suggestions refer to various decision-making processes, such as what product to purchase, what music to listen to, or what online news to read. Recommender systems are used in a variety of areas, with commonly recognised examples taking the form of playlist generators for video and music services, product recommenders for online stores, or content recommenders for social media platforms and open web content recommenders. These systems can operate using a single type of input, like music, or multiple inputs within and across platforms like news, books and search queries. There are also popular recommender systems for specific topics like restaurants and online dating. Recommender systems have also been developed to explore research articles and experts, collaborators, and financial services.

[Wikipedia: Recommender System](https://en.wikipedia.org/wiki/Recommender_system)

Training a model to generate high quality recommendations is hard because most users never review the products or services that they use. Implicit recommendations are available for training and are typically based on interactions with the products or services. While this data is abundant it lacks negative examples, as the interaction that is used for training has occurred even if the user did not like the product or service.

### Dataset Summary

This dataset contains the user, timestamp, artist, song records. The user may have provided additional details such as their gender, age and location. These additional details are both optional and unverified.

Last.fm provides a service known as scrobbling where you can submit tracks you have listened to in other apps. When you submit these you do not have to provide musicbrainz ids and as such these can be ambiguous. To make this dataset easier to use I have removed rows where the musicbrainz identifiers are missing.

The train, validation and test splits have separate users with no overlap.

To make training on this dataset easier I have added the user_index, artist_index and track_index columns. These are integer columns that uniquely identify every user, artist and track.

### Dataset Source

The original data was collected by [Òscar Celma @ MTG/UPF](http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html).

### Dataset License

The data contained is distributed with permission of Last.fm. The data is made available for non-commercial use. Those interested in using the data or web services in a commercial context should contact: partners@last.fm

For more information see Last.fm [terms of service](http://www.last.fm/api/tos).

### Citation Information

When using this dataset you must reference the [Last.fm](http://last.fm/) webpage.

```
@book{Celma:Springer2010,
    author = {Celma, O.},
    title = {{Music Recommendation and Discovery in the Long Tail}},
    publisher = {Springer},
    year = {2010}
}
```
