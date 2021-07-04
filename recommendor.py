import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy import spatial
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

df = pd.read_csv('./netflix_titles.csv', index_col=0)
le = LabelEncoder()
neighbors = NearestNeighbors()
nmf = NMF(n_components=100)
vectorised = TfidfVectorizer()


class Recommendation:

    def __init__(self, searched):
        self.searchedMovie = searched.lower()
        self.sameCastTitles = pd.DataFrame()
        self.searchedIndex = None
        self.cast_genre_similar = None
        self.cast_year_similar = None
        self.genre_year_similar = None
        self.plots_similar = None
        self.similar_titles = None
        self.preparation()

    def preparation(self):

        entry = df.loc[df['title'].str.lower() == self.searchedMovie]
        self.searchedIndex = entry.index[0]

        # EXTRACTING MOVIES WITH SAME ACTORS
        castMembers = str(entry.iloc[0, 3]).split(', ')
        if len(castMembers) > 0:
            topActors = castMembers[:1]
        else:
            topActors = [actor for actor in castMembers]

        for actor in topActors:
            titles = self.sameCastTitles.append(df[[actor in str(cast) for cast in df['cast']]]
                                                [['title', 'cast', 'release_year', 'listed_in']])
            self.sameCastTitles = self.sameCastTitles.append(titles)
        self.sameCastTitles.drop_duplicates(inplace=True)

        # IDENTIFYING UNIQUE ACTORS AND GENRES FROM SAME CAST TITLES
        allActors = []
        allGenres = []
        for i in self.sameCastTitles.index:
            actors = str(self.sameCastTitles.loc[i, 'cast']).split(", ")
            genres = str(self.sameCastTitles.loc[i, 'listed_in']).split(", ")
            allActors = list(set(allActors) | set(actors))
            allGenres = list(set(allGenres) | set(genres))
            self.sameCastTitles.at[i, 'cast'] = actors
            self.sameCastTitles.at[i, 'listed_in'] = genres
        maxActors = max(map(len, self.sameCastTitles['cast']))
        maxGenres = max(map(len, self.sameCastTitles['listed_in']))

        # LABELLING EVERY ACTOR WITH UNIQUE LABELS
        le.fit_transform(allActors)

        # REPLACING ACTORS WITH THEIR LABELS IN EXTRACTED DATAFRAME
        labelsColumn = []
        for i in self.sameCastTitles.index:
            labelsList = list(le.transform(self.sameCastTitles.loc[i, 'cast']))
            labelsList = labelsList + [-1] * (maxActors - len(labelsList))
            labelsColumn.append(labelsList)
        self.sameCastTitles['labeledCast'] = labelsColumn

        # LABELLING EVERY GENRE WITH UNIQUE LABELS
        le.fit_transform(allGenres)

        # REPLACING GENRES WITH THEIR LABELS IN EXTRACTED DATAFRAME
        labelsColumn = []
        for i in self.sameCastTitles.index:
            labelsList = list(le.transform(self.sameCastTitles.loc[i, 'listed_in']))
            labelsList = labelsList + [-1] * (maxGenres - len(labelsList))
            labelsColumn.append(labelsList)
        self.sameCastTitles['labeledGenres'] = labelsColumn

        # COMPUTING COSINE SIMILARITIES BETWEEN THE CAST OF SEARCHED TITLE WITH SIMILAR CAST TITLES
        similarityColumn = []
        searchedCast = self.sameCastTitles.loc[self.searchedIndex, 'labeledCast']
        for i in self.sameCastTitles.index:
            similarCast = self.sameCastTitles.loc[i, 'labeledCast']
            cosineSimilarity = 1 - spatial.distance.cosine(searchedCast, similarCast)
            similarityColumn.append(round(cosineSimilarity, 2))
        self.sameCastTitles['castSimilarity'] = similarityColumn

        # COMPUTING COSINE SIMILARITIES BETWEEN THE GENRE OF SEARCHED TITLE WITH SIMILAR CAST TITLES
        similarityColumn = []
        searchedGenre = self.sameCastTitles.loc[self.searchedIndex, 'labeledGenres']
        for i in self.sameCastTitles.index:
            similarGenre = self.sameCastTitles.loc[i, 'labeledGenres']
            cosineSimilarity = 1 - spatial.distance.cosine(searchedGenre, similarGenre)
            similarityColumn.append(round(cosineSimilarity, 2))
        self.sameCastTitles['genreSimilarity'] = similarityColumn

        # COMPARING CAST-GENRE, CAST-YEAR and YEAR-GENRE
        cg = []
        cy = []
        gy = []
        for i in self.sameCastTitles.index:
            cg.append([self.sameCastTitles.loc[i, 'castSimilarity'], self.sameCastTitles.loc[i, 'genreSimilarity']])
            cy.append([self.sameCastTitles.loc[i, 'release_year'], self.sameCastTitles.loc[i, 'castSimilarity']])
            gy.append([self.sameCastTitles.loc[i, 'release_year'], self.sameCastTitles.loc[i, 'genreSimilarity']])
        self.sameCastTitles['CGsimilar'] = cg
        self.sameCastTitles['CYsimilar'] = cy
        self.sameCastTitles['GYsimilar'] = gy


    def getCastGenreSimilar(self):
        n = 8 if len(self.sameCastTitles.index) > 8 else len(self.sameCastTitles.index)
        neighbors.fit(list(self.sameCastTitles['CGsimilar']))
        nearestIndex = neighbors.kneighbors([self.sameCastTitles.loc[self.searchedIndex, 'CGsimilar']],
                                            n_neighbors=n, return_distance=False)[0]
        self.cast_genre_similar = list(self.sameCastTitles.iloc[nearestIndex, 0])
        print(self.cast_genre_similar)

    def getCastYearSimilar(self):
        n = 8 if len(self.sameCastTitles.index) > 8 else len(self.sameCastTitles.index)
        neighbors.fit(list(self.sameCastTitles['CYsimilar']))
        nearestIndex = neighbors.kneighbors([self.sameCastTitles.loc[self.searchedIndex, 'CYsimilar']],
                                            n_neighbors=n, return_distance=False)[0]
        self.cast_year_similar = list(self.sameCastTitles.iloc[nearestIndex, 0])
        print(self.cast_year_similar)

    def getGenreYearSimilar(self):
        n = 8 if len(self.sameCastTitles.index) > 8 else len(self.sameCastTitles.index)
        neighbors.fit(list(self.sameCastTitles['GYsimilar']))
        nearestIndex = neighbors.kneighbors([self.sameCastTitles.loc[self.searchedIndex, 'GYsimilar']],
                                            n_neighbors=n, return_distance=False)[0]
        self.genre_year_similar = list(self.sameCastTitles.iloc[nearestIndex, 0])
        print(self.genre_year_similar)

    def getPlotSimilar(self):
        index = list(df.index)
        plots = list(df['description'])
        tfidf_matrix = vectorised.fit_transform(plots)
        nmf_features = nmf.fit_transform(tfidf_matrix)
        norm_features = normalize(nmf_features)
        samePlotTitles = pd.DataFrame(norm_features, index=index)
        movie = samePlotTitles.loc[self.searchedIndex]
        similarities = samePlotTitles.dot(movie)
        similar = list(similarities.nlargest(n=20).index)
        self.plots_similar = list(df.loc[similar, 'title'])
        print(self.plots_similar)

    def getSimilarTitles(self):
        self.similar_titles = df[[self.searchedMovie.split(":")[0] in title.lower() for title in df['title']]]['title']
        print(self.similar_titles)
