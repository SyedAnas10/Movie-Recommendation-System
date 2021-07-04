from recommendor import Recommendation

if __name__ == '__main__':
    searched = input("Enter a movie title to search \n "
                     "Note: Titles not present in data set will raise an error : \n")
    re = Recommendation(searched)
    print("Titles similar with respect to Cast-Genre")
    re.getCastGenreSimilar()
    print("\nTitles similar with respect to Cast-Year")
    re.getCastYearSimilar()
    print("\nTitles similar with respect to Genre-Year")
    re.getGenreYearSimilar()
    print("\nTitles with similar names")
    re.getSimilarTitles()
    print("\nTitle with similar plots or description provided")
    re.getPlotSimilar()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
