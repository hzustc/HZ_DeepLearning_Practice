from news_recommender.Recalls.hot_level_recall import HotLevelRecall

if __name__ == '__main__':
    model = HotLevelRecall(sample=True)
    model.evalute()
