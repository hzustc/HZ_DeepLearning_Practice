from news_recommender.Recalls.itemcf_recall import ItemCF
if __name__ == '__main__':
    module = ItemCF(sample=True)


    module.itemcf_sim(module.all_click_df, module.item_created_time_dict)
    module.evalute()