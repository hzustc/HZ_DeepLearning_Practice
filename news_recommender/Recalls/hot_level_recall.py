from news_recommender.settings import data_path, metric_recall, save_path
from news_recommender.tools import get_all_click_df, get_hist_and_last_click, metrics_recall, get_all_click_sample

import pickle
import collections
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed



class HotLevelRecall:

    def __init__(self,sample = True):
        self.sample = sample

    def hot_level_recall(self,all_click_df,user_id,topk,hours,):
        recall_time_range = all_click_df[all_click_df['user_id'] == user_id]['click_timestamp'].max() - 3600 * hours
        df = all_click_df[all_click_df['click_timestamp'] > recall_time_range]

        item_score_dict = collections.Counter(df['click_article_id'])
        item_rank = sorted(item_score_dict.items(), key=lambda x: x[1], reverse=True)[:topk]
        return item_rank





    def evalute(self):
        # 假设这些函数已经定义
        # get_all_click_sample(), get_hist_and_last_click(), hot_level_recall(), metrics_recall()

        all_click_df = get_all_click_sample(data_path) if self.sample else get_all_click_df(data_path)

        if metric_recall:
            trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
        else:
            trn_hist_click_df = all_click_df

        user_recall_items_dict = collections.defaultdict(dict)

        # 设置并行参数
        hours = 1
        recall_item_num = 50

        # 定义一个获取召回数据的函数
        def recall_for_user(user):
            return user, self.hot_level_recall(trn_hist_click_df, user, recall_item_num, hours)

        # 使用 ThreadPoolExecutor 来并行处理
        with ThreadPoolExecutor() as executor:
            # 提交任务
            futures = {executor.submit(recall_for_user, user): user for user in trn_hist_click_df['user_id'].unique()}

            # 获取并处理结果
            for future in tqdm(as_completed(futures), total=len(futures)):
                user, recalled_items = future.result()
                user_recall_items_dict[user] = recalled_items

        # 保存召回结果
        pickle.dump(user_recall_items_dict, open(save_path + 'hot_level_recall.pkl', 'wb'))

        # 召回效果评估
        if metric_recall:
            metrics_recall(user_recall_items_dict, trn_last_click_df, topk=recall_item_num)


#
# from news_recommender.settings import data_path, metric_recall, save_path
# from news_recommender.tools import get_all_click_df, get_hist_and_last_click, metrics_recall, get_all_click_sample
#
# import pickle
# import collections
# from tqdm import tqdm
# from concurrent.futures import ProcessPoolExecutor, as_completed
#
# # 将 recall_for_user 函数移到类外部
# def recall_for_user(user, hot_level_recall_obj, all_click_df, topk, hours):
#     return user, hot_level_recall_obj.hot_level_recall(all_click_df, user, topk, hours)
#
# class HotLevelRecall:
#
#     def __init__(self, sample=True):
#         self.sample = sample
#
#     def hot_level_recall(self, all_click_df, user_id, topk, hours):
#         recall_time_range = all_click_df[all_click_df['user_id'] == user_id]['click_timestamp'].max() - 3600 * hours
#         df = all_click_df[all_click_df['click_timestamp'] > recall_time_range]
#
#         item_score_dict = collections.Counter(df['click_article_id'])
#         item_rank = sorted(item_score_dict.items(), key=lambda x: x[1], reverse=True)[:topk]
#         return item_rank
#
#     def evalute(self):
#         all_click_df = get_all_click_sample(data_path) if self.sample else get_all_click_df(data_path)
#
#         if metric_recall:
#             trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
#         else:
#             trn_hist_click_df = all_click_df
#
#         user_recall_items_dict = collections.defaultdict(dict)
#
#         # 设置并行参数
#         hours = 1
#         recall_item_num = 50
#
#         # 使用 ProcessPoolExecutor 来并行处理
#         with ProcessPoolExecutor() as executor:
#             # 提交任务，使用外部的 recall_for_user 函数
#             futures = {executor.submit(recall_for_user, user, self, trn_hist_click_df, recall_item_num, hours): user for user in trn_hist_click_df['user_id'].unique()}
#
#             # 获取并处理结果
#             for future in tqdm(as_completed(futures), total=len(futures)):
#                 try:
#                     user, recalled_items = future.result()  # 获取结果
#                     user_recall_items_dict[user] = recalled_items
#                 except Exception as e:
#                     print(f"Error processing user {futures[future]}: {e}")
#
#         # 保存召回结果
#         pickle.dump(user_recall_items_dict, open(save_path + 'hot_level_recall.pkl', 'wb'))
#
#         # 召回效果评估
#         if metric_recall:
#             metrics_recall(user_recall_items_dict, trn_last_click_df, topk=recall_item_num)
#
