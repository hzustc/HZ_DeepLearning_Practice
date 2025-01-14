import math
import os
import time

import numpy as np
import pandas as pd
import pickle
import collections
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from news_recommender.settings import save_path
from news_recommender.tools import metrics_recall, get_user_item_time


#todo:召回5分钟

class HotLevelRecall:

    def __init__(self, all_click_df, recall_nums, hours):
        #计算运行时间
        start_time = time.time()
        self.all_click_df = all_click_df.sort_values(by='click_timestamp').reset_index(drop=True)
        self.user_item_time_dict = get_user_item_time(self.all_click_df)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"sort运行时间: {execution_time} 秒")
        self.recall_nums = recall_nums
        self.hours = hours

    def is_item_in_user_list(self, user_id, item_id):
        # 从字典中获取指定用户的物品ID列表
        item_time_list = self.user_item_time_dict.get(user_id, [])

        # 提取物品ID列表
        item_ids = [item for item, _ in item_time_list]

        # 判断物品ID是否在用户的物品ID列表中
        return item_id in item_ids

    def hot_level_recall_for_user(self, user_id):
        """
        对单个用户执行热门物品召回，仅基于点击次数计算分数。
        """
        all_click_df = self.all_click_df
        hours = self.hours

        # 获取该用户最后一次点击的时间戳
        last_click_time = all_click_df[all_click_df['user_id'] == user_id]['click_timestamp'].max()

        time_range = 3600*hours*1000

        # 假设你要查找时间范围
        start_time = last_click_time-time_range
        end_time = last_click_time+time_range

        # 使用 searchsorted 查找开始和结束位置
        start_idx = all_click_df['click_timestamp'].searchsorted(start_time, side='left')
        end_idx = all_click_df['click_timestamp'].searchsorted(end_time, side='right')

        # 通过切片快速过滤符合时间范围的数据
        df = all_click_df.iloc[start_idx:end_idx]


        # #过滤在用户最后点击时间前后的指定小时范围内的记录
        # df = all_click_df[
        #     (all_click_df['click_timestamp'] > last_click_time - time_range) &
        #     (all_click_df['click_timestamp'] < last_click_time + time_range)
        # ]


        # 仅使用点击次数计算分数
        item_score_dict = collections.Counter(df['click_article_id'])

        # 对分数取对数
        item_score_dict = {item_id: math.log(score + 1) for item_id, score in item_score_dict.items() if not self.is_item_in_user_list(user_id,item_id) }

        item_rank = sorted(item_score_dict.items(), key=lambda x: x[1], reverse=True)[:self.recall_nums]


        return user_id, item_rank

    def hot_level_recall_parallel(self, recall_user_list):
        """
        对多个用户并行执行热门物品召回，仅使用点击次数。
        """
        user_recall_dict = {}
        start_time = time.time()
        print(start_time)
        # 使用线程池并行处理每个用户的召回
        with ThreadPoolExecutor() as executor:
            future_to_user = {executor.submit(self.hot_level_recall_for_user, user_id): user_id for user_id in recall_user_list}
            for future in tqdm(as_completed(future_to_user), total=len(future_to_user)):
                user_id, item_rank = future.result()
                user_recall_dict[user_id] = item_rank
        end_time = time.time()
        print(end_time)
        execution_time = end_time - start_time
        print(f"运行时间: {execution_time} 秒")
        # 保存结果
        with open(r'D:\AI\HZ_DeepLearning_Practice\news_recommender\tmp_results\hot_recall_dict.pkl', 'wb') as f:
            pickle.dump(user_recall_dict, f)
        return user_recall_dict
    def save(self, to_path=None,recall_dict=None):
        with open(to_path, 'wb') as f:
            pickle.dump(recall_dict, f)

if __name__ == '__main__':
    all_click_df = pd.read_pickle(r'D:\AI\HZ_DeepLearning_Practice\news_recommender\tmp_results\train_click_df.pkl')
    test_click_df =pd.read_pickle(r'D:\AI\HZ_DeepLearning_Practice\news_recommender\tmp_results\valid_click_last_df.pkl')

    recall_user_list = test_click_df['user_id'].unique()
    recall_user_set = set(recall_user_list)

    obj = HotLevelRecall(all_click_df, recall_nums=50, hours=1,)
    hot_recall_dict = obj.hot_level_recall_parallel(recall_user_list)
    metrics_recall(hot_recall_dict,test_click_df,topk=30)
    obj.save(to_path=os.path.join(save_path, 'hot_recall_self_test_dict.pkl'), recall_dict=hot_recall_dict)


    # train_click_df = pd.read_csv(r'D:\AI\HZ_DeepLearning_Practice\news_recommender\data_raw\train_click_log.csv')
    # testA_click_df = pd.read_csv(r'D:\AI\HZ_DeepLearning_Practice\news_recommender\data_raw\testA_click_log.csv')
    # all_click_df = pd.concat([train_click_df, testA_click_df], axis=0)
    # recall_user_list = testA_click_df['user_id'].unique()
    # recall_user_set = set(recall_user_list)
    #
    # obj = HotLevelRecall(all_click_df, topk=50, hours=10)
    # hot_recall_dict = obj.hot_level_recall_parallel(recall_user_list)
    # obj.save(to_path=os.path.join(save_path, 'hot_recall_dict.pkl'), recall_dict=hot_recall_dict)
    # pickle.dump(hot_recall_dict, open(r'D:\AI\HZ_DeepLearning_Practice\news_recommender\tmp_results\hot_recall_dict.pkl', 'wb'))


