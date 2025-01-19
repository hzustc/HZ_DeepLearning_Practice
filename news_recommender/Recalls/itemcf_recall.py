import collections
import math
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from news_recommender.settings import save_path, data_path
from news_recommender.tools import get_user_item_time, get_item_topk_click, \
    user_multi_recall_dict, metrics_recall, get_item_info_df, get_item_info_dict, plot_recall_score_distribution


class ItemCF:
    def __init__(self, all_click_df):

        self.all_click_df = all_click_df


        mm = MinMaxScaler()
        self.all_click_df['click_timestamp'] = mm.fit_transform(self.all_click_df[['click_timestamp']])
        self.item_info_df = get_item_info_df(data_path)
        self.item_info_df['created_at_ts'] = mm.fit_transform(self.item_info_df[['created_at_ts']])
        # 获取文章的属性信息，保存成字典的形式方便查询
        self.item_type_dict, self.item_words_dict, self.item_created_time_dict = get_item_info_dict(self.item_info_df)
        print('itemcf对象初始化')

    def min_max_normalize(self, similarity_dict):
        print('相似度矩阵归一化')
        # 提取所有相似度值
        all_values = [similarity_dict[key1][key2] for key1 in similarity_dict for key2 in similarity_dict[key1]]

        # 计算最大值和最小值
        max_value = max(all_values)
        min_value = min(all_values)

        # 归一化相似度字典
        normalized_dict = {}
        for key1 in similarity_dict:
            normalized_dict[key1] = {}
            for key2 in similarity_dict[key1]:
                value = similarity_dict[key1][key2]
                # 最大最小归一化公式
                normalized_value = (value - min_value) / (max_value - min_value)
                normalized_dict[key1][key2] = normalized_value

        return normalized_dict

    def itemcf_sim(self, df, item_created_time_dict, to_path):
        # todo: 相似矩阵归一化
        print('计算相似度矩阵')
        user_item_time_dict = get_user_item_time(df)
        i2i_sim = {}
        item_cnt = collections.defaultdict(int)

        for user, item_time_list in tqdm(user_item_time_dict.items(), desc="Processing user-item interactions"):
            for loc1, (i, i_click_time) in enumerate(item_time_list):
                #热门物品打压系数
                item_cnt[i] += 1
                i2i_sim.setdefault(i, {})
                for loc2, (j, j_click_time) in enumerate(item_time_list):
                    if (i == j):
                        continue

                    #顺序逆序关系权重
                    loc_alpha = 1.0 if loc2 > loc1 else 0.7

                    #位置相关性权重
                    loc_weight = loc_alpha * (0.9 ** (np.abs(loc2 - loc1) - 1))

                    #点击时间权重
                    click_time_weight = np.exp(0.7 ** np.abs(i_click_time - j_click_time))

                    #物品创建时间权重
                    created_time_weight = np.exp(0.8 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))

                    i2i_sim[i].setdefault(j, 0)

                    #对相似度进行用户活跃度打压
                    i2i_sim[i][j] +=   loc_weight * click_time_weight * created_time_weight / math.log(
                        len(item_time_list) + 1)

        #热门物品打压
        i2i_sim_ = i2i_sim.copy()
        for i, related_items in i2i_sim.items():
            for j, wij in related_items.items():
                i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])

        i2i_sim_ = self.min_max_normalize(i2i_sim_)

        pickle.dump(i2i_sim_, open(to_path, 'wb'))
        return i2i_sim_

    def item_based_recommend(self, user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num,
                             item_topk_click, item_created_time_dict, emb_i2i_sim):

        user_hist_items = user_item_time_dict[user_id]
        item_rank = {}

        # 遍历用户历史点击的每个物品，逐个处理
        for loc, (i, click_time) in enumerate(user_hist_items):
            if i not in i2i_sim:
                continue
            for j, wij in sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
                if j in user_hist_items:
                    continue
                created_time_weight = np.exp(0.8 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))
                loc_weight = (0.9 ** (len(user_hist_items) - loc))
                content_weight = 1.0
                if emb_i2i_sim.get(i, {}).get(j, None) is not None:
                    content_weight += emb_i2i_sim[i][j]
                if emb_i2i_sim.get(j, {}).get(i, None) is not None:
                    content_weight += emb_i2i_sim[j][i]

                fresh_weight = 1
                # 计算最终得分
                score = wij * fresh_weight * content_weight * created_time_weight * loc_weight
                item_rank.setdefault(j, 0)
                item_rank[j] += score

        # if len(item_rank) < recall_item_num:
        #     for i, item in enumerate(item_topk_click):
        #         if item in item_rank:
        #             continue
        #         item_rank[item] = - i - 100
        #         if len(item_rank) == recall_item_num:
        #             break

        # 返回根据得分排序后的物品推荐列表
        if len(item_rank) != 0:
            # # 对得分取对数
            item_rank_log = {item_id: math.log(score + 1) for item_id, score in item_rank.items()}

            item_rank = sorted(item_rank_log.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]
        return item_rank

    def recommend(self, last_num=2, sim_item_topk=60, recall_item_num=30, recall_user_list = None,
                  to_path='itemcf_recall_dict.pkl', sim_file_path='itemcf_i2i_sim_evalute_model_submit.pkl'):
        print('召回阶段数据读取')
        trn_hist_click_df = self.all_click_df


        trn_hist_click_df = trn_hist_click_df.sort_values(['user_id', 'click_timestamp'], ascending=[True, True])
        trn_hist_click_df = trn_hist_click_df.groupby('user_id').tail(last_num)

        user_recall_items_dict = collections.defaultdict(dict)
        user_item_time_dict = get_user_item_time(trn_hist_click_df)

        i2i_sim = pickle.load(open(sim_file_path, 'rb'))
        emb_i2i_sim = pickle.load(open(os.path.join(save_path, 'emb_i2i_sim.pkl'), 'rb'))

        item_topk_click = get_item_topk_click(trn_hist_click_df, k=100)

        print('开始召回')
        # 将推荐逻辑提取成独立的函数，避免 pickle 问题
        def parallel_recommend(user):
            return user, self.item_based_recommend(
                user, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num,
                item_topk_click, self.item_created_time_dict, emb_i2i_sim
            )

        from concurrent.futures import ThreadPoolExecutor
        # 使用线程池来加速推荐过程

        with ThreadPoolExecutor() as executor:

            futures = {executor.submit(parallel_recommend, user): user for user in
                       recall_user_list}
            for future in tqdm(futures,desc="Processing users", total=len(recall_user_list)):
                user, recommendation = future.result()
                user_recall_items_dict[user] = recommendation

        user_multi_recall_dict['itemcf_sim_itemcf_recall'] = user_recall_items_dict

        pickle.dump(user_multi_recall_dict['itemcf_sim_itemcf_recall'], open(to_path, 'wb'))

        norecall_num = 0;
        no_5_num = 0
        for user in recall_user_list:
            if len(user_recall_items_dict[user]) == 0:
                norecall_num += 1
            if len(user_recall_items_dict[user]) < 5 :
                no_5_num += 1
        print(f"{norecall_num} 个用户没有召回物品")
        print(f"{no_5_num} 个用户召回物品少于5个")
        return user_recall_items_dict



if __name__ == '__main__':

    all_click_df = pd.read_pickle(r'D:\AI\HZ_DeepLearning_Practice\news_recommender\tmp_results\train_click_df.pkl')
    test_click_df =pd.read_pickle(r'D:\AI\HZ_DeepLearning_Practice\news_recommender\tmp_results\valid_click_last_df.pkl')

    recall_user_list = test_click_df['user_id'].unique()
    recall_user_set = set(recall_user_list)


    module = ItemCF(all_click_df=all_click_df)

    module.itemcf_sim(module.all_click_df, module.item_created_time_dict,
                      to_path=os.path.join(save_path, 'itemcf_i2i_self_test_sim.pkl'))

    itemcf_recall_dict = module.recommend(last_num=2, sim_item_topk=60, recall_item_num=50,
                                          to_path=os.path.join(save_path, 'itemcf_recall_self_test_dict.pkl'),
                                          sim_file_path=os.path.join(save_path, 'itemcf_i2i_self_test_sim.pkl'),
                                          recall_user_list=recall_user_list)
    metrics_recall(itemcf_recall_dict,test_click_df,topk=30)
    pickle.dump(itemcf_recall_dict, open(os.path.join(save_path, 'itemcf_recall_self_test_dict.pkl'), 'wb'))
    plot_recall_score_distribution(itemcf_recall_dict)


