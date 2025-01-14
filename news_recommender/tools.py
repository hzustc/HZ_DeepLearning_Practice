import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os, math, warnings, math, pickle
from tqdm import tqdm
import faiss
import collections
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

from news_recommender.settings import save_path

warnings.filterwarnings('ignore')


# debug模式： 从训练集中划出一部分数据来调试代码
def get_train_and_test_click(data_path=r'D:\AI\HZ_DeepLearning_Practice\news_recommender\data_raw', sample_nums=150000):
    """
        训练集中采样一部分数据调试
        data_path: 原数据的存储路径
        sample_nums: 采样数目（这里由于机器的内存限制，可以采样用户做）
    """

    all_click_df = pd.read_csv(r'D:\AI\HZ_DeepLearning_Practice\news_recommender\data_raw\train_click_log.csv')

    #all_click_df = all_click_df.sample(1000)
    all_user_ids = all_click_df.user_id.unique()

    np.random.seed(42)
    train_user_ids = np.random.choice(all_user_ids, size=sample_nums, replace=False)
    valid_user_ids = list(set(all_user_ids)-set(train_user_ids))

    train_click_df = all_click_df[all_click_df['user_id'].isin(train_user_ids)]
    valid_click_df = all_click_df[all_click_df['user_id'].isin(valid_user_ids)]

    valid_click_df = valid_click_df.sort_values(by=['user_id', 'click_timestamp'])
    valid_click_last_df = valid_click_df.groupby('user_id').tail(1)

    # 如果用户只有一个点击，hist为空了，会导致训练的时候这个用户不可见，此时默认泄露一下
    def hist_func(user_df):
        if len(user_df) == 1:
            return user_df
        else:
            return user_df[:-1]

    valid_click_df = valid_click_df.groupby('user_id').apply(hist_func).reset_index(drop=True)

    testA_click_df = pd.read_csv(r'D:\AI\HZ_DeepLearning_Practice\news_recommender\data_raw\testA_click_log.csv')
    testB_click_df = pd.read_csv(r'D:\AI\HZ_DeepLearning_Practice\news_recommender\data_raw\testB_click_log.csv')
    length = len(testA_click_df)
    user_id_counts = testA_click_df['user_id'].value_counts()
    # 筛选出 user_id 出现次数大于 1 的行
    testA_click_df = testA_click_df[testA_click_df['user_id'].isin(user_id_counts[user_id_counts > 1].index)]
    print(f'delete {length - len(testA_click_df)} rows')

    length = len(testB_click_df)
    user_id_counts = testB_click_df['user_id'].value_counts()
    # 筛选出 user_id 出现次数大于 1 的行
    testB_click_df = testB_click_df[testB_click_df['user_id'].isin(user_id_counts[user_id_counts > 1].index)]
    print(f'delete {length - len(testB_click_df)} rows')

    train_click_df = pd.concat([train_click_df,valid_click_df, testA_click_df,testB_click_df]).drop_duplicates(['user_id', 'click_article_id', 'click_timestamp'])

    valid_click_last_df.to_pickle(r'D:\AI\HZ_DeepLearning_Practice\news_recommender\tmp_results\valid_click_last_df.pkl')
    train_click_df.to_pickle(r'D:\AI\HZ_DeepLearning_Practice\news_recommender\tmp_results\train_click_df.pkl')
    return train_click_df,valid_click_last_df


# 读取点击数据，这里分成线上和线下，如果是为了获取线上提交结果应该讲测试集中的点击数据合并到总的数据中
# 如果是为了线下验证模型的有效性或者特征的有效性，可以只使用训练集
def get_all_click_df(data_path=r'D:\AI\HZ_DeepLearning_Practice\news_recommender\data_raw', offline=True):
    if offline:
        all_click = pd.read_csv(r'D:\AI\HZ_DeepLearning_Practice\news_recommender\data_raw\train_click_log.csv')
    else:
        trn_click = pd.read_csv(r'D:\AI\HZ_DeepLearning_Practice\news_recommender\data_raw\train_click_log.csv')
        tst_click = pd.read_csv(r'D:\AI\HZ_DeepLearning_Practice\news_recommender\data_raw\testA_click_log.csv')

        all_click = trn_click.append(tst_click)

    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    return all_click


# 读取文章的基本属性
def get_item_info_df(data_path=r'D:\AI\HZ_DeepLearning_Practice\news_recommender\data_raw'):
    item_info_df = pd.read_csv(r'D:\AI\HZ_DeepLearning_Practice\news_recommender\data_raw\articles.csv')

    # 为了方便与训练集中的click_article_id拼接，需要把article_id修改成click_article_id
    item_info_df = item_info_df.rename(columns={'article_id': 'click_article_id'})

    return item_info_df


# %%
# 读取文章的Embedding数据
def get_item_emb_dict(data_path=r'D:\AI\HZ_DeepLearning_Practice\news_recommender\data_raw'):
    item_emb_df = pd.read_csv(r'D:\AI\HZ_DeepLearning_Practice\news_recommender\data_raw\articles_emb.csv')

    item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x]
    item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols])
    # 进行归一化
    item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)

    item_emb_dict = dict(zip(item_emb_df['article_id'], item_emb_np))
    pickle.dump(item_emb_dict, open(r'D:\AI\HZ_DeepLearning_Practice\news_recommender\tmp_results\item_content_emb.pkl', 'wb'))

    return item_emb_dict


max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))


# 根据点击时间获取用户的点击文章序列   {user1: [item1: time1, item2: time2..]...}
def get_user_item_time(click_df):
    click_df = click_df.sort_values('click_timestamp')

    def make_item_time_pair(df):
        return list(zip(df['click_article_id'], df['click_timestamp']))

    user_item_time_df = click_df.groupby('user_id')['click_article_id', 'click_timestamp'].apply(
        lambda x: make_item_time_pair(x)) \
        .reset_index().rename(columns={0: 'item_time_list'})
    user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))

    return user_item_time_dict


# 根据时间获取商品被点击的用户序列  {item1: [user1: time1, user2: time2...]...}
# 这里的时间是用户点击当前商品的时间，好像没有直接的关系。
def get_item_user_time_dict(click_df):
    def make_user_time_pair(df):
        return list(zip(df['user_id'], df['click_timestamp']))

    click_df = click_df.sort_values('click_timestamp')
    item_user_time_df = click_df.groupby('click_article_id')['user_id', 'click_timestamp'].apply(
        lambda x: make_user_time_pair(x)) \
        .reset_index().rename(columns={0: 'user_time_list'})

    item_user_time_dict = dict(zip(item_user_time_df['click_article_id'], item_user_time_df['user_time_list']))
    return item_user_time_dict


# %%
# 获取当前数据的历史点击和最后一次点击
def get_hist_and_last_click(all_click):
    all_click = all_click.sort_values(by=['user_id', 'click_timestamp'])
    click_last_df = all_click.groupby('user_id').tail(1)

    # 如果用户只有一个点击，hist为空了，会导致训练的时候这个用户不可见，此时默认泄露一下
    def hist_func(user_df):
        if len(user_df) == 1:
            return user_df
        else:
            return user_df[:-1]

    click_hist_df = all_click.groupby('user_id').apply(hist_func).reset_index(drop=True)

    return click_hist_df, click_last_df


# %%
# 获取文章id对应的基本属性，保存成字典的形式，方便后面召回阶段，冷启动阶段直接使用
def get_item_info_dict(item_info_df):
    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    item_info_df['created_at_ts'] = item_info_df[['created_at_ts']].apply(max_min_scaler)

    item_type_dict = dict(zip(item_info_df['click_article_id'], item_info_df['category_id']))
    item_words_dict = dict(zip(item_info_df['click_article_id'], item_info_df['words_count']))
    item_created_time_dict = dict(zip(item_info_df['click_article_id'], item_info_df['created_at_ts']))

    return item_type_dict, item_words_dict, item_created_time_dict


# %%
def get_user_hist_item_info_dict(all_click):
    # 获取user_id对应的用户历史点击文章类型的集合字典
    user_hist_item_typs = all_click.groupby('user_id')['category_id'].agg(set).reset_index()
    user_hist_item_typs_dict = dict(zip(user_hist_item_typs['user_id'], user_hist_item_typs['category_id']))

    # 获取user_id对应的用户点击文章的集合
    user_hist_item_ids_dict = all_click.groupby('user_id')['click_article_id'].agg(set).reset_index()
    user_hist_item_ids_dict = dict(zip(user_hist_item_ids_dict['user_id'], user_hist_item_ids_dict['click_article_id']))

    # 获取user_id对应的用户历史点击的文章的平均字数字典
    user_hist_item_words = all_click.groupby('user_id')['words_count'].agg('mean').reset_index()
    user_hist_item_words_dict = dict(zip(user_hist_item_words['user_id'], user_hist_item_words['words_count']))

    # 获取user_id对应的用户最后一次点击的文章的创建时间
    all_click_ = all_click.sort_values('click_timestamp')
    user_last_item_created_time = all_click_.groupby('user_id')['created_at_ts'].apply(
        lambda x: x.iloc[-1]).reset_index()

    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    user_last_item_created_time['created_at_ts'] = user_last_item_created_time[['created_at_ts']].apply(max_min_scaler)

    user_last_item_created_time_dict = dict(zip(user_last_item_created_time['user_id'], \
                                                user_last_item_created_time['created_at_ts']))

    return user_hist_item_typs_dict, user_hist_item_ids_dict, user_hist_item_words_dict, user_last_item_created_time_dict


# %%
# 获取近期点击最多的文章
def get_item_topk_click(click_df, k):
    topk_click = click_df['click_article_id'].value_counts().index[:k]
    return topk_click


# 依次评估召回的前10, 20, 30, 40, 50个文章中的击中率
def metrics_recall(user_recall_items_dict, trn_last_click_df, topk=5):
    """
    计算召回的命中率（Hit Rate）和平均倒数排名（MRR）
    :param user_recall_items_dict: 用户召回的物品字典，格式为 {user_id: [(item_id, pred_prob), ...]}
    :param trn_last_click_df: 每个用户最后一次点击的物品数据，包含 ['user_id', 'click_article_id']
    :param topk: 评估的top K值
    """
    last_click_item_dict = dict(zip(trn_last_click_df['user_id'], trn_last_click_df['click_article_id']))
    user_num = len(last_click_item_dict)

    # 计算Hit Rate
    print("Hit Rate:")
    for k in range(5, topk + 1, 5):
        hit_num = 0
        for user, item_list in user_recall_items_dict.items():
            if len(item_list) == 0 :
                continue
            if user in last_click_item_dict:

                # 获取前k个召回的物品
                tmp_recall_items = [x[0] for x in user_recall_items_dict[user][:k]]
                # 检查用户的最后点击是否在召回列表中
                if  last_click_item_dict[user] in set(tmp_recall_items):
                    hit_num += 1

        hit_rate = round(hit_num * 1.0 / user_num, 5)
        print(f"Topk: {k}, Hit Num: {hit_num}, Hit Rate: {hit_rate}, User Num: {user_num}")

    # 计算MRR
    print("\nMRR (Mean Reciprocal Rank):")
    for k in range(5, topk + 1, 5):
        mrr = 0
        for user, item_list in user_recall_items_dict.items():
            if len(item_list) == 0 :
                continue
            if user in last_click_item_dict:

                # 获取前k个召回的物品
                tmp_recall_items = [x[0] for x in user_recall_items_dict[user][:k]]
                for loc, item in enumerate(tmp_recall_items):
                    if  item == last_click_item_dict[user]:
                        mrr += 1 / (loc + 1)
                        break  # 找到匹配的物品后结束循环
        mrr = round(mrr * 1.0 / user_num, 5)
        print(f"Topk: {k}, MRR: {mrr}")



def get_user_activate_degree_dict(all_click_df):
    all_click_df_ = all_click_df.groupby('user_id')['click_article_id'].count().reset_index()

    # 用户活跃度归一化
    mm = MinMaxScaler()
    all_click_df_['click_article_id'] = mm.fit_transform(all_click_df_[['click_article_id']])
    user_activate_degree_dict = dict(zip(all_click_df_['user_id'], all_click_df_['click_article_id']))

    return user_activate_degree_dict


def combine_recall_results(user_multi_recall_dict, weight_dict=None, topk=25):
    final_recall_items_dict = {}

    # 对每一种召回结果按照用户进行归一化，方便后面多种召回结果，相同用户的物品之间权重相加
    def norm_user_recall_items_sim(sorted_item_list):
        # 如果冷启动中没有文章或者只有一篇文章，直接返回，出现这种情况的原因可能是冷启动召回的文章数量太少了，
        # 基于规则筛选之后就没有文章了, 这里还可以做一些其他的策略性的筛选
        if len(sorted_item_list) < 2:
            return sorted_item_list

        min_sim = sorted_item_list[-1][1]
        max_sim = sorted_item_list[0][1]

        norm_sorted_item_list = []
        for item, score in sorted_item_list:
            if max_sim > 0:
                norm_score = 1.0 * (score - min_sim) / (max_sim - min_sim) if max_sim > min_sim else 1.0
            else:
                norm_score = 0.0
            norm_sorted_item_list.append((item, norm_score))

        return norm_sorted_item_list

    print('多路召回合并...')
    for method, user_recall_items in tqdm(user_multi_recall_dict.items()):
        print(method + '...')
        # 在计算最终召回结果的时候，也可以为每一种召回结果设置一个权重
        if weight_dict == None:
            recall_method_weight = 1
        else:
            recall_method_weight = weight_dict[method]

        for user_id, sorted_item_list in user_recall_items.items():  # 进行归一化
            user_recall_items[user_id] = norm_user_recall_items_sim(sorted_item_list)

        for user_id, sorted_item_list in user_recall_items.items():
            # print('user_id')
            final_recall_items_dict.setdefault(user_id, {})
            for item, score in sorted_item_list:
                final_recall_items_dict[user_id].setdefault(item, 0)
                final_recall_items_dict[user_id][item] += recall_method_weight * score

    final_recall_items_dict_rank = {}
    # 多路召回时也可以控制最终的召回数量
    for user, recall_item_dict in final_recall_items_dict.items():
        final_recall_items_dict_rank[user] = sorted(recall_item_dict.items(), key=lambda x: x[1], reverse=True)[:topk]

    # 将多路召回后的最终结果字典保存到本地
    pickle.dump(final_recall_items_dict, open(os.path.join(save_path, 'final_recall_items_dict.pkl'), 'wb'))

    return final_recall_items_dict_rank







#%%
# 定义一个多路召回的字典，将各路召回的结果都保存在这个字典当中
user_multi_recall_dict =  {'itemcf_sim_itemcf_recall': {},
                           'embedding_sim_item_recall': {},
                           }



#%%
# 这里直接对多路召回的权重给了一个相同的值，其实可以根据前面召回的情况来调整参数的值
weight_dict = {'itemcf_sim_itemcf_recall': 1.0,
               'embedding_sim_item_recall': 1.0,
               }


