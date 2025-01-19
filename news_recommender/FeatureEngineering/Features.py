import os.path
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from news_recommender.tools import get_train_and_test_click, get_hist_and_last_click, get_item_emb_dict


class Features:
    def __init__(self, ):
        pass

    def negative_sampling(self, recall_dict, label_click_df, sample_rate=0.01):
        print('负采样')
        # 将 recall_dict 列表转换为 DataFrame
        data_df = self.generate_user_item_scroe(recall_dict, None)
        data_df = self.generate_labels(data_df, label_click_df)
        pos_data_df = data_df[data_df['label'] == 1]
        neg_data_df = data_df[data_df['label'] == 0]
        print('pos_data_num:', len(pos_data_df), 'neg_data_num:', len(neg_data_df), 'pos/neg:',
              len(pos_data_df) / len(neg_data_df))

        # 分组采样函数
        def neg_sample_func(group_df):
            neg_num = len(group_df)
            sample_num = max(int(neg_num * sample_rate), 1)  # 保证最少有一个
            sample_num = min(sample_num, 5)  # 保证最多不超过5个，这里可以根据实际情况进行选择
            return group_df.sample(n=sample_num, replace=True)

        # 对用户进行负采样，保证所有用户都在采样后的数据中
        neg_data_user_sample = neg_data_df.groupby('user_id', group_keys=False).apply(neg_sample_func)
        # 对文章进行负采样，保证所有文章都在采样后的数据中
        neg_data_item_sample = neg_data_df.groupby('click_article_id', group_keys=False).apply(neg_sample_func)

        # 将上述两种情况下的采样数据合并
        neg_data_new_df = pd.concat([neg_data_user_sample, neg_data_item_sample], ignore_index=True)
        # 由于上述两个操作是分开的，可能将两个相同的数据给重复选择了，所以需要对合并后的数据进行去重
        neg_data_new_df = neg_data_new_df.sort_values(['user_id', 'score']).drop_duplicates(
            ['user_id', 'click_article_id'],
            keep='last')
        print('pos_data_num:', len(pos_data_df), 'neg_data_num:', len(neg_data_df), 'pos/neg:',
              len(pos_data_df) / len(neg_data_new_df))
        # 将正样本数据合并
        data_new_df = pd.concat([pos_data_df, neg_data_new_df], ignore_index=True)

        return data_new_df

    # 负采样函数，这里可以控制负采样时的比例, 这里给了一个默认的值

    def generate_user_item_scroe(self, recall_dict, avg_sample_num=None):
        # 使用列表推导式直接构造数据
        print('召回字典生成数据集')
        data = [
            [user, item, score]
            for user, item_score_list in tqdm(recall_dict.items())
            for item, score in item_score_list[:avg_sample_num]
        ]

        # 将 data 列表转换为 DataFrame
        columns = ['user_id', 'click_article_id', 'score']
        data_df = pd.DataFrame(data, columns=columns)
        return data_df

    def generate_labels(self, df, label_df):
        print('生成label')
        user_last_click_dict = label_df.set_index('user_id')['click_article_id'].to_dict()
        df['label'] = df['user_id'].map(user_last_click_dict) == df['click_article_id']
        df['label'] = df['label'].astype(int)
        return df

    def generate_time_diff(self, df, train_last_df):
        print('生成特征：time_diff')
        # 读取文章信息表
        item_info_df = pd.read_csv(r'D:\AI\HZ_DeepLearning_Practice\news_recommender\data_raw\articles.csv')

        # 合并 data 和 train_last_df，获取用户的最后点击时间
        merged_df = pd.merge(df, train_last_df[['user_id', 'click_timestamp']], on='user_id', how='left')

        # 合并文章信息，获取文章创建时间
        merged_df = pd.merge(merged_df, item_info_df[['article_id', 'created_at_ts']],
                             left_on='click_article_id', right_on='article_id', how='left')

        # 计算时间差
        merged_df['time_diff'] = (merged_df['click_timestamp'] - merged_df['created_at_ts']) / 3600000

        # 选择需要的列
        df = merged_df[list(df.columns) + ['time_diff']]
        return df

    def generate_item_info(self, df, ):
        print('生成特征：文章原始特征')
        # 读取文章信息表
        item_info_df = pd.read_csv(r'D:\AI\HZ_DeepLearning_Practice\news_recommender\data_raw\articles.csv')

        #召回物品的原始特征
        df = pd.merge(df, item_info_df, left_on='click_article_id', right_on='article_id', how='left')

        return df

    def generate_user_info(self, df, all_click_df, last_click_df):
        item_info_df = pd.read_csv(r'D:\AI\HZ_DeepLearning_Practice\news_recommender\data_raw\articles.csv')
        print('生成特征：用户统计特征')
        user_group_all_click_df = all_click_df.groupby('user_id')

        #用户最后一次点击的环境信息
        last_click_env_df = last_click_df.drop(columns=['click_article_id'])
        df = pd.merge(df, last_click_env_df, on='user_id', how='left')

        #用户最后一次点击文章类型
        temp_df = pd.merge(last_click_df[['user_id', 'click_article_id']], item_info_df, left_on=
        'click_article_id', right_on='article_id', how='left')
        temp_df.rename(columns={'click_article_id': 'last_click_article_id', 'category_id': 'last_click_category_id'},
                       inplace=True)
        df = pd.merge(df, temp_df[['user_id', 'last_click_category_id','last_click_article_id']], on='user_id', how='left')

        #用户最后一次点击文章类型与召回物品是否一致
        df['last_click_category_id_match'] = (df['last_click_category_id'] == df['category_id']).astype(int)

        #用户最后一次点击与召回物品的embedding相似度
        item_emb_dict = get_item_emb_dict()
        df['emb_sim'] = df.apply(lambda x: item_emb_dict[x['click_article_id']].dot(item_emb_dict[x['last_click_article_id']]), axis=1)

        return df

    def generate_multiple_recall_score(self, df, recall_dict_dict):
        print('生成特征：多路召回分数')

        for recall_name, recall_dict in recall_dict_dict.items():
            # 将 recall_dict 转换为 DataFrame 格式
            recall_df = pd.DataFrame(
                [(user_id, item_id, score) for user_id, items in recall_dict.items() for item_id, score in items],
                columns=['user_id', 'click_article_id', recall_name + '_score']
            )

            # 使用 merge 将召回分数合并到原始 df
            df = pd.merge(df, recall_df, on=['user_id', 'click_article_id'], how='left')

        return df

    def save(self, df, file_path=r'D:\AI\HZ_DeepLearning_Practice\news_recommender\tmp_results\recall_features.pkl'):
        df.to_pickle(file_path)
        print('保存数据集:success')

    def generate_features(self, train_click_df=None, df=None, recall_dict=None):
        print('生成特征')

        _, train_last_click_df = get_hist_and_last_click(train_click_df)

        df = self.generate_time_diff(df, train_last_click_df)
        df = self.generate_item_info(df)
        df = self.generate_user_info(df, train_click_df, train_last_click_df)
        df = self.generate_multiple_recall_score(df, recall_dict)

        return df


if __name__ == '__main__':
    valid_click_last_df = pd.read_pickle(
        r'D:\AI\HZ_DeepLearning_Practice\news_recommender\tmp_results\valid_click_last_df.pkl')
    train_click_df = pd.read_pickle(r'D:\AI\HZ_DeepLearning_Practice\news_recommender\tmp_results\train_click_df.pkl')
    _, label_click_df = get_hist_and_last_click(train_click_df)

    file_path = r'D:\AI\HZ_DeepLearning_Practice\news_recommender\tmp_results\merged_recall_dict_sum.pkl'
    recall_dict = pickle.load(open(file_path, 'rb'))

    train_dataset_without_label = Features().generate_features(train_click_df=train_click_df, recall_dict=recall_dict)
    Features().save(train_dataset_without_label,
                    os.path.join(r'D:\AI\HZ_DeepLearning_Practice\news_recommender\tmp_results',
                                 'train_dataset_without_label.pkl'))
    train_dataset = Features().generate_labels(train_dataset_without_label, label_click_df)
    Features().save(train_dataset, os.path.join(r'D:\AI\HZ_DeepLearning_Practice\news_recommender\tmp_results',
                                                'train_dataset_with_label.pkl'))
