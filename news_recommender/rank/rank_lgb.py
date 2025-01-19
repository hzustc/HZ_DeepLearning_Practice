#%%
import os
import pickle
import lightgbm as lgb
import pandas as pd
from news_recommender.FeatureEngineering.Features import Features
from news_recommender.Recalls.hot_level_recall import HotLevelRecall
from news_recommender.Recalls.itemcf_recall import ItemCF
from news_recommender.muti_recall_merge import merge_recall_dicts
from news_recommender.tools import metrics_recall, get_hist_and_last_click

#%%
# 常量定义
SAVE_PATH = r'D:\AI\HZ_DeepLearning_Practice\news_recommender\rank_data'
TRAIN_CLICK_PATH = r'D:\AI\HZ_DeepLearning_Practice\news_recommender\tmp_results\train_click_df.pkl'
TEST_CLICK_PATH = r'D:\AI\HZ_DeepLearning_Practice\news_recommender\tmp_results\valid_click_last_df.pkl'
#
# TRAIN_CLICK_PATH = r'D:\AI\HZ_DeepLearning_Practice\news_recommender\data_raw\train_and_testA.pkl'
# TEST_CLICK_PATH = r'D:\AI\HZ_DeepLearning_Practice\news_recommender\data_raw\testA_last_click_df.pkl'

def load_data(train_path, test_path):
    """加载训练和测试数据"""
    all_click_df = pd.read_pickle(train_path)
    test_click_df = pd.read_pickle(test_path)
    hist_click_df, label_click_df = get_hist_and_last_click(all_click_df)
    return all_click_df, hist_click_df, label_click_df, test_click_df

def generate_itemcf_recall_dict(click_df, recall_user_list,save_path, prefix="",recall_nums = 30):
    # ItemCF召回
    itemcf_module = ItemCF(all_click_df=click_df)
    itemcf_module.itemcf_sim(
        itemcf_module.all_click_df, itemcf_module.item_created_time_dict,
        to_path=os.path.join(save_path, f'{prefix}itemcf_i2i_sim.pkl')
    )
    itemcf_recall_dict = itemcf_module.recommend(
        last_num=2, sim_item_topk=60, recall_item_num=recall_nums,
        to_path=os.path.join(save_path, f'{prefix}itemcf_recall_dict.pkl'),
        sim_file_path=os.path.join(save_path, f'{prefix}itemcf_i2i_sim.pkl'),
        recall_user_list=recall_user_list
    )
    return itemcf_recall_dict

def generate_hot_recall_dict(click_df, recall_user_list, save_path, prefix="",recall_nums = 20,recall_time_range = 1):
    hot_recall_module = HotLevelRecall(click_df, recall_nums=recall_nums, hours=recall_time_range)
    hot_recall_dict = hot_recall_module.hot_level_recall_parallel(recall_user_list)
    hot_recall_module.save(
        to_path=os.path.join(save_path, f'{prefix}hot_recall_dict.pkl'),
        recall_dict=hot_recall_dict
        )

    return hot_recall_dict

def generate_recall_dict(click_df, recall_user_list, save_path, prefix=""):
    """生成召回字典并保存"""
    # ItemCF召回
    itemcf_module = ItemCF(all_click_df=click_df)
    itemcf_module.itemcf_sim(
        itemcf_module.all_click_df, itemcf_module.item_created_time_dict,
        to_path=os.path.join(save_path, f'{prefix}itemcf_i2i_sim.pkl')
    )
    itemcf_recall_dict = itemcf_module.recommend(
        last_num=2, sim_item_topk=60, recall_item_num=100,
        to_path=os.path.join(save_path, f'{prefix}itemcf_recall_dict.pkl'),
        sim_file_path=os.path.join(save_path, f'{prefix}itemcf_i2i_sim.pkl'),
        recall_user_list=recall_user_list
    )

    # 热门召回
    hot_recall_module = HotLevelRecall(click_df, recall_nums=50, hours=10)
    hot_recall_dict = hot_recall_module.hot_level_recall_parallel(recall_user_list)
    hot_recall_module.save(
        to_path=os.path.join(save_path, f'{prefix}hot_recall_dict.pkl'),
        recall_dict=hot_recall_dict
    )

    # 合并召回结果
    recall_dict_files = [
        os.path.join(save_path, f'{prefix}hot_recall_dict.pkl'),
        os.path.join(save_path, f'{prefix}itemcf_recall_dict.pkl'),
    ]
    weights = [0.8, 1]
    merged_recall_dict = dict(merge_recall_dicts(recall_dict_files, merge_strategy='sum', weights=weights))

    # 保存合并后的召回字典
    with open(os.path.join(save_path, f'{prefix}merged_recall_dict.pkl'), 'wb') as f:
        pickle.dump(merged_recall_dict, f)

    return merged_recall_dict


def generate_features_and_labels(click_df, recall_dict, save_path, prefix="",negative_sample = True,sample_rate = 0.01,label_df = None,sample_num = 20,mutiple_recall_dict = None):
    """生成特征和标签数据集"""
    features = Features()
    if negative_sample:
        dataset = features.negative_sampling(recall_dict, label_df,sample_rate)
    else:
        dataset = features.generate_user_item_scroe(recall_dict, avg_sample_num=sample_num)
        dataset = features.generate_labels(dataset, label_df)
    dataset = features.generate_features(click_df, dataset,mutiple_recall_dict)
    features.save(dataset, os.path.join(save_path, f'{prefix}dataset_with_label.pkl'))
    return dataset


def train_lightgbm(X_train, y_train, X_test, y_test,categorical_feature=[]):
    """训练LightGBM模型"""
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'early_stopping_rounds': 10,
        'is_unbalance': True,
    }
    gbm = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=[lgb_eval],categorical_feature=categorical_feature)
    return gbm


def evaluate_model(model, X_test, test_label_click_df,columns):
    """评估模型并生成推荐结果"""
    y_pred_proba = model.predict(X_test[columns], num_iteration=model.best_iteration)
    X_test['pred_prob'] = y_pred_proba
    top_5_recommendations = X_test.groupby('user_id').apply(
        lambda x: x.sort_values('pred_prob', ascending=False).head(5)
    ).reset_index(drop=True)
    lightgbm_recall_dict = top_5_recommendations.groupby('user_id').apply(
        lambda x: list(zip(x['click_article_id'], x['pred_prob']))
    ).to_dict()
    pickle.dump(lightgbm_recall_dict,open(r'D:\AI\HZ_DeepLearning_Practice\news_recommender\tmp_results\lightgbm_recall_dict.pkl', 'wb'))
    metrics_recall(lightgbm_recall_dict, test_label_click_df, topk=5)




#%%
if __name__ == "__main__":
    #%%
    # 加载数据
    all_click_df, hist_click_df, label_click_df, test_click_df = load_data(TRAIN_CLICK_PATH, TEST_CLICK_PATH)
    #%%
    weights = [0.8, 1]
    itemcf_recall_nums = 50
    hot_level_recall_nums = 50
    hot_level_recall_time_range=1
    normalize = True
    neg_sample = True
    merge_strategy = 'sum'
    prefix = 'test_'
    sample_num = 50
    #%%
    # 训练集召回
    train_recall_user_list = label_click_df['user_id'].unique()
    #%%
    train_itemcf_recall_dict = generate_itemcf_recall_dict(hist_click_df, train_recall_user_list, SAVE_PATH,recall_nums=itemcf_recall_nums)

    #%%
    train_hot_recall_dict = generate_hot_recall_dict(hist_click_df, train_recall_user_list, SAVE_PATH,recall_nums=hot_level_recall_nums,recall_time_range=hot_level_recall_time_range)

    #%%
    train_itemcf_recall_dict = pickle.load(open(os.path.join(SAVE_PATH ,'itemcf_recall_dict.pkl') ,'rb'))
    train_hot_recall_dict = pickle.load(open(os.path.join(SAVE_PATH ,'hot_recall_dict.pkl') ,'rb'))
#%%
    train_mutiple_recall_dict_dict = {
        'itemcf_recall_dict': train_itemcf_recall_dict,
        'hot_recall_dict': train_hot_recall_dict
    }

#%%
    # 合并召回结果
    recall_dict_files = [
        os.path.join(SAVE_PATH, f'hot_recall_dict.pkl'),
        os.path.join(SAVE_PATH, f'itemcf_recall_dict.pkl'),
    ]

    train_merged_recall_dict = dict(merge_recall_dicts(recall_dict_files, merge_strategy=merge_strategy, weights=weights,normalize=normalize))

    # 保存合并后的召回字典
    with open(os.path.join(SAVE_PATH, f'merged_recall_dict.pkl'), 'wb') as f:
        pickle.dump(train_merged_recall_dict, f)
#%%
    train_merged_recall_dict = pickle.load(open(os.path.join(SAVE_PATH ,'merged_recall_dict.pkl') ,'rb'))
#%%
    # 训练集评估
    metrics_recall(train_merged_recall_dict, label_click_df, topk=30)
#%%

    # 测试集召回
    test_recall_user_list = test_click_df['user_id'].unique()
    #%%
    test_itemcf_recall_dict = generate_itemcf_recall_dict(all_click_df, test_recall_user_list, SAVE_PATH, prefix=prefix,recall_nums=itemcf_recall_nums)

    #%%
    test_hot_recall_dict = generate_hot_recall_dict(all_click_df, test_recall_user_list, SAVE_PATH, prefix=prefix,recall_nums=hot_level_recall_nums,recall_time_range=hot_level_recall_time_range)
    #%%
    test_itemcf_recall_dict = pickle.load(open(os.path.join(SAVE_PATH ,'test_itemcf_recall_dict.pkl') ,'rb'))
    test_hot_recall_dict = pickle.load(open(os.path.join(SAVE_PATH ,'test_hot_recall_dict.pkl') ,'rb'))
    #%%

    test_mutiple_recall_dict_dict = {
        'itemcf_recall_dict': train_itemcf_recall_dict,
        'hot_recall_dict': test_hot_recall_dict
    }
#%%

    # 合并召回结果
    test_recall_dict_files = [
        os.path.join(SAVE_PATH, f'test_hot_recall_dict.pkl'),
        os.path.join(SAVE_PATH, f'test_itemcf_recall_dict.pkl'),
    ]

    test_merged_recall_dict = dict(merge_recall_dicts(test_recall_dict_files, merge_strategy=merge_strategy, weights=weights, normalize=normalize))

    # 保存合并后的召回字典
    with open(os.path.join(SAVE_PATH, f'test_merged_recall_dict.pkl'), 'wb') as f:
        pickle.dump(test_merged_recall_dict, f)
#%%
    test_merged_recall_dict = pickle.load(open(os.path.join(SAVE_PATH ,'test_merged_recall_dict.pkl') ,'rb'))
#%%
    # 测试集评估
    metrics_recall(test_merged_recall_dict, test_click_df, topk=30)
#%%
    # 训练集特征和标签
    train_dataset = generate_features_and_labels(hist_click_df, train_merged_recall_dict, SAVE_PATH,label_df=label_click_df,negative_sample=neg_sample,sample_num=sample_num,mutiple_recall_dict=train_mutiple_recall_dict_dict)
#%%
    # 测试集特征和标签
    test_dataset = generate_features_and_labels(all_click_df, test_merged_recall_dict, SAVE_PATH, prefix=prefix,negative_sample=False,label_df=test_click_df,sample_num=sample_num,mutiple_recall_dict=test_mutiple_recall_dict_dict)

#%%
    # 训练LightGBM模型
    Xcol = ['score', 'time_diff','category_id','created_at_ts','words_count','click_timestamp','click_environment','click_deviceGroup','click_os','click_country','click_region','click_referrer_type','last_click_category_id','emb_sim','itemcf_recall_dict_score','hot_recall_dict_score','last_click_category_id_match']
    category_features = ['category_id','click_environment','click_deviceGroup','click_os','click_country','click_region','click_referrer_type','last_click_category_id','last_click_category_id_match']
    X_train, y_train = train_dataset[Xcol], train_dataset['label']
    X_test, y_test = test_dataset[Xcol], test_dataset['label']
    gbm = train_lightgbm(X_train, y_train, X_test, y_test, category_features)

#%%
    # 评估模型
    evaluate_model(gbm, test_dataset, test_click_df, Xcol)
#%%
    # 输出特征重要性（默认是 'split' 类型）
    importance = gbm.feature_importance(importance_type='gain')
    feature_names = Xcol

    # 将特征重要性和特征名称组合在一起
    feature_importance = list(zip(feature_names, importance))

    # 按重要性排序
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    # 打印特征重要性
    for feature, importance_value in feature_importance:
        print(f"{feature}: {importance_value}")