import pickle
from memory_profiler import profile
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from news_recommender.tools import metrics_recall
import pandas as pd
from matplotlib import rcParams



def min_max_normalize(item_rank):
    """
    对每个用户的物品分数进行最大最小归一化
    :param item_rank: 用户的物品分数列表 [(article_id, score), ...]
    :return: 归一化后的 [(article_id, normalized_score), ...]
    """
    if not item_rank:
        return item_rank

    scores = [score for _, score in item_rank]

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return item_rank  # 如果最大值和最小值相同，直接返回原分数

    return [(article_id, (score - min_score) / (max_score - min_score)) for article_id, score in item_rank]


def min_max_normalize_global(scores):
    """
    对所有分数进行全局最大最小归一化
    :param scores: 所有分数的列表
    :return: 归一化后的分数列表
    """
    if not scores:
        return scores

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return scores  # 如果最大值和最小值相等，直接返回原分数

    return [(score - min_score) / (max_score - min_score) for score in scores]


def plot_score_distribution(before_scores, after_scores, file_name, global_normalize):
    """
    绘制归一化前后的分数分布
    :param before_scores: 归一化前的分数
    :param after_scores: 归一化后的分数
    :param file_name: 召回字典文件名，用于图表标题
    :param global_normalize: 是否为全局归一化，影响图表标题
    """
    plt.figure(figsize=(10, 6))

    # 设置seaborn样式
    sns.set(style="whitegrid")

    # 设置字体为 SimHei（黑体）
    rcParams['font.sans-serif'] = ['SimHei']  # 用来显示中文
    rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 归一化前的分数分布
    sns.kdeplot(before_scores, label="归一化前", color="blue", fill=True)

    # 归一化后的分数分布
    sns.kdeplot(after_scores, label="归一化后", color="red", fill=True)

    # 设置图表标题
    title = f"{file_name} 归一化前后分数分布 ({'全局归一化' if global_normalize else '按用户归一化'})"
    plt.title(title)

    plt.xlabel("分数")
    plt.ylabel("概率密度")

    plt.legend()
    plt.show()

#@profile
def merge_recall_dicts(recall_dict_files, merge_strategy="sum", weights=None, normalize=False, global_normalize=False):
    """
    合并多个召回字典，支持全局或按用户归一化和自定义权重
    并绘制每个召回字典中物品分数归一化前后的分布图
    :param recall_dict_files: 保存多个召回字典的 pickle 文件路径列表
    :param merge_strategy: 合并策略，默认为 "sum"（分数累加），可以选择 "max"（取最大分数）或 "avg"（取平均分）
    :param weights: 权重列表，长度应与 recall_dict_files 相同，默认为 None 时各字典权重相等
    :param normalize: 是否进行归一化，默认为 True
    :param global_normalize: 是否进行全局归一化，默认为 False（按用户归一化）
    :return: 合并后的召回字典
    """
    merged_recall_dict = defaultdict(lambda: defaultdict(list))  # 默认值为 list，用于存储多个分数

    if weights is None:
        weights = [1] * len(recall_dict_files)  # 如果没有指定权重，默认每个字典权重为 1

    for i, file in enumerate(recall_dict_files):
        with open(file, 'rb') as f:
            recall_dict = pickle.load(f)

        weight = weights[i]  # 获取当前字典的权重

        # 提取原始分数
        before_scores = [score for user_items in recall_dict.values() for _, score in user_items]

        if normalize and global_normalize:
            # 对所有分数进行全局归一化
            normalized_scores = min_max_normalize_global(before_scores)
            normalized_index = 0

        normalized_item_rank = []
        after_scores = []

        for user_id, item_rank in recall_dict.items():
            if item_rank:  # 如果召回列表非空
                if normalize:
                    if global_normalize:
                        # 使用归一化后的全局分数替换原始分数
                        for article_id, _ in item_rank:
                            normalized_item_rank.append((article_id, normalized_scores[normalized_index]))
                            after_scores.append(normalized_scores[normalized_index])
                            normalized_index += 1
                    else:
                        # 按用户归一化
                        normalized_item_rank = min_max_normalize(item_rank)
                        after_scores.extend([score for _, score in normalized_item_rank])
                else:
                    normalized_item_rank = item_rank
                    after_scores.extend([score for _, score in item_rank])

                for article_id, score in normalized_item_rank:
                    merged_recall_dict[user_id][article_id].append(score * weight)

        # 绘制当前召回字典的归一化前后的分数分布
        #plot_score_distribution(before_scores, after_scores, file, global_normalize)

    # 根据合并策略处理
    for user_id, items in merged_recall_dict.items():
        if items:  # 仅对非空召回结果进行处理
            if merge_strategy == "sum":
                for article_id in items:
                    merged_recall_dict[user_id][article_id] = sum(items[article_id])  # 累加分数
            elif merge_strategy == "max":
                for article_id in items:
                    merged_recall_dict[user_id][article_id] = max(items[article_id])  # 取最大分数
            elif merge_strategy == "avg":
                for article_id in items:
                    merged_recall_dict[user_id][article_id] = sum(items[article_id]) / len(items[article_id])  # 取平均分数

            # 将合并后的字典转换为按分数降序排列的形式
            sorted_items = sorted(items.items(), key=lambda x: x[1], reverse=True)  # 按分数排序
            merged_recall_dict[user_id] = sorted_items
        else:
            # 用户的召回列表为空时，可以选择返回空的推荐列表
            merged_recall_dict[user_id] = []

    return merged_recall_dict




# 示例用法
if __name__ == '__main__':



    recall_dict_files = [
        r'D:\AI\HZ_DeepLearning_Practice\news_recommender\tmp_results\hot_recall_self_test_dict.pkl',
        r'D:\AI\HZ_DeepLearning_Practice\news_recommender\tmp_results\itemcf_recall_self_test_dict.pkl',
        # 添加更多文件路径
    ]
    test_click_df =pd.read_pickle(r'D:\AI\HZ_DeepLearning_Practice\news_recommender\tmp_results\valid_click_last_df.pkl')

    # recall_dict_files = [
    #     r'D:\AI\HZ_DeepLearning_Practice\news_recommender\tmp_results\hot_recall_dict.pkl',
    #     r'D:\AI\HZ_DeepLearning_Practice\news_recommender\tmp_results\itemcf_recall_dict.pkl',
    #     # 添加更多文件路径
    # ]
    # test_click_df =pd.read_pickle(r'D:\AI\HZ_DeepLearning_Practice\news_recommender\tmp_results\valid_click_last_df.pkl')

    # 定义每个召回字典的权重
    weights = [1, 1.2]  # 第一个召回字典的权重为0.7，第二个为0.3

    merge_model = 'sum'
    # 使用 sum 策略和自定义权重合并召回字典
    merged_recall_dict = dict(merge_recall_dicts(recall_dict_files, merge_strategy=merge_model, weights=weights,normalize=True,global_normalize=False))
    metrics_recall(merged_recall_dict, test_click_df, topk=50)
    # 保存合并后的召回字典
    with open(rf'D:\AI\HZ_DeepLearning_Practice\news_recommender\tmp_results\merged_recall_self_test_dict_{merge_model}.pkl',
              'wb') as f:
        pickle.dump(merged_recall_dict, f)

    # 可选：评估合并后的召回效果
    from news_recommender.tools import metrics_recall

    # valid_click_last_df = pd.read_pickle(
    #     r'D:\AI\HZ_DeepLearning_Practice\news_recommender\tmp_results\valid_click_last_df.pkl')
    # metrics_recall(merged_recall_dict, valid_click_last_df, topk=50)
