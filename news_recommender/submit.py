import pandas as pd
import pickle

# 读取用户点击日志数据
valid_click_df = pd.read_csv(r'D:\AI\HZ_DeepLearning_Practice\news_recommender\data_raw\testA_click_log.csv')

# 加载推荐结果
# itemcf_recall = pickle.load(open(r'D:\AI\HZ_DeepLearning_Practice\news_recommender\tmp_results\lightgbm_recall_dict.pkl', 'rb'))
itemcf_recall = pickle.load(open(r'D:\AI\HZ_DeepLearning_Practice\news_recommender\rank_data\test_merged_recall_dict.pkl', 'rb'))
# 获取用户ID列表
userlist = set(valid_click_df['user_id'].values)

# 创建一个列表用于存放最终结果
submit_list = []

# 遍历每个用户并为他们生成推荐
for user in userlist:
    if user in itemcf_recall and len(itemcf_recall[user]) >= 5:
        # 提取前5篇推荐的文章
        articles = [itemcf_recall[user][i][0] for i in range(5)]
        # 将结果追加到列表中
        submit_list.append([user] + articles)
    else:
        print(f"Warning: No recommendation for user {user}")

# 将提交列表转换为 DataFrame
submit = pd.DataFrame(submit_list, columns=['user_id', 'article_1', 'article_2', 'article_3', 'article_4', 'article_5'])

# 按 user_id 排序
submit = submit.sort_values(by='user_id')

# 保存提交文件为 CSV
submit.to_csv(r'D:\AI\HZ_DeepLearning_Practice\news_recommender\tmp_results\submit.csv', index=False)
