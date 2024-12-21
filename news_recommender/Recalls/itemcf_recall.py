from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import collections
import numpy as np
import math
from tqdm import tqdm
from news_recommender.settings import save_path, metric_recall, data_path
from news_recommender.tools import get_user_item_time, get_hist_and_last_click, get_item_topk_click, \
    user_multi_recall_dict, metrics_recall, get_item_info_df, get_item_emb_dict, get_item_info_dict, \
    get_all_click_sample, get_all_click_df


class ItemCF:
    def __init__(self, sample=True):
        self.all_click_df = get_all_click_sample(data_path) if sample else get_all_click_df(data_path)
        self.item_info_df = get_item_info_df(data_path)
        # 获取文章的属性信息，保存成字典的形式方便查询
        self.item_type_dict, self.item_words_dict, self.item_created_time_dict = get_item_info_dict(self.item_info_df)

    def itemcf_sim(self, df, item_created_time_dict):
        user_item_time_dict = get_user_item_time(df)
        i2i_sim = {}
        item_cnt = collections.defaultdict(int)

        for user, item_time_list in tqdm(user_item_time_dict.items()):
            for loc1, (i, i_click_time) in enumerate(item_time_list):
                item_cnt[i] += 1
                i2i_sim.setdefault(i, {})
                for loc2, (j, j_click_time) in enumerate(item_time_list):
                    if (i == j):
                        continue

                    loc_alpha = 1.0 if loc2 > loc1 else 0.7
                    loc_weight = loc_alpha * (0.9 ** (np.abs(loc2 - loc1) - 1))
                    click_time_weight = np.exp(0.7 ** np.abs(i_click_time - j_click_time))
                    created_time_weight = np.exp(0.8 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))
                    i2i_sim[i].setdefault(j, 0)
                    i2i_sim[i][j] += loc_weight * click_time_weight * created_time_weight / math.log(
                        len(item_time_list) + 1)

        i2i_sim_ = i2i_sim.copy()
        for i, related_items in i2i_sim.items():
            for j, wij in related_items.items():
                i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])

        pickle.dump(i2i_sim_, open(save_path + 'itemcf_i2i_sim.pkl', 'wb'))
        return i2i_sim_

    def item_based_recommend(self, user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num,
                             item_topk_click, item_created_time_dict, emb_i2i_sim):
        user_hist_items = user_item_time_dict[user_id]
        item_rank = {}
        for loc, (i, click_time) in enumerate(user_hist_items):
            for j, wij in sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
                if j in user_hist_items:
                    continue
                created_time_weight = np.exp(0.5 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))
                loc_weight = (0.9 ** (len(user_hist_items) - loc))
                content_weight = 1.0
                if emb_i2i_sim.get(i, {}).get(j, None) is not None:
                    content_weight += emb_i2i_sim[i][j]
                if emb_i2i_sim.get(j, {}).get(i, None) is not None:
                    content_weight += emb_i2i_sim[j][i]
                item_rank.setdefault(j, 0)
                item_rank[j] += created_time_weight * loc_weight * content_weight * wij

        if len(item_rank) < recall_item_num:
            for i, item in enumerate(item_topk_click):
                if item in item_rank:
                    continue
                item_rank[item] = - i - 100
                if len(item_rank) == recall_item_num:
                    break

        item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]
        return item_rank

    def evalute(self):
        if metric_recall:
            trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(self.all_click_df)
        else:
            trn_hist_click_df = self.all_click_df

        trn_hist_click_df = trn_hist_click_df.sort_values(['user_id', 'click_timestamp'], ascending=[True, True])
        trn_hist_click_df = trn_hist_click_df.groupby('user_id').tail(3)

        user_recall_items_dict = collections.defaultdict(dict)
        user_item_time_dict = get_user_item_time(trn_hist_click_df)

        i2i_sim = pickle.load(open(save_path + 'itemcf_i2i_sim.pkl', 'rb'))
        emb_i2i_sim = pickle.load(open(save_path + 'emb_i2i_sim.pkl', 'rb'))

        sim_item_topk = 20
        recall_item_num = 100
        item_topk_click = get_item_topk_click(trn_hist_click_df, k=100)

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
                       trn_hist_click_df['user_id'].unique()}
            for future in tqdm(futures):
                user, recommendation = future.result()
                user_recall_items_dict[user] = recommendation

        user_multi_recall_dict['itemcf_sim_itemcf_recall'] = user_recall_items_dict
        pickle.dump(user_multi_recall_dict['itemcf_sim_itemcf_recall'],
                    open(save_path + 'itemcf_recall_dict.pkl', 'wb'))

        if metric_recall:
            metrics_recall(user_multi_recall_dict['itemcf_sim_itemcf_recall'], trn_last_click_df, topk=recall_item_num)


if __name__ == '__main__':
    module = ItemCF(sample=True)
    module.itemcf_sim(module.all_click_df, module.item_created_time_dict)
    module.evalute()
