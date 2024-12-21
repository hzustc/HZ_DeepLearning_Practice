from collections import defaultdict
from math import sqrt
import pandas as pd
item_user_score_dict = defaultdict(
    item1=defaultdict(u1=1,u2=2,u3=3),
    item2=defaultdict(u1=5,u2=6,u3=8),
    item3=defaultdict(u1=5,u2=6),
)

sim_dict = defaultdict(defaultdict)
for i1 ,i1_user_score in item_user_score_dict.items():

    for i2,i2_user_score in item_user_score_dict.items():
        if i1==i2:
            continue
        sim = 0
        z_1,z_2=0,0
        for user in i1_user_score:

            sim+=i1_user_score.get(user,0)*i2_user_score.get(user,0)

            z_1 +=i1_user_score.get(user,0)**2
            z_2 +=i2_user_score.get(user,0)**2

        sim_dict[i1][i2]=sim/(sqrt(z_1)*sqrt(z_2))



sim_item = sorted(list(sim_dict['item3'].items()),key=lambda x:x[1],reverse=True)[:1]
score = 0
for item,sim in sim_item:
    score +=item_user_score_dict[item]['u3']*sim

print(score)

data = pd.DataFrame(item_user_score_dict).fillna(0)
data.corr()