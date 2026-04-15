import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats


# ================================================================================================
# # ---colors----
# blue_cmap = plt.cm.Blues
# orange_cmap = plt.cm.Oranges
# n = 2 # 每组柱子的数量
# london_colors = blue_cmap(np.linspace(0.3, 0.8, n))
# paris_colors  = orange_cmap(np.linspace(0.3, 0.8, n))

# colors = (
#     list(london_colors) +
#     list(paris_colors)
# )


# ## ---layout 2facets----
# import matplotlib.gridspec as gridspec
# fig = plt.figure(figsize=(14, 5)) # sharey=True
# gs = gridspec.GridSpec(1, 2, figure=fig, hspace=0.5, wspace=0.3)# 1行2列

# axes = []
# # 第一行两个：分别占两列宽 (0:2, 2:4, 4:6)
# ax1 = fig.add_subplot(gs[0, 0:1])
# ax2 = fig.add_subplot(gs[0, 1:2], sharey=ax1)#**
# axes.extend([ax1, ax2])


# ## ---legend 2facets----
# fig.suptitle(
#         # "Effets marginaux des tactiques visuelles entre les groupes",
#         "Comparaison des tactiques textuelles entre les noueaux et anciens hôtes en Juin 2024",
#         fontsize=16,
#         y=1.01
#     )

# plt.figtext(
#     0.5, 0.01,
#     "Significativité de la différence : *** p<0.001, ** p<0.01, * p<0.05, . p<0.1",
#     ha="center",
#     fontsize=12,
#     style="italic",   
# )
# plt.tight_layout(rect=[0, 0.1, 0.8, 1])# [left, bottom, right, top]
# plt.subplots_adjust(bottom=0.3)#控制子图到底的距离，给text留出空间
# plt.show()



# ===================================================================================================


tactics_bio = [
    "ouverture",
    "authenticité",
    "sociabilité",
    "auto_promotion",
    "exemplarité",
]

tactics_pic=['host_picture_type']# no_person, pro_style, life_style



def p_to_sig(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    elif p < 0.1: return '.'
    else: return ''
    
def ttest_new_old(df, times: list=["2312"], in_paris=[0, 1], vars=tactics_bio):
    # check time & in_paris
    df_t=df.copy()
        
    df_t = df_t[df_t['time'].isin(times)]
    df_t = df_t[df_t['in_paris'].isin(in_paris)]

    print(df.shape, '=>', df_t.shape)
    
    groups_in_paris=df_t['in_paris'].unique()
    groups_time=df_t['time'].unique()
    # print(groups_in_paris)
    
    rows=[]
    for time in groups_time:    
        for p in groups_in_paris:        
            for var in vars:
                sub = df_t[(df_t['time'] == time) & (df_t['in_paris'] == p)]# impo!!
                new = sub[sub['is_new_host'] == 1][var].dropna()
                old = sub[sub['is_new_host'] == 0][var].dropna()

                t_stat, p_value = stats.ttest_ind(new, old, equal_var=False)
                sig=p_to_sig(p_value)
                
                row_new= pd.DataFrame({
                    "time": [time],
                    "in_paris":[p],
                    "variable":var,
                    "t_stat": [t_stat],
                    "p_value": [p_value],
                    "sig":[sig],
                    'is_new_host':1,
                    'mean':[new.mean()],
                    'label':f"{var} {sig}"
                })
                
                row_old= pd.DataFrame({
                    "time": [time],
                    "in_paris":[p],
                    "variable":var,
                    "t_stat": [t_stat],
                    "p_value": [p_value],
                    "sig":[sig],
                    'is_new_host':0,
                    'mean':[old.mean()],
                    'label':f"{var} {sig}"
                })
                
                rows.append(row_new)
                rows.append(row_old)
    return pd.concat(rows, axis=0)




def ttest_superhost(df, times: list=["2012"], in_paris=[0, 1], vars=tactics_bio):
    # check time & in_paris
    df_t=df.copy()
    df_t['host_is_superhost']=df_t['host_is_superhost'].apply(lambda x : 1 if x=="t" else 0)
    df_t = df_t[df_t['time'].isin(times)]
    df_t = df_t[df_t['in_paris'].isin(in_paris)]

    print(df.shape, '=>', df_t.shape)
    
    groups_in_paris=df_t['in_paris'].unique()
    groups_time=df_t['time'].unique()
    # print(groups_in_paris)
    
    rows=[]
    for time in groups_time:    
        for p in groups_in_paris:        
            for var in vars:
                sub = df_t[(df_t['time'] == time) & (df_t['in_paris'] == p)]                
                super = sub[sub['host_is_superhost'] == 1][var].dropna()                
                host = sub[sub['host_is_superhost'] == 0][var].dropna()
                t_stat, p_value = stats.ttest_ind(super, host, equal_var=False)
                sig=p_to_sig(p_value)
                
                row_super= pd.DataFrame({
                    "time": [time],
                    "in_paris":[p],
                    "variable":var,
                    "t_stat": [t_stat],
                    "p_value": [p_value],
                    "sig":[sig],
                    'host_is_superhost':1,
                    'mean':[super.mean()],
                    'label':f"{var} {sig}"
                })
                
                row_host= pd.DataFrame({
                    "time": [time],
                    "in_paris":[p],
                    "variable":var,
                    "t_stat": [t_stat],
                    "p_value": [p_value],
                    "sig":[sig],
                    'host_is_superhost':0,
                    'mean':[host.mean()],
                    'label':f"{var} {sig}"
                })
                rows.append(row_super)
                rows.append(row_host)
    return pd.concat(rows, axis=0)


# 2312 vs 2406
def ttest_by_time(df, times: list=["2012"], in_paris=[0, 1], is_new_host=[1], vars=tactics_bio):
    # ---data---
    df_t=df.copy()        
    df_t = df_t[df_t['time'].isin(times)]
    df_t = df_t[df_t['in_paris'].isin(in_paris)]
    df_t = df_t[df_t['is_new_host'].isin(is_new_host)]
    print(df.shape, '=>', df_t.shape)
    print(df_t[['is_new_host','time', "in_paris"]].value_counts(dropna=False))
    groups_in_paris=df_t['in_paris'].unique()
    groups_new_host=df_t['is_new_host'].unique()
    # print(groups_in_paris)
    
    rows=[]
    for new in groups_new_host: # 统一
        for p in groups_in_paris: #分图 
            for var in vars:
                sub = df_t[(df_t['is_new_host'] == new) & (df_t['in_paris'] == p)]                
                # hue
                early = sub[sub['time'] == "2312"][var].dropna()
                late = sub[sub['time'] == "2406"][var].dropna()
                
                t_stat, p_value = stats.ttest_ind(early, late, equal_var=False)
                sig=p_to_sig(p_value)
                change=late.mean()-early.mean()
                            
                row_early= pd.DataFrame({
                    "time": ["2312"],
                    "in_paris":[p],
                    "variable":var,
                    "t_stat": [t_stat],
                    "p_value": [p_value],
                    "sig":[sig],
                    'is_new_host':new,
                    'mean':[early.mean()],
                    "change":"+" if change > 0 else "-"
                    # 'label':f"{var} {sig}"

                })
                row_late= pd.DataFrame({
                    "time": ["2406"],
                    "in_paris":[p],
                    "variable":var,
                    "t_stat": [t_stat],
                    "p_value": [p_value],
                    "sig":[sig],
                    'is_new_host':new,
                    'mean':[late.mean()],
                    "change":"+" if change > 0 else "-"
                    # 'label':f"{var} {sig}"
                })
                rows.append(row_early)
                rows.append(row_late)
    return pd.concat(rows, axis=0)