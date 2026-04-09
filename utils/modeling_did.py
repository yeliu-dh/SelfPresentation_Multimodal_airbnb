import os, sys, importlib, pathlib
import pandas as pd
from pathlib import Path
sys.path.append (os.path.abspath(".."))

# modeling
import statsmodels.api as sm
import statsmodels.formula.api as smf

# plot
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec


def p_to_sig(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    elif p < 0.1: return '.'
    else: return ''
    
    
    
# ===========================TACTIC MODEL==================================

def run_tactic_model_by_time(df, y):
    """
    authenticité ~ controls + tactics + C(time) + in_paris * C(time)
    
    """
    formula= (f"{y} ~ C(host_identity_verified)  + review_scores_rating + has_rating +"
            "number_of_reviews_ltm + "
            "C(host_is_superhost) +"
            "years_since_host + professional_host + "
            "host_response_rate +  C(host_response_time)+"
            "has_text + lang_en + lang_fr + text_length+"
            "price  + availability_90 + C(room_type) + C(instant_bookable) + "
            "C(time) + in_paris * C(time)"
    )
    model_tactic=smf.ols(formula, data=df).fit()
    return model_tactic




def get_event_study_data(model, var):
    params = model.params
    bse = model.bse
    pvalues=model.pvalues
    
    # 提取 interaction terms
    data = pd.DataFrame({
        "coef": params,
        "se": bse,
        'pval':pvalues
    }).reset_index()

    data.columns = ["variable", "coef", "se", "pval"]

    # 只保留 in_paris × time
    data = data[data["variable"].str.contains("in_paris:C\\(time\\)")]
    
    # 提取时间点
    data["time"] = data["variable"].str.extract(r"T\.(\d+)")

    # 排序
    data = data.sort_values("time")

    # 置信区间
    data["ci_low"] = data["coef"] - 1.96 * data["se"]
    data["ci_high"] = data["coef"] + 1.96 * data["se"]
    
    # baseline（2306 = 0）
    baseline = pd.DataFrame({
        "time": ["2306"],
        "coef": [0],
        "ci_low": [0],
        "ci_high": [0]
    })

    data = pd.concat([baseline, data], ignore_index=True)

    # 转 numeric
    # data["time"] = pd.to_numeric(data["time"])
    data = data.sort_values("time")
    # data['time']=data['time'].astype(str)
    # display(df)
    data['variable']=var
    data['sig']=data['pval'].apply(p_to_sig)        
    data['label']=data['time']+' '+data['sig']        
    
    return data



def get_ddd_effect(model, var, 
                term="in_paris:C(time)[T.2406] - in_paris:C(time)[T.2312] = 0"):
    """
    in_paris:C(time)[T.2406] - in_paris:C(time)[T.2312]    
    """
    print(var)
    test=model.t_test(term)
    return pd.DataFrame({
            'variable':var,
            "coef": test.effect[0],
            "se": test.sd[0],
            "pval": test.pvalue,
            "sig":p_to_sig(test.pvalue),
            "ci_low": test.conf_int()[0][0],
            "ci_high": test.conf_int()[0][1]
        })
    
    

def plot_one_did(data, axes=None, i=0, var='authenticité'): 
    if not axes:
        fig, ax=plt.subplots(figsize=(6, 4))   
    else :    
        ax=axes[i]

    
    df=data.copy()
    sub = df[df["variable"] == var].copy()

    # 确保排序
    sub = sub.sort_values("time")

    # baseline处理（2306 = 0）
    if (sub["time"] == '2306').any():
        sub.loc[sub["time"] == '2306', "coef"] = 0
        sub.loc[sub["time"] == '2306', "ci_low"] = 0
        sub.loc[sub["time"] == '2306', "ci_high"] = 0
    
    # 置信区间errorbar
    ax.errorbar(
        # sub['time'],
        sub['label'],
        sub["coef"],
        yerr=[
            sub["coef"] - sub["ci_low"],
            sub["ci_high"] - sub["coef"]
        ],
        fmt='-o',       # 👈 线 + 点
        capsize=3,
    )

    ## 基准线london
    hline_v=0
    xmax=ax.get_xlim()[1]

    ax.axhline(hline_v, linestyle="--", color="gray", linewidth=1)
    # 在水平线右侧加文字
    ax.text(
        x=xmax-0.05,#（放在最右边）x 位置（根据你的数据范围调整）
        y=hline_v+0.001,            # 跟线同一个高度
        s="london",
        va='bottom',          # 垂直对齐
        ha='right',           # 水平对齐
        color="gray", 
        fontsize=10
    )

    ax.set_title(var)
    ax.set_xlabel("Temps")
    ax.set_ylabel('Différence')
    ax.tick_params(axis='x', rotation=30)
       
       
    
# importlib.reload(modeling_did)
# from utils.modeling_did import get_event_study_data, get_ddd_effect, run_tactic_model_by_time, plot_one
# colors = sns.color_palette("tab20", 6)



### =======================COUNTERFACTUAL=========================

def get_cf_data(df, model, var):
    """
    不改变时间，不改变地点，仅减去jo（交互项之差）
    """
    
    df["y_hat"] = model.predict(df)

    delta_2406 = model.params["in_paris:C(time)[T.2406]"]
    delta_2312 = model.params["in_paris:C(time)[T.2312]"]

    adjustment = delta_2406 - delta_2312
    print(f'ddd of {var}: {adjustment}')
    df["y_cf"] = df["y_hat"]

    df.loc[
        (df["in_paris"]==1) & (df["time"]=="2406"),
        "y_cf"
    ] -= adjustment

    # print(df[['y_hat','y_cf']].describe())
    agg=df.groupby(['time','in_paris'])[['y_hat','y_cf']].mean().reset_index()
    agg['variable']=var
    
    
    return agg 


def plot_one_cf(agg_all, axes=None, i=0,  var="authenticité", ddd_sig=""):
    if not axes:
        fig, ax=plt.subplots(figsize=(6,4))
    else :
        ax=axes[i]

    agg=agg_all[agg_all['variable']==var]
 
    
    # # categorical mapping
    # times = sorted(agg["time"].unique())
    # time_map = {t: i for i, t in enumerate(times)}
    
    ## plt
    paris = agg[agg["in_paris"] == 1]    
    # ---paris obs---
    ax.plot(
        paris["time"],#.map(time_map),
        paris["y_hat"],
        marker="o",
        label="Observed (Paris)",
            color='tab:orange'
    )
    
    
    # ---paris cf---
    ax.plot(
        paris["time"],#.map(time_map),
        paris["y_cf"],
        marker="o",
        linestyle="--",
        label="Counterfactual  (Paris)",
        color='tab:orange'
    )
    
    # ---london obs---
    control = agg[agg["in_paris"] == 0]
    ax.plot(
        control["time"],#.map(time_map),
        control["y_hat"],
        marker="o",
        # alpha=0.4,
        label="Observed (London)",
        color='tab:blue'
    )

    # event line
    # ax.set_xticks(list(time_map.values()))
    # ax.set_xticklabels(list(time_map.keys()))

    ax.set_title(f"{var} {ddd_sig}")
    ax.set_ylabel("Niveau de tactique")
    ax.set_xlabel("Temps")

    return





# importlib.reload(modeling_did)
# from utils.modeling_did import get_event_study_data, get_ddd_effect, run_tactic_model_by_time, plot_one

