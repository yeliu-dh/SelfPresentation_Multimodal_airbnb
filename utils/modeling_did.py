import os, sys, importlib, pathlib
import pandas as pd    
import numpy as np

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

    data = data.sort_values("time")
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
    
    
def get_ddd_effect_interaction(model, var):
    """
    in_paris:C(time)[T.2406] - in_paris:C(time)[T.2312]    
    """
    if var :
        term=f"in_paris:C(time)[T.2406]:{var} - in_paris:C(time)[T.2312]:{var} = 0"
    # print(term)
    
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
    
    

def plot_one_did(data, axes=None, i=0, var='authenticité', title=None): 
    if not axes:
        fig, ax=plt.subplots(figsize=(6, 4))   
    else :    
        ax=axes[i]

    
    df=data.copy()
    sub = df[df["variable"] == var].copy()
    sub = sub.sort_values("time")
    x=list(range(len(sub['time'])))

    color_did='tab:red'

    # baseline处理（2306 = 0）
    if (sub["time"] == '2306').any():
        sub.loc[sub["time"] == '2306', "coef"] = 0
        sub.loc[sub["time"] == '2306', "ci_low"] = 0
        sub.loc[sub["time"] == '2306', "ci_high"] = 0
    
    # 置信区间errorbar
    ax.errorbar(
        x,
        # sub['time'],
        # sub['label'],
        sub["coef"],
        yerr=[
            sub["coef"] - sub["ci_low"],
            sub["ci_high"] - sub["coef"]
        ],
        fmt='-o',       # 线 + 点
        capsize=3,
        label="Paris", 
        color=color_did
    )
    # + sig
    for xi, yi, sig in zip(x, sub["coef"], sub["sig"]):
        ax.text(xi+0.05, yi + 0.001, sig, ha='center', 
        color=color_did, fontsize=12)

    ## 基准线london
    hline_v=0
    ax.axhline(hline_v, linestyle="--", color="gray", linewidth=1, label='Londres')
    
    # ---legend---
    # tick
    ax.set_xticks(x)
    ax.set_xticklabels(sub['time'])
    ax.tick_params(axis="x", rotation=30)
    
    # xylabel
    ax.set_xlabel("Temps")
    ax.set_ylabel('Différence')

    # title
    if title is None:
        title=var
    ax.set_title(title)

    if not axes:    
        ax.legend()


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
    ] -= adjustment#==去掉2406交互项，再加一次2312交互项
    

    # print(df[['y_hat','y_cf']].describe())
    agg=df.groupby(['in_paris','time'])[['y_hat','y_cf']].mean().reset_index()
    agg['variable']=var
    # agg=agg.sort_values([
    
    return agg 


def plot_one_cf(agg_all, axes=None, i=0,  var="authenticité", ddd_sig="", title=None):
    if not axes:
        fig, ax=plt.subplots(figsize=(6,4))
    else :
        ax=axes[i]

    agg=agg_all[agg_all['variable']==var]
 

    ## plt
    paris = agg[agg["in_paris"] == 1]    
    # ---paris obs---
    ax.plot(
        paris["time"],#.map(time_map),
        paris["y_hat"],
        marker="o",
        label="Observé (Paris)",
        color='tab:orange'
    )
    
    
    # ---paris cf---
    ax.plot(
        paris["time"],#.map(time_map),
        paris["y_cf"],
        marker="o",
        linestyle="--",
        label="Contrefactuel(Paris)",
        color='tab:orange'
    )
    
    # ---london obs---
    control = agg[agg["in_paris"] == 0]
    ax.plot(
        control["time"],#.map(time_map),
        control["y_hat"],
        marker="o",
        # alpha=0.4,
        label="Observé (Londres)",
        color='tab:blue'
    )
    
    title=var if title==None else title
    
    ax.set_title(f"{title} {ddd_sig}")
    ax.set_ylabel("Niveau moyen")
    ax.set_xlabel("Temps")
    ax.tick_params(axis='x', rotation=30)

    if not axes:
        ax.legend(fontsize=10)
        
    return 



##======================================= predict curve===================================

def plot_predict_curve(model, df, var):
    # default control values
    default_vals = {}
    for col in model.model.exog_names:
        if ':' in col or col == 'Intercept':
            continue
        var_name = col.split('[')[0].replace('C(', '').replace(')', '')
        if var_name in df.columns:
            if df[var_name].dtype == 'O' or df[var_name].nunique() < 10:
                default_vals[var_name] = df[var_name].mode()[0]
            else:
                default_vals[var_name] = df[var_name].mean()
    tactic_range = np.linspace(df[var].min(), df[var].max(), 100)
    print(f"default_vals: {default_vals}")
    
    rows = []
    for val in tactic_range:
        row = default_vals.copy()
        row[var] = val
        rows.append(row)

    pred_df = pd.DataFrame(rows)

    # type alignment
    for col in pred_df.columns:
        if col in df.columns:
            pred_df[col] = pred_df[col].astype(df[col].dtype)

    preds = model.get_prediction(pred_df)
    sf = preds.summary_frame(alpha=0.05)

    pred_df["y"] = sf["mean"]
    pred_df["low"] = sf["mean_ci_lower"]
    pred_df["high"] = sf["mean_ci_upper"]

    return pred_df






## =================================real effet vars====================================
import numpy as np
from scipy import stats

def linear_combo(model, terms):
    coefs = model.params
    cov = model.cov_params()
    
    # 系数向量
    beta = coefs[terms].values
    
    # 权重（这里都是1）
    w = np.ones(len(terms))
    
    # effect
    effect = np.sum(beta)
    
    # variance
    sub_cov = cov.loc[terms, terms].values
    var = w @ sub_cov @ w
    
    se = np.sqrt(var)
    
    # t值
    t = effect / se
    
    # p-value（双侧）
    df = model.df_resid
    pval = 2 * (1 - stats.t.cdf(abs(t), df))
    
    # 95% CI
    crit = stats.t.ppf(0.975, df)
    ci_low = effect - crit * se
    ci_high = effect + crit * se
    
    return effect, se, pval, ci_low, ci_high



def get_real_cf_effect_data(model, tactics, ddd_df):
    results = []
    for var in tactics:
        print(var)
        specs = [
            # london:
            ("2306", 0, [var]),
            ("2312", 0, [var, f"C(time)[T.2312]:{var}"]),
            ("2406", 0, [var, f"C(time)[T.2406]:{var}"]),
            
            # paris:
            ("2306", 1, [var, f"in_paris:{var}"]),
            ("2312", 1, [
                var,
                f"C(time)[T.2312]:{var}",
                f"in_paris:{var}",
                f"in_paris:C(time)[T.2312]:{var}"
            ]),
            ("2406", 1, [
                var,
                f"C(time)[T.2406]:{var}",
                f"in_paris:{var}",
                f"in_paris:C(time)[T.2406]:{var}"
            ])        
        ]
        
        for time, paris, terms in specs:
            effect, se, pval, ci_low, ci_high = linear_combo(model, terms)    
            if time=="2406" and paris==1:
                adjustment=ddd_df[ddd_df['variable']==var]['coef'].loc[0]
                effect_cf=effect-adjustment
                results.append({
                "time": time,
                "in_paris": paris,
                "variable":var,
                "effect_hat": effect,
                "effect_cf": effect_cf,
                "se": se,
                "pval": pval,
                "sig":p_to_sig(pval),
                "ci_low": ci_low,
                "ci_high": ci_high
                })
                
            else :
                results.append({
                    "time": time,
                    "in_paris": paris,
                    "variable":var,
                    "effect_hat": effect,
                    "effect_cf": effect,
                    "se": se,
                    "pval": pval,
                    "sig":p_to_sig(pval),
                    "ci_low": ci_low,
                    "ci_high": ci_high
                })
            
    df_plot = pd.DataFrame(results)
    
    
    order = ["2306", "2312", "2406"]
    df_plot["time"] = pd.Categorical(df_plot["time"], categories=order, ordered=True)

    return df_plot

    
    
def plot_one_real_effect(df_plot, var, axes=None, i=0, title=None):   
    # check
    df=df_plot[df_plot['variable']==var].copy()
    if df_plot.empty:
        print(f"[warning] filter to 0!")
        
    if axes is None:
        fig, ax=plt.subplots(figsize=(6, 4))
    else :
        ax=axes[i]
    

    for i, (key, grp) in enumerate(df.groupby("in_paris")):
        print(key)
        # display(grp)

        # print(key)# 0/1
        grp = grp.sort_values("time")
        x = list(range(len(grp)))   # 0,1,2
            
        ax.errorbar(
            # grp["time"],
            x, 
            grp["effect_hat"],
            yerr=[
                grp["effect_hat"] - grp["ci_low"],
                grp["ci_high"] - grp["effect_hat"]
            ],
            marker='s', # 方块
            linestyle='-',
            alpha=0.85,
            label=f"{'Paris' if key==1 else 'Londres'}"
        )
        try:
            for xi, yi, sig in zip(x, grp["effect_hat"], grp["sig"]):
                ax.text(xi+0.05, yi + 0.001, sig, ha='center', 
                        color=f"{'tab:orange' if key==1 else 'tab:blue'}", fontsize=12)
                
        except Exception as e:
            print(e)    

        
    ax.axhline(0, color="lightgray", linestyle="--")  # 零效应线
    ax.set_xticks(x)
    ax.set_xticklabels(grp['time'])
    
    ax.set_xlabel("Temps")
    ax.set_ylabel(f"Effet de tactique")
    if title is None:
        title=var
    ax.set_title(f"{title}")
    ax.tick_params(axis="x",rotation=30)
    if not axes:
        ax.legend()
    # ax.get_legend().remove()#存在时才能删除！
    
    # plt.show()
    return 



    
def plot_one_real_cf_effect(df_plot, var, axes=None, i=0, title=None):   
    # ---data---
    df=df_plot[df_plot['variable']==var].copy()
    if df_plot.empty:
        print(f"[warning] filter to 0!")
        
    if axes is None:
        fig, ax=plt.subplots(figsize=(6, 4))
    else :
        ax=axes[i]
    
    ## plot
    x = list(range(df['time'].nunique()))   # 0,1,2
    # print(f"axis x: {x}\n")
    
    
    ## paris
    paris=df[df['in_paris']==1]
    color_paris='tab:orange'
    
    # ---paris obs---    
    ax.errorbar(
        x,
        # paris["time"],#.map(time_map),
        paris["effect_hat"],
        yerr=[
                paris["effect_hat"] - paris["ci_low"],
                paris["ci_high"] - paris["effect_hat"]
            ],
        marker="s",
        # alpha=0.6,

        label="Observé (Paris)",
        color=color_paris
    )
    for xi, yi, sig in zip(x, paris["effect_hat"], paris["sig"]):
        ax.text(xi+0.05, yi + 0.001, sig, ha='center', 
        color=color_paris, fontsize=12)
                    
    # ---paris cf---
    ax.errorbar(
        x,
        # paris["time"],#.map(time_map),
        paris["effect_cf"],
        marker="s",
        linestyle="--",
        label="Contrefactuel(Paris)",
        color=color_paris
    )
    for xi, yi, sig in zip(x, paris["effect_cf"], paris["sig"]):
        ax.text(xi+0.05, yi + 0.001, sig, ha='center', 
        color=color_paris, fontsize=12)
    

    # ---london obs---
    
    control = df[df["in_paris"] == 0]
    color_london="tab:blue"
    
    ax.errorbar(
        x,
        # control["time"],#.map(time_map),
        control["effect_hat"],
        yerr=[
                control["effect_hat"] - control["ci_low"],
                control["ci_high"] - control["effect_hat"]
            ],
        marker="s",
        alpha=0.6,
        label="Observé (Londres)",
        color=color_london
    )
    for xi, yi, sig in zip(x, control["effect_hat"], control["sig"]):
            ax.text(xi+0.08, yi-0.001, sig, ha='center', 
            color=color_london, fontsize=12)                         
    
        
    ax.axhline(0, color="lightgray", linestyle="--")  # 零效应线
    ax.set_xticks(x)# x
    ax.set_xticklabels(df['time'].unique())
    
    ax.set_xlabel("Temps")
    ax.set_ylabel(f"Effet de tactique")
    if title is None:
        title=var
    ax.set_title(f"{title}")
    ax.tick_params(axis="x",rotation=30)
    if not axes:
        ax.legend()
    # ax.get_legend().remove()#存在时才能删除！
    
    # plt.show()
    return 



def plot_one_did_interaction(df_plot, var='authenticité', axes=None, i=0, title=None):
    if not axes:
        fig, ax=plt.subplots(figsize=(6, 4))   
    else :    
        ax=axes[i]

    # ---dat---
    sub = df_plot[df_plot["variable"] == var].copy()
    sub = sub.sort_values("time")   
    # display(sub)
    
    color_did="tab:red"
    
    # ---plot---
    x=list(range(len(sub['time'])))
    print('axis x:', x)
    
    ax.errorbar(
        x,
        sub["effect"],
        yerr=[
            sub["effect"] - sub["ci_low"],
            sub["ci_high"] - sub["effect"]
        ],
        fmt='-s',       # 线 + 点
        capsize=3,
        color=color_did,
        label="Différence Paris-Londres"
    )
    # + sig
    for xi, yi, sig in zip(x, sub["effect"], sub["sig"]):
        ax.text(xi+0.05, yi + 0.001, sig, ha='center', 
        color=color_did, fontsize=12)
    
    ## ref
    hline_v=sub[sub['time']=="2306"]['effect'].iloc[0]
    ax.axhline(hline_v, linestyle="--", color="tab:blue", linewidth=1, label='Londres')
    hline_0=0
    ax.axhline(hline_0, linestyle="--", color="gray", linewidth=1)

    # ---legend---
    ax.set_xticks(x)
    ax.set_xticklabels(sub['time'])
    ax.tick_params(axis="x", rotation=30)
    
    ax.set_xlabel("Temps")
    ax.set_ylabel("Différence Paris-Londres")
    if title is None:
        title=var
    ax.set_title(f"{title}")
    if not axes:
        ax.legend()
    
    
    
    

# importlib.reload(modeling_did)
# from utils.modeling_did import get_event_study_data, get_ddd_effect, run_tactic_model_by_time, plot_one


# importlib.reload(modeling_did)
# from utils.modeling_did import get_event_study_data, get_ddd_effect, run_tactic_model_by_time, plot_one

