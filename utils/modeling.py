## vif + model+ plot + latex

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# vif
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices


# ols mod
import statsmodels.api as sm
import statsmodels.formula.api as smf

# my utils
from utils.io import save_csv_as_latex
from utils.preprocess_listings import desc_catORnum


def write_formula(df, x_vars, y_var, key_vars=None, group_col=None):
    # y not in x:
    if y_var in x_vars:
        x_vars.remove(y_var)
    if group_col !=None and group_col in x_vars:
        x_vars.remove(group_col)
        
    # cat/num
    cat_vars = [var for var in x_vars if not pd.api.types.is_numeric_dtype(df[var])]
    num_vars = [var for var in x_vars if pd.api.types.is_numeric_dtype(df[var])]
    # print(f"[INFO] {len(cat_vars)} categorial vars: {cat_vars};\n"
    #         f"{len(num_vars)} numeric vars :{num_vars}\n")

    cat_vars_str = ' + '.join([f"C({c})" for c in cat_vars])
    num_vars_str = ' + '.join(num_vars)

    formula = f"{y_var} ~ {cat_vars_str.strip()} + {num_vars_str.strip()}"

    # init    
    key_vars_str=" "
    if key_vars != None:
        key_vars_str=' + '.join(key_vars)#避免无tactics输入
    
    if group_col !=None and key_vars_str.strip():
        # print(f'Interaction item : {group_col}!')
        # print(f"BEFOR:{key_vars_str}")
        if not pd.api.types.is_numeric_dtype(df[group_col]):# group_col cat
            key_vars_str=f"C({group_col}) * ({key_vars_str})"
        else : 
            key_vars_str=f"{group_col} * ({key_vars_str})"
        # print(f"AFTER:{key_vars_str}\n")

    if key_vars_str.strip():#如果key_vars_str exists:
        formula +=f"+ {key_vars_str.strip()}"
        # f"({y_var} ~ {cat_vars_str.strip()} + {num_vars_str.strip()} + {key_vars_str.strip()})"
    
    print(f"[INFO] formula :\n {formula}\n")
    
    return formula 



def check_vif (df, x_vars, y_var, key_vars, group_col):

    formula=write_formula(df=df, x_vars=x_vars, y_var=y_var, key_vars=key_vars, group_col=group_col)
        
    try :
        y, X = dmatrices(formula, data=df, return_type='dataframe')
    except Exception as e :
        print(f"[ERROR] in check_vif : \n{e}")
        
    vif_df = pd.DataFrame()
    vif_df['Variables']=X.columns
    vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_df['Niveau_colinearite'] = vif_df["VIF"].apply(
        lambda x: " " if x <= 1 else
                "*" if x <= 5 else
                "**" if x <= 10 else
                "***"
    )
    display(vif_df)  
    
    return



# def save_summary_as_latex(summary, output_folder, tex_filename):
#     latex_str =summary.as_latex()
#     outpath_summary=os.path.join(output_folder, 'latex', tex_filename)

#     with open(outpath_summary, "w") as f:
#         f.write(latex_str)
#         print(f"[SAVE] model summary saved to {outpath_summary}!")
#     return 


def p_to_sig(p):#!= get_term_sig
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''


def get_group_effect(df, model, key_var, group_col):
    if pd.api.types.is_numeric_dtype(df[group_col]):# 数值型
        interaction_term = f"{group_col}:{key_var}"
    else :
        interaction_term = f"C({group_col})[T.t]:{key_var}"

    
    
    # 构造一个 系数选择向量 c, 所有terms归零
    lin_comb = np.zeros(len(model.params))
    
    # 标记tac主效应和交互项的位置：检验1 × β_main + 1 × β_interaction
    lin_comb[model.params.index.get_loc(key_var)] = 1
    lin_comb[model.params.index.get_loc(interaction_term)] = 1

    # t 检验
    t_res = model.t_test(lin_comb)
    print(f"{interaction_term}\n" 
          f"coef : {t_res.effect}; pval :{t_res.pvalue:.3f} \n"
        #   f"{t_res}\n"
        )

    return t_res




def build_model (df_input, x_vars, key_vars=None, to_fillna0=False,
                y_var='booking_rate_l30d', group_col=None, 
                
                # outpath_folder='mod_results', 
                # tex_filename='ols_summary.tex', 
                # save=False
                ):
    df=df_input.copy()
    # os.makedirs(outpath_folder, exist_ok=True)
    
    if group_col!=None and group_col in x_vars:
        x_vars.remove(group_col)    
        print(f"[CHECK] gruop col {group_col} removed from x_vars!\n")

    # collect all vars :
    all_vars=x_vars+[y_var]
    if key_vars!=None:
        all_vars+=key_vars    
    print(f"[CHECK] isna False in all {len(all_vars)} vars : {df[all_vars].isna().value_counts(dropna=False)}\n")
    
    
    if to_fillna0==True and key_vars!=None:
        df[key_vars]=df[key_vars].fillna(0)
        
    formula=write_formula(df=df, x_vars=x_vars, y_var=y_var, key_vars=key_vars, group_col=group_col)
    
    model=smf.ols(formula, data=df).fit()
    summary=model.summary()
    print(summary)
    
    # if save==True:
    #     save_summary_as_latex(summary, output_folder=outpath_folder, 
    #                       tex_filename=tex_filename)
    
    if group_col:
        for k_var in key_vars:
            t_res=get_group_effect(df, model, key_var=k_var, group_col=group_col)            
            
    return df, formula, model




from utils.io import save_csv_as_latex


#========================================MODELS========================================


def make_one_models_table(models_dict, params):
    # 创建空表
    df_table = pd.DataFrame(index=params)

    for name, model in models_dict.items():
        coefs = model.params
        ses   = model.bse
        pvals = model.pvalues
        
        col_vals = []
        for var in params:
            if var in coefs.index:
                # 加星号
                stars = ""
                if pvals[var] < 0.01:
                    stars = "***"
                elif pvals[var] < 0.05:
                    stars = "**"
                elif pvals[var] < 0.10:
                    stars = "*"

                coef_str = f"{coefs[var]:.4f}{stars}"
                se_str   = f"({ses[var]:.4f})"
                
                col_vals.append(coef_str + "\n" + se_str)
            else:
                col_vals.append("")
        
        df_table[name] = col_vals

    # --- 添加整体指标 ---
    overall_stats = ["R²", "Adj. R²", "N"]
    df_overall = pd.DataFrame(index=overall_stats)

    for name, model in models_dict.items():
        r2 = f"{model.rsquared:.3f}"
        adj_r2 = f"{model.rsquared_adj:.3f}"
        nobs = f"{int(model.nobs)}"
        df_overall[name] = [r2, adj_r2, nobs]

    # 拼接
    df_table = pd.concat([df_table, df_overall])
    return df_table


def make_models_table(models_dict, vars_kp, ndigits=None, 
                     save=False, output_folder=None, filename_noext=None):
    
    """
    vars_kp: key_vars and groupcol
    => params_kp, params_ctrl
    
    models_dict: {"Model1": model1, "Model2": model2, ...}
    
    """
    
    last_key = list(models_dict.keys())[-1]
    model_interaction=models_dict[last_key]#去最后一个覆盖所有交叉项 
    
    # extract params
    params_kp = model_interaction.params.index[
            model_interaction.params.index.to_series().apply(
                lambda x: any(v in x for v in vars_kp)
            )
        ]
    params_ctrl=[v for v in model_interaction.params.index if v not in params_kp and v!="Intercept"]
    # print(f"params kp :{params_kp}")
    # print(f"params_ctrl:{params_ctrl}\n")
    
    
    # interate models, get summary table:
    table_kp= make_one_models_table(models_dict, params_kp)
    table_ctrl= make_one_models_table(models_dict, params_ctrl)
    display(table_kp)
    display(table_ctrl)
    
    if save and output_folder:
        os.makedirs(output_folder, exist_ok=True)
        if filename_noext==None:
            filename_kp='table_mods_keyvars.tex'
            filename_ctrl='table_mods_ctrlvars.tex'
        else : 
            filename_kp=filename_noext+'_keyvars.tex'
            filename_ctrl=filename_noext+'_ctrlvars.tex'
        
        output_path_kp=os.path.join(output_folder,'latex', filename_kp)
        output_path_ctrl=os.path.join(output_folder, 'latex',filename_ctrl)
        
        save_csv_as_latex(table_kp, 
                        output_path=output_path_kp,
                        caption="Résultat des modèles de régression OLS",
                        label="tab:table_mods_keyvars",
                        escape=False, #* 
                        index=True,
                        ndigits=ndigits)
        
        save_csv_as_latex(table_ctrl, 
                        output_path=output_path_ctrl,
                        caption="Variables de contrôles dans les modèles de régression OLS",
                        label="tab:table_mods_keyvars", 
                        escape=False, #* 
                        index=True,
                        ndigits=ndigits)
        

    return table_kp,table_ctrl











#=========================================PLOT================================================


def get_term_sig(model, term):
    pval = model.pvalues.get(term, None)
    sig = ""
    if pval is not None:
        if pval < 0.001:
            sig = '***'
        elif pval < 0.01:
            sig = '**'
        elif pval < 0.05:
            sig = '*'
    return sig




def plot_key_var(df_input, x_vars, y_var, tactics_vars, 
                     tactic_fr, group_col=None, 
                     ax=None, show=True):
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import statsmodels.formula.api as smf
    import re
    
    """
    直接输入formula保证变量一致，因为df、tactics、formula的名字改变，直接重新建模
    可以改变group_col，但label中直接显示该col的类别，只有host_is_superhost重写成Superhôtes/Autres！
    
    """
    df=df_input.copy()
    df[tactics_vars]=df[tactics_vars].fillna(0)
    formula=write_formula(df=df, x_vars=x_vars, y_var=y_var, key_vars=tactics_vars, group_col=group_col)

    #-----------------------处理变量名！（无法识别accent）-----------------------    
    # rename_tactics_map={"ouverture_pondéré":"openness",
    #                     "authenticité_pondéré":"authenticity",
    #                     "sociabilité_pondéré":"sociability",
    #                     "auto_promotion_pondéré":"self_promotion",
    #                     "exemplarité_pondéré": "exemplification"}
    
    rename_tactics_map={"ouverture":"openness",
                        "authenticité":"authenticity",
                        "sociabilité":"sociability",
                        "auto_promotion":"self_promotion",
                        "exemplarité": "exemplification"}

    # new df : 
    df.rename(columns=rename_tactics_map, inplace=True)
    
    # new tactic 
    tactic=rename_tactics_map[tactic_fr]
    if tactic not in df.columns :
        print(f"[WARNING] {tactic} NOT in df!!!")

    # new formula :
    for var, new_var in rename_tactics_map.items():
        formula=formula.replace(var, new_var)
            
    # new model :
    model=smf.ols(formula, data=df).fit()
    

    #------------------ 取控制变量的默认值--------------------
    default_vals = {}
    for col in model.model.exog_names:# == terms==model.params.index
        if ':' in col or col == 'Intercept':#不考虑交互项和截距
            continue
        var_name = col.split('[')[0].replace('C(', '').replace(')', '')#分类变量去掉外面的括号
        if var_name in df.columns:
            if df[var_name].dtype == 'O' or df[var_name].nunique() < 10:
                default_vals[var_name] = df[var_name].mode()[0]
            else:
                default_vals[var_name] = df[var_name].mean()
    vars_in_formula = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', formula)
    default_vals = {k: v for k, v in default_vals.items() if k in vars_in_formula}
    # 相当于取出所有的terms

    #----------------------预测线------------------------ 
    tactic_range = np.linspace(df[tactic].min(), df[tactic].max(), 100)
    # 取该tac的最大最小值，切割成100份
    
    rows = []
    # 情况 A：有分组变量（如 superhost）
    if group_col is not None:
        groups = df[group_col].unique()

        for g in groups:
            for val in tactic_range:
                row = default_vals.copy()
                row[tactic] = val
                row[group_col] = g
                rows.append(row)

    # 情况 B：无分组变量，只画一条线
    else:
        for val in tactic_range:
            row = default_vals.copy()
            row[tactic] = val # 对于考察的var覆盖默认值
            rows.append(row)

    predict_df = pd.DataFrame(rows)
    # display(predict_df)
    
    # 类型对齐（非常重要）why?
    for col in predict_df.columns:
        if col in df.columns:
            predict_df[col] = predict_df[col].astype(df[col].dtype)

    # 模型预测，从model已知y==booking
    predict_df['predicted_booking_rate'] = model.predict(predict_df)
    preds = model.get_prediction(predict_df)
    pred_summary = preds.summary_frame(alpha=0.05)

    predict_df['ci_lower'] = pred_summary['mean_ci_lower']
    predict_df['ci_upper'] = pred_summary['mean_ci_upper']
    # display(predict_df)
    
    
    #----------------------plot-----------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    # 情况 A：有分组，多条线
    if group_col is not None:
        # for g in groups:
        #     group_df = predict_df[predict_df[group_col] == g]
        #     label = str(g)
            
        #     if group_col=="host_is_superhost":
        #         label= f'Superhôtes' if g == "t" else f'Autres'      
        
            # ax.plot(group_df[tactic], group_df['predicted_booking_rate'],
            #         label=label)
            # ax.fill_between(group_df[tactic],
            #                 group_df['ci_lower'], group_df['ci_upper'],
            #                 alpha=0.2)

        
        
        for g in groups:
            group_df = predict_df[predict_df[group_col] == g]
            
            # 取对应的显著性
            if g == "t":
                # term = f"C({group_col})[T.t]:{tactic}" # 直接取了交互项的sig，但是t组应该是主效应+交互项
                t_res=get_group_effect(df=df, model=model, key_var=tactic, group_col=group_col)         
                sig=p_to_sig(t_res.pvalue)
                
            else:
                term = tactic  # 或者 main effect
                sig = get_term_sig(model, term)

            if group_col=="host_is_superhost":
                label= f'Superhôtes' if g == "t" else f'Autres'
        
            label = f"{label} {sig}" if sig else label

            ax.plot(group_df[tactic], group_df['predicted_booking_rate'],
                    label=label)
            ax.fill_between(group_df[tactic],
                            group_df['ci_lower'], group_df['ci_upper'],
                            alpha=0.2)

        ax.legend()


    # 情况 B：无分组，单条线
    else:
        ax.plot(predict_df[tactic], predict_df['predicted_booking_rate'], linewidth=2)
        ax.fill_between(
            predict_df[tactic],
            predict_df['ci_lower'],
            predict_df['ci_upper'],
            alpha=0.2
        )
        
    
    #------------------- sig-----------------------
    if group_col!=None:
        term = f"C({group_col})[T.t]:{tactic}" 
    else : 
        term = tactic
    sig=get_term_sig(model, term)

    # tactic_fr_map={"ouverture_pondéré":"ouverture",
    #         "authenticité_pondéré":"authenticité",
    #         "sociabilité_pondéré":"sociabilité",
    #         "auto_promotion_pondéré":"auto_promotion",
    #         "exemplarité_pondéré": "exemplarité"}    

    # ax.set_xlabel(f"{tactic_fr_map.get(tactic_fr,None)}")
    ax.set_xlabel(f"{tactic_fr}")
    ax.set_ylabel('Taux de réservation prédit')
    
    # if group_col:
    #     if group_col=="host_is_superhost":
    #         title=f"{tactic_fr_map.get(tactic_fr,None)} × Superhôte"# {sig}
    #     else : #其他分组变量
    #         title=f"{tactic_fr_map.get(tactic_fr,None)} × {group_col}"# {sig}
    # else :#无分组变量
    #     title=f"Tactique {tactic_fr_map.get(tactic_fr,None)} {sig}"
        
    if group_col:
        if group_col=="host_is_superhost":
            title=f"{tactic_fr} × Superhôte"# {sig}
        else : #其他分组变量
            title=f"{tactic_fr} × {group_col}"# {sig}
    else :#无分组变量
        title=f"Tactique {tactic_fr} {sig}"
        
    ax.set_title(title)
    ax.legend()
    
    if show and ax is None:
        plt.show()

    return fig, ax





def layout_plots(df_input, x_vars, y_var, tactics_vars, 
                 group_col=None, save=False,
                 output_folder=None, filename=None):
    import os
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    
    # --------------------layout-------------------
    # 使用更细的列分辨率（2 行 × 6 列），方便把第二行两个图放在中间列
    fig = plt.figure(figsize=(20,9))
    # gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.3, wspace=0.5)
    gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.4, wspace=1)
    
    axes = []
    
    # 第一行三个：分别占两列宽 (0:2, 2:4, 4:6)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[0, 4:6])
   
    axes.extend([ax1, ax2, ax3])

    # 第二行两个：让它们居中，分别占中间的两列 (1:3 和 3:5)
    ax4 = fig.add_subplot(gs[1, 1:3])
    ax5 = fig.add_subplot(gs[1, 3:5])
    # ax4 = fig.add_subplot(gs[1, 0:1])  # 左图
    # ax5 = fig.add_subplot(gs[1, 2:3])  # 右图
    axes.extend([ax4, ax5])

    # 确保 tactics_vars 长度为 5（或小于等于 len(axes)）
    n_to_plot = min(len(tactics_vars), len(axes))

    #-------------------plot----------------------------
    # 注意：不要在这里重新定义 ax（不要写 ax = axes[i]）
    for ax, tactic_fr in zip(axes[:n_to_plot], tactics_vars[:n_to_plot]):
        # 传入 ax，且在这里不显示单图（plot_interaction 内 show=False）
        plot_key_var(
            df_input=df_input,
            x_vars=x_vars,
            y_var=y_var,
            tactics_vars=tactics_vars,
            tactic_fr=tactic_fr,
            group_col=group_col,
            ax=ax,
            show=False
        )
    
    # 如果 tactics < 5，删除多余 axes（可选）
    if len(tactics_vars) < len(axes):
        for extra_ax in axes[len(tactics_vars):]:
            fig.delaxes(extra_ax)
    # plt.tight_layout()
    
    #set suptitle
    if group_col:
        if group_col=="host_is_superhost":
            suptitle=f"Figures d'interation entre les tactiques et le statut de Superhôte"
        else : #其他分组变量
            suptitle=f"Figures d'interation entre les tactiques et {group_col}"
    
    else :#无分组变量
        # suptitle=f"Tactique {tactic_fr_map.get(tactic_fr,None)} {sig}"    
        suptitle=""
        
    fig.suptitle(
        suptitle,
        fontsize=18,
        y=0.98
    )

    # save
    os.makedirs(output_folder, exist_ok=True)
    if filename is None:
        if group_col!=None:
            filename = f"{y_var}_tacticsX{group_col}_plots.jpg"
        else : 
            filename=f"{y_var}_tactics_plots.jpg"  
              
    outpath_plots = os.path.join(output_folder, filename)
    if save:    
        plt.savefig(outpath_plots, dpi=300, bbox_inches='tight')
    
        print(f"[SAVE] plots saved to {outpath_plots}!")
    
    # show
    plt.show()
    
    return fig, axes

##===========================================MOD===============================================
"""
## exemple des vars baseline :

y_var="booking_rate_l30d"
x_vars=["host_identity_verified", "host_has_profile_pic",         
    "review_scores_rating", #"has_rating", "number_of_reviews",
    "years_since_host","professional_host",'host_is_superhost', #'calculated_host_listings_count',
    "lang", "len",
    "price", "availability_30", "room_type", "instant_bookable"
]  
tactics_vars=["ouverture", "authenticité","sociabilité","auto_promotion","exemplarité"]
group_col="host_is_superhost"
output_folder='../mod_results'

#加不加has_rating, count差别不大！
# calculated_host_listings_count
# "property_type"
 

"""
def get_booking_rate_l30d(df):
    if "number_of_reviews_l30d" not in df.columns or "availability_30" not in df.columns:
        print(f"[WARNING]number_of_reviews_l30d or availability_30 not in df!!!\n")
    else :        
        df['booking_rate_l30d'] = df.apply(
                lambda row: min(row['number_of_reviews_l30d'] / row['availability_30'], 1.0)
                if row['availability_30'] > 0 else None,
                axis=1
            )
    desc_catORnum(df, vars=["booking_rate_l30d"])

    return 

    
    
    
def add_booking_rate_l90d(df, df_nextQ):
    # no match / neg value stay nan in number_of_reviewsQ3
    
    df_reviews=df.copy()
    df_reviews=df_reviews.rename(columns={"number_of_reviews":'number_of_reviews_till_Q'})
    
    reviews_nextQ=df_nextQ[['id','number_of_reviews']]
    reviews_nextQ.columns=['id','number_of_reviews_till_nextQ']
    
    df_reviews=df_reviews.merge(reviews_nextQ, left_on='id', right_on="id", how="left")
    
    df_reviews['number_of_reviews_nextQ']=df_reviews['number_of_reviews_till_nextQ']-df_reviews["number_of_reviews_till_Q"]
    df_reviews['number_of_reviews_nextQ']=df_reviews['number_of_reviews_nextQ'].apply(lambda x : np.nan if x<0 else x)
    
    print(f"[CHECK] number_of_reviews_nextQ not match (to drop) :{len(df_reviews[df_reviews['number_of_reviews_nextQ'].isna()])};\n"
          f"ON availability_90 ==0 (to drop) {len(df_reviews[df_reviews['availability_90']==0])}!\n")

    df_reviews['booking_rate_l90d'] = df_reviews.apply(
                lambda row: min(row['number_of_reviews_nextQ'] / row['availability_90'], 1.0)
                if row['availability_90'] > 0 else None,
                axis=1
            )

    return df_reviews




def modeling_main(df_input, x_vars, y_var, key_vars, group_col,
                output_folder=None,
                to_fillna0=True,
                run_vif=False,
                # save_models_summary=False,  
                save_models_table=True,
                save_plots=True,
                ndigits=None
                ):
    print("check data".center(100,'='))
    if to_fillna0:
        print(f"[INFO] fillna in key cols!`\n")
        
        
    print("check output folder".center(100,'='))
    os.makedirs(output_folder, exist_ok=True)
    print(f"[INFO] results saved to {output_folder}!\n")
    
    if run_vif:    
        print("vif".center(100, '='))    
        check_vif (df=df_input, x_vars=x_vars, y_var=y_var, key_vars=key_vars, group_col=None)

    print("\n","basic model".center(100,"="),"\n")
    df, formula, model_basic=build_model (df_input=df_input,
            x_vars=x_vars, key_vars=None, #*
            to_fillna0=to_fillna0,
            y_var=y_var, group_col=None, #*
            # outpath_folder=output_folder, 
            # tex_filename='ols_summary_basic.tex', 
            # save=save_models_summary
            )
    
    print("\n", "tactics model".center(100, '='),"\n")
    df, formula, model_tactics=build_model (df_input=df_input,
            x_vars=x_vars, key_vars=key_vars, 
            to_fillna0=to_fillna0,
            y_var=y_var, group_col=None, #*
            # outpath_folder=output_folder, 
            # tex_filename='ols_summary_tactics.tex', 
            # save=save_models_summary
            )
    
    print("\n","interaction model".center(100,'='),"\n")
    df, formula, model_interaction=build_model (df_input=df_input, 
            x_vars=x_vars, key_vars=key_vars, 
            to_fillna0=to_fillna0,
            y_var=y_var, group_col=group_col, 
            # outpath_folder=output_folder, 
            # tex_filename='ols_summary_interaction.tex', 
            # save=save_models_summary
            )
    
    print("models table".center(100,'=')) 
    models_dict = {
        "Basique": model_basic,
        "Tactiques  ": model_tactics,
        "x Superhôte": model_interaction
    }
    table_kp, table_ctrl=make_models_table(models_dict=models_dict, 
                    vars_kp=key_vars+[group_col], #+langue？ 
                    ndigits=ndigits,
                    save=save_models_table, 
                    output_folder=output_folder, 
                    filename_noext=None)
        
        
    print("tactics plots".center(100,'='))
    layout_plots(df_input=df, x_vars=x_vars,y_var=y_var,
            tactics_vars=key_vars, 
            group_col=None,
            save=save_plots, 
            output_folder=output_folder,
            filename=None)
    
    print("interaction plots".center(100,"="))
    layout_plots(df_input=df, x_vars=x_vars,y_var=y_var,
            tactics_vars=key_vars,
            save=save_plots, 
            group_col='host_is_superhost', 
            output_folder=output_folder,
            )
    
    
    
    return 

