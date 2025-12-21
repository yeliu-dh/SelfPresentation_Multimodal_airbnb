import pandas as pd
import os, sys, importlib
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


## fa
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer import FactorAnalyzer
from pingouin import cronbach_alpha
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D



## my utils
from utils.io import save_csv_as_latex

def run_fa(path_df_items, items, output_folder="../output_fa",#fallback!
        output_folder_data="../data_processed",# for df_scores(tactics)
        n_factors=5, rotation="oblimin", 
        fa_names=None,
        dict_factor_items=None,
        get_items_heatmap=True,
        get_items_table=True, low_comm_t=0.5,
        get_factor_scores_by='weighted',
        run_fa_on_scores="weighted",
        get_scores_heatmap=True,
        get_cronbachs_a=True,
        get_barplot=False, get_pca3d=False, 
        ndigits=4):
    # no suffix!
    # 英语提示，但是图片标题用法语！
    ## fa DROPNA!
    
    df=pd.read_csv(path_df_items)
    print(f"[INFO] df_items : {len(df)} rows!")
    
    #output
    os.makedirs(output_folder, exist_ok=True)
    print(f"[INFO] all output saved to '{output_folder}'!\n"
          f"df with fa scores saved to '{output_folder_data}' !\n"
          f"[CHECK] INITIALISE DF, or MERGE fa scores will go WRONG!!!")

    print(f"============================RUN FA=================================")
    #-------------------------- check data -----------------------------
    missing_labels = [col for col in items if col not in df.columns]
    if missing_labels:
        print(f"[WARNING] These items are missing in df: {missing_labels}\n")

    else :
        df_dropna=df[df[items].sum(axis=1)!=0]
        data=df_dropna[items]
        print(f"[INFO] len scores in df notna:{len(data)}\n")
    
    # ---------------------------kmo-----------------------------------
    kmo_all, kmo_model = calculate_kmo(data)
    print(f"[INFO] kmo_model score (> 0.7) :{kmo_model}\n")


    # ---------------------------fa fit -------------------------------
    if rotation =="oblimin":
        print(f"[INFO] oblimin allows cross-loadings!\n")        
    fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method='minres')
    fa.fit(data)
    ### output impo :fa, data


    print("---------------------------heatmap of items-------------------------------")
    if get_items_heatmap==True: 
        loadings = fa.loadings_
        print(f"[INFO] good loadings > 0.5")
        
        if fa_names :
            print(f"Rename loadings df with fa_names in proper order !")
            loadings_df=pd.DataFrame(loadings, index=data.columns, columns=fa_names)
        else :
            loadings_df=pd.DataFrame(loadings, index=data.columns)  
    
        plt.figure(figsize=(10,8))
        sns.heatmap(loadings_df, annot=True, cmap='coolwarm')
        plt.title('Charges factorielles des items')
        # plt.xticks(rotation=45)
        plt.tight_layout()
        outpath_heatmap=os.path.join(output_folder, 'heatmap_fa_items.jpeg')
        plt.savefig(outpath_heatmap, dpi=300)
        plt.show()
        print(f"✅ [SAVE] heatmap of items saved to {outpath_heatmap}!\n")
    
    
    
    print(f'organise input FA_ITEMS_DICT as latex table'.center(100, '-'))
    if dict_factor_items:# dict to csv
        fa_items_str_dict = {
            k: ", ".join(v)
            for k, v in dict_factor_items.items()
        }

        fa_items_df=pd.DataFrame.from_dict(fa_items_str_dict, orient='index',columns=['items'])
        fa_items_df.reset_index(names='Tactique', inplace=True)
        display(fa_items_df)
               
        outpath_fa_items=os.path.join(output_folder, "latex","fa_items_table.tex")
        save_csv_as_latex(table_csv=fa_items_df, 
                          output_path=outpath_fa_items, 
                          caption="Bilan des items des tactiques de la présentation de soi", 
                          label="tab:fa_items_table", 
                          ndigits=ndigits,escape=True, index=False)
        
    
    if get_items_table==True:#***
        print("# ------------------------------------tab items fa------------------------------------")
        loadings = fa.loadings_
        communalities = fa.get_communalities()
        uniqueness = fa.get_uniquenesses()

        # gamma 和 sigma
        abs_loadings = np.abs(loadings)
        gamma = abs_loadings.max(axis=1)                   # 最大因子载荷 !
        sigma = np.sort(abs_loadings, axis=1)[:, -2]       # 第二大因子载荷!

        # get DataFrame
        desired_order = None  # 先定义，保证一定存在
        if fa_names:
            fa_names_in_tab=fa_names
            ##如有fa输入按照这个整理!
            desired_order=['gamma','sigma','ouverture','authenticité','sociabilité',"auto_promotion","exemplarité",'communalité','spécificité']
            
        else :
            #没有fa_names，则无法按照上面的顺序整理！
            fa_names_in_tab=[f"Factor{i+1}" for i in range(0, loadings.shape[1])]
            
        columns = ['gamma', 'sigma'] + \
                fa_names_in_tab + \
                ["communalité","spécificité"]
                # ['communality', 'uniqueness']

        df_fa = pd.DataFrame(
            np.column_stack([gamma, sigma, loadings, communalities, uniqueness]),
            index=data.columns,
            columns=columns
        )
        df_fa.round(ndigits)
        
        # reorder 
        # desired_order=['gamma','sigma','ouverture','authenticité','sociabilité',"auto_promotion","exemplarité",'communalité','spécificité']
        if desired_order:
            df_fa=df_fa[desired_order]
        
        if dict_factor_items and fa_names:
            df_fa_num = df_fa.copy()# round的是时候不能有'-'
            factor_cols = fa_names #选取因子列
            
            # 先把所有因子列设为 "-"
            df_fa.loc[:, factor_cols] = '-'

            # 再逐个因子填回“所属 item”的 loading
            for factor, items in dict_factor_items.items():
                for item in items:
                    if item in df_fa.index and factor in df_fa.columns:
                        df_fa.loc[item, factor] = round(
                            float(df_fa_num.loc[item, factor]), ndigits
                        )
                        df_fa.loc[item]
        # display(df_fa)
    
            
        #info
        print(f"[INFO] mean COMM (> 0.6):{communalities.mean()}\n"
                    f"[INFO] mean UNIQ : {uniqueness.mean()}\n")
        low_comm_t=0.5
        low_comm_items = df_fa[df_fa['communalité'] <low_comm_t ]
        print(f"[WARNING] {len(low_comm_items)}/{len(df_fa)} low communality items (< {low_comm_t}):\n{';'.join(low_comm_items.index)}\n")

        # save as csv
        outpath_items_fa_csv=os.path.join(output_folder, 'items_fa.csv')
        df_fa.to_csv(outpath_items_fa_csv, index=True)##index!!!
        print(f"✅ [SAVE] items fa CSV saved to {outpath_items_fa_csv}!\n")
        
        # save as latex               
        outpath_items_fa_latex=os.path.join(output_folder, 'latex','items_fa.tex')        
        save_csv_as_latex(table_csv=df_fa, output_path=outpath_items_fa_latex, 
                          caption="Tableau de l'analyse factorielle des items", 
                          label="tab:items_fa", 
                          ndigits=ndigits,
                          escape=True,
                          index=True)
        
        ## ps.
        print(f"[CHECK] remember to RENAME factors by heatmap of zsc data!!!")            
    

    print("get fa scores(tactics)".center(100,"="))
    # 从原始 df 中筛选非空行 (df_dropna) 做 FA
    # 得到 FA scores (df_scores) (无id索引)！
    # 拼接回 df_dropna → 得到 df_tactics
    # 按 "host_about" 去重 → df_bio_tactics
    # 最后再 merge 回原始 df → listings_tactics
    
    # init，空的df不会拼接上
    df_scores_mean=pd.DataFrame()
    df_scores_weighted=pd.DataFrame()
    desired_order_tactics=['ouverture','authenticité','sociabilité',"auto_promotion","exemplarité"]   
    # reorder for heatmap!
    
    if get_factor_scores_by  in ["mean","both"]:
        print(f"[INFO] calulate factor score by taking the average of its corresponding items, ref to a pre-defined map.\n")
        
        # fa_items_dict={
        #     "ouverture":['open to different cultures','cosmopolitan','international view','cultural exchange'],
        #     "authenticité":['personal life','life experiences','divers interests','hobbies','enjoy life'],
        #     'sociabilité':['meet new people', 'welcoming', 'friendly','sociable', 'interpersonal interaction'],
        #     'auto_promotion':['thoughtful service', 'attentive to needs','willing to help','responsive'],
        #     'exemplarité':["fan of Airbnb","Airbnb community",'love Airbnb', 'travel with Airbnb']
        # }
        
        if dict_factor_items:            
            fa_scores_mean = pd.DataFrame(index=data.index)
            for f, items in dict_factor_items.items():
                fa_scores_mean[f] = data[items].mean(axis=1)
            df_scores_mean=pd.DataFrame(fa_scores_mean).reset_index(drop=True)
            
            mean_order=[tac for tac in desired_order_tactics]
            print(f"MEAN ORDER:{mean_order}")
            df_scores_mean=df_scores_mean[mean_order]
        else : 
            print(f"[CHECK] need FA_ITEMS_DICT!!")
            

    if get_factor_scores_by in ["weighted","both"]:
        fa_scores_weighted = fa.transform(data)#用和items fa相同的fa transforme data就可以得到加权weighted!的scores==tactics!
        df_scores_weighted=pd.DataFrame(fa_scores_weighted).reset_index(drop=True)
       
        if fa_names:
            df_scores_weighted.columns=[col for col in fa_names]
            #reorder:
            weighted_order=[tac for tac in desired_order_tactics]
            print(f"WEIGHTED ORDER:{weighted_order}")
            df_scores_weighted=df_scores_weighted[weighted_order]


    if get_factor_scores_by not in ['mean','weighted', 'both']:
        print(f"[WARNING] choose from 'mean' or 'weighted' or 'both'!!")
    
    
    # include mean AND/OR weighted 
    df_scores=pd.concat([df_scores_mean, df_scores_weighted],axis=1)       
    print(f'[INFO] shape df scores = 5*len(get_factor_scores_by):{df_scores.shape}')
    

    print("---------------------------------run fa on scores(must do)--------------------------------------")
    
    # plot
    def run_scores_fa(df_scores, run_fa_on_scores, get_scores_heatmap=get_scores_heatmap, 
                    output_folder=output_folder):
        title_map={'mean':'moyens',
                   'weighted':'pondérés'}
        
        # init fa & fit 
        fa_scores = FactorAnalyzer(n_factors=2, rotation=rotation, method='minres') 
        fa_scores.fit(df_scores)
        
        # data
        loadings_scores = fa_scores.loadings_
        loadings_scores_df=pd.DataFrame(loadings_scores, index=df_scores.columns)
        loadings_scores_df.columns=["Dimension 1","Dimension 2"]
        display(loadings_scores_df)
        # save as csv
        
        outpath_scores_df=os.path.join(output_folder, 'scores_fa.csv')
        loadings_scores_df.to_csv(outpath_scores_df, index=True)
        print(f"✅[SAVE] scores df saved to {outpath_scores_df}!")
        
        
        # save as latex
        outpath_scores_fa_latex=os.path.join(output_folder, "latex",f"scores_{run_fa_on_scores}_fa.tex")        
        save_csv_as_latex(table_csv=loadings_scores_df, output_path=outpath_scores_fa_latex,
                          caption="Tableau de l'analyse factorielle des tactiques",
                          label="tab: tactics_fa",
                          ndigits=ndigits, escape=True, index=True)       
        # plot
        if get_scores_heatmap:      
            sns.heatmap(loadings_scores_df, annot=True, cmap='coolwarm')
            
            plt.title(f'Charge factorielles des tactiques (scores {title_map.get(run_fa_on_scores,None)})')
            plt.tight_layout()
            
            outpath_scores_heatmap=os.path.join(output_folder,f'heatmap_fa_scores_{run_fa_on_scores}.jpg')
            plt.savefig(outpath_scores_heatmap,dpi=300)
            
            plt.show()
            print(f"✅ [SAVE] heatmap of {title_map.get(run_fa_on_scores,None)} scores saved to {outpath_scores_heatmap}!\n")
        
        return
    
    if run_fa_on_scores in ['mean', "both"] and not df_scores_mean.empty:   
        run_scores_fa(df_scores=df_scores_mean, run_fa_on_scores='mean', output_folder=output_folder)
    
    if run_fa_on_scores in ['weighted', "both"] and not df_scores_weighted.empty:
        run_scores_fa(df_scores=df_scores_weighted, run_fa_on_scores="weighted", output_folder=output_folder)
    
    if run_fa_on_scores not in ["mean",'weighted', "both"] :
        print(f"[INFO] choisir between 'mean','weighted','both' to run fa on tactics' scores!\n")




    print("----------------------------merge data to listings(keeping nan)---------------------------------")
    df_dropna = df_dropna.reset_index(drop=True)
    df_scores=df_scores.reset_index(drop=True)
    df_tactics=pd.concat([df_dropna, df_scores], axis=1)# horizontal
    
    df_bio_tactics=df_tactics.drop_duplicates(subset="host_about")
    cols_tosave=["host_about"]+df_scores.columns.tolist()
    df_bio_tactics=df_bio_tactics[cols_tosave]
    listings_tactics=df.merge(df_bio_tactics, left_on='host_about', right_on='host_about', how="left")
    
    #数据放在和OUTPUT_FOLDER同级文件夹: data_processed
    os.makedirs(output_folder_data, exist_ok=True)
    # filename=path_df_items.replace("_items", "_scores")
    filename=os.path.basename(path_df_items).replace("_items", "_scores")
    outpath_listings_tactics=os.path.join(output_folder_data,filename)
    listings_tactics.to_csv(outpath_listings_tactics, index=False)
    display(listings_tactics.head())
    
    print(f"[INFO] scores shape (len(data), 1/2*n_factors) :{df_scores.shape}\n"
            f"len(df_dropna) should == len(df_scores)==len(df_tactics): {len(df_dropna)}, {len(df_scores)}, {len(df_tactics)}\n"
            f"len(DF)==len(listings_tactics):{len(df)},{len(listings_tactics)}\n")
    print(f"✅ [SAVE] listings with tactics scores by '{get_factor_scores_by}'way saved to {outpath_listings_tactics}!!\n")
            
        
    
    print("###====================================CHECK=======================================###")   

    if get_cronbachs_a==True and dict_factor_items:
        print("---------------------------cronbach's alpha-------------------------------")
        print(f"[INFO] good alpha > 0.7\n")
        print(f"Cronbach's alpha checks internal consistency, whether items mesure a lantent construct !\n")
        
        # predifine mapping dict
        # items2tac={
        #     "openness":['open to different cultures','cosmopolitan','international view','cultural exchange'],
        #     "authenticity":['personal life','life experiences','divers interests','hobbies','enjoy life'],
        #     'sociability':['meet new people', 'welcoming', 'friendly','sociable', 'interpersonal interaction'],
        #     'self_promotion':['thoughtful service', 'attentive to needs','willing to help','responsive'],
        #     'exemplification':["fan of Airbnb","Airbnb community",'love Airbnb', 'travel with Airbnb']
        # }
        
        dict_alpha={}
        for tactic, items in dict_factor_items.items():
            alpha, _ = cronbach_alpha(data[items])
            # print(f"- {tactic}: {alpha}")          
            dict_alpha[tactic]=alpha
        dict_alpha_df = pd.Series(dict_alpha, name='cronbach_alpha').reset_index()
        dict_alpha_df.columns = ['tactique', 'cronbach_alpha']
        display(dict_alpha_df)
        
        outpath_alpha=os.path.join(output_folder, "cronbachs_alpha.csv")
        dict_alpha_df.to_csv(outpath_alpha, index=False)
        print(f"✅ [SAVE] cronbach's alpha on items saved to {outpath_alpha}!\n")
    
    elif get_cronbachs_a and not dict_factor_items:
        print(f"[CHECK] get cronbach's alpha need dict_factor_items!!")   

        
        
    if get_barplot==True:        
        print(f"\n-----------------------------barplot comm&uniq-------------------------------")
        communalities = fa.get_communalities()
        uniqueness = fa.get_uniquenesses()
        print(f"[INFO] mean COMM (> 0.6):{communalities.mean()}\n"
            f"[INFO] mean UNIQ : {uniqueness.mean()}\n")
        
        
        comm_df = pd.DataFrame({
            'Communality': communalities,
            'Uniqueness': uniqueness
        }, index=data.columns)
        comm_df = comm_df.sort_values("Communality")

        plt.figure(figsize=(10,6))
        colors = sns.color_palette("viridis", n_colors=2)  # 两个指标

        comm_df.plot(kind='bar', stacked=False, color=colors)# edgecolor='black'
        plt.title("Communalités et spécificités des items", fontsize=16)
        plt.ylabel("Proportion de la Variance Expliquée")
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Metric')
        plt.tight_layout()
        outpath_barplot=os.path.join(output_folder,"comm_uniq_barplot.jpg")
        plt.savefig(outpath_barplot, dpi=300)
        plt.show()
        print(f"✅ [SAVE] barplot saved to {outpath_barplot}!\n")


    
    
    if get_pca3d==True:
        print("# -----------------------------3D PCA-------------------------------")
        pca_3d=PCA(n_components=3)
        loadings_3d=pca_3d.fit_transform(loadings)
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111, projection='3d')
        
        # 缩放向量
        norms = np.linalg.norm(loadings_3d, axis=1)
        loadings_3d_scaled = loadings_3d / norms[:, np.newaxis] #0.8
        print("[INFO] Explained variance ratios 3D (>0.6):", pca_3d.explained_variance_ratio_.sum())

        # 画箭头
        for i, var in enumerate(data.columns):
            ax.quiver(0, 0, 0,
                    loadings_3d_scaled[i, 0],
                    loadings_3d_scaled[i, 1],
                    loadings_3d_scaled[i, 2],
                    color='red', arrow_length_ratio=0.1)

            ax.text(loadings_3d_scaled[i, 0]*1.1,
                    loadings_3d_scaled[i, 1]*1.1,
                    loadings_3d_scaled[i, 2]*1.1,
                    var, fontsize=10)

        # 坐标轴限制
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')

        plt.title('3D PCA Correlation Circle (Scaled)')
        plt.tight_layout()
        outpath_pca=os.path.join(output_folder, 'pca_3d.jpg')
        plt.savefig(outpath_pca,dpi=300)
        plt.show()
        print(f"✅ [SAVE] PCA 3D saved to {outpath_pca}!\n")
    
        
    return 