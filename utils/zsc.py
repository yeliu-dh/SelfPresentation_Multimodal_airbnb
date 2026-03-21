import pandas as pd
import numpy as np
import time, os, sys,re
from tqdm import tqdm
from datetime import datetime
from transformers import pipeline
import torch

"""

classifier_bge = pipeline("zero-shot-classification", model="MoritzLaurer/bge-m3-zeroshot-v2.0", use_safetensors=False,device=2)

sequences = [
    "The government passed a new policy.",
    "This dish uses a lot of spices.",
    "She won the tennis championship."
]

labels = ["politics", "cooking", "sports"]

results = classifier_bge(sequences, candidate_labels=labels, multi_label=True)


"""


def zsc_text(row, classifier, by_lang, dict_items):
        # host_about
        text = row["host_about"]            
        
        # labels
        if by_lang :
            # 若分语言，寻找对应语言的items，若没有对应语言的items，默认en items
            lang= row["lang"]
            candidate_labels=dict_items.get(lang, dict_items['en'])
        else :
            candidate_labels=dict_items['en']

        # zsc
        dict_fr2en=dict(zip(dict_items["fr"], dict_items["en"]))
        
        try:
            res = classifier(text, candidate_labels, multi_label=True)#*** 
            # labels fr2en:
            if by_lang and lang == "fr":
                labels_en = [dict_fr2en[label] for label in res["labels"]]
            else:
                labels_en = res["labels"]  # already EN
            dict_scores = dict(zip(labels_en, res["scores"]))

            return  {
                    "host_id": row["host_id"],
                    "host_about":text,
                    # "lang": lang,
                    **dict_scores
                }
            
        except Exception as e:
            # 报错则将所有labels en初始化为nan
            all_en_labels = dict_fr2en.values() if lang == "fr" else dict_items["en"]
            return {
                "host_id": row["host_id"],
                "host_about":text,
                # "lang": lang,
                **{label: np.nan for label in all_en_labels}
            }
            


def run_zsc(
    df_input,
    model_name="MoritzLaurer/bge-m3-zeroshot-v2.0",#"MoritzLaurer/bge-m3-zeroshot-v2.0", #"tasksource/ModernBERT-large-nli",#par défaut
    by_lang=True,
    path_results=None, 
    save_interval=1000, 
    ):
    
    """
    INPUT : df_processed

    OUTPUT : 
        results: csv : 
            host_id:int, host_about:str, items:int
        df_zsced=df_processed.merge(results_zsc)
        
    """
    
    # ---input: df_unique--- 
    df=df_input.copy()
    df_unique=df.dropna(subset="host_about").drop_duplicates(subset="host_about")
    print(f"[info] df: {len(df)}; df_unique : {len(df_unique)}\n")
    
    
    # ---dict_items---
    labels_en=[
        'open to different cultures', 'cosmopolitan','international view', 'cultural exchange',
        'personal life', 'life experiences', 'divers interests', 'hobbies', 'enjoy life',
        'meet new people', 'welcoming', 'friendly', 'sociable', 'interpersonal interaction',
        'thoughtful service', 'attentive to needs', 'willing to help', 'responsive',
        'fan of Airbnb', 'Airbnb community','love Airbnb', 'travel with Airbnb'
    ]
    labels_fr=[
            'ouvert aux différentes cultures', 'cosmopolite','vue internationale', 'échange culturel',
            'vie personnelle', 'parcours personnel', 'loisirs', 'passions', 'aimer la vie',
        'rencontrer de nouvelles personnes', 'accueillant', 'amical','sociable', 'interaction interpersonnelle',
        'rendre service','attentif aux besoins', 'prêt à aider', 'réactif',
        "adepte d'Airbnb",'communauté Airbnb', 'aime Airbnb', 'voyager par Airbnb'
        ]
    dict_items={'en':labels_en,
                'fr':labels_fr}
    # dict_fr2en=dict(zip(dict_items["fr"], dict_items["en"]))
    
    if len(labels_en)!=len(labels_fr):
        print(f"[CHECK] labels fr match labels en !")
    
    # ---IO:no repetition---
    # NB. host_id, id都是唯一的，但是host_about不是！
    
    if os.path.exists(path_results):
        results_df=pd.read_csv(path_results)
        processed=set(results_df['host_about'])
        pending_df=df_unique[~df_unique['host_about'].isin(processed)]  
        
        print(f"[info] {len(processed)} rows already zsced; {len(pending_df)} rows to zsc!\n")

    else :
        pending_df=df_unique.copy()#input
        results_df=pd.DataFrame()#output  
    
    
    # ---model---
    print(f"loading model...\n")
    device=0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device
    )

    ## zsc 
    results= []
    start_time = time.time()
    if by_lang:
        print(f"zsc by lang !")

    for idx, row in tqdm(pending_df.iterrows(), total=len(pending_df), desc="ZSC text..."):
        result=zsc_text(row, classifier, by_lang, dict_items)
        results.append(result)
        
        # ---save every----
        if (idx + 1) % save_interval == 0:
            results_df = pd.concat([results_df, pd.DataFrame(results)], ignore_index=True)
            results_df.to_csv(path_results, index=False)
            results=[]# 重置
            # results_df不断增加并保存，results储存每save_interval条结果
            print(f"[INTERVAL SAVE] {len(results_df)} / {len(df_unique)} rows zsced and saved!\n")            
    
    # ---final save---
    if results:
        results_df = pd.concat([results_df, pd.DataFrame(results)], ignore_index=True)
        results_df.to_csv(path_results, index=False)
        print(f"✅ [FINAL SAVE] {len(results_df)}/{len(df_unique)} rows saved to {path_results}!\n")
    
    end_time = time.time()
    print(f"\n[DONE] ZSC sur {len(df)} textes avec {len(dict_items['en'])} EN labels/{len(dict_items['fr'])} FR labels \n"
          f"par {model_name} prend {(end_time - start_time)/3600:.2f} hours ( {(end_time - start_time):.2f} sec)!\n")

    
    # --merge--
    cols_to_merge=['host_about']+labels_en
    results_to_merge=results_df[cols_to_merge]    
    df_zsc=df.merge(results_to_merge, left_on="host_about", right_on='host_about', how='left')
    print(f"df_zsc: {df_zsc.shape};")
    
    
    return df_zsc



# def merge_by_host_about(path_df_zsc, path_df_filtered, desired_items_order=None,
#             text_col="text", 
#             save=False, output_folder=None, filename=None):
    
#     df_zsc=pd.read_csv(path_df_zsc)
#     # print(f"[INFO] df_zsc columns :{df_zsc.columns}")
    
#     # ONLY text_col+items
#     cols_to_merge=[c for c in df_zsc.columns if c not in ["id",'lang']]
#     df_zsc_to_merge=df_zsc.copy()[cols_to_merge]
#     if desired_items_order :
#         desired_order=[text_col]+desired_items_order
#         df_zsc_to_merge=df_zsc_to_merge[desired_order]

#     df_filtered=pd.read_csv(path_df_filtered)
#     print(f"[INFO] df_zsc :{len(df_zsc)}; df_filtered: {len(df_filtered)}!")
#     print(f"[CHECK]cols to merge :{cols_to_merge}\n")
    
#     print("merge items scores back to df_filtered".center(100,'-'))
#     df_items=df_filtered.merge(df_zsc_to_merge, left_on="host_about", right_on=text_col, how='left')
#     print(f"[INFO] len df items : {len(df_items)}\n"
#         f"no match rows stay NaN on items cols!!")

#     if save:
#         os.makedirs(output_folder, exist_ok=True)
#         if filename==None:
#           filename=os.path.basename(path_df_filtered).replace("_filtered.csv", f"_items.csv")
        
#         outpath_df_items=os.path.join(output_folder,filename) 
#         df_items.to_csv(outpath_df_items, index=False)
        
#         print(f"✅[SAVE] df merged with items scores saved to {outpath_df_items}!\n"
#               f" ready for fa!")    
#     return df_items

# def merge_by_host_about(path_df_zsc, path_df_filtered, 
#                         save=False, output_folder=None):
    
#     df_zsc=pd.read_csv(path_df_zsc)
#     # print(f"[INFO] df_zsc columns :{df_zsc.columns}")
    
#     items_cols=[c for c in df_zsc.columns if c not in ["id", 'text','lang']]
    
#     df_filtered=pd.read_csv(path_df_filtered)
#     print(f"[INFO] df_zsc :{len(df_zsc)}; df_filtered: {len(df_filtered)}!")
#     print(f"[CHECK] items cols:{items_cols}\n")
    
#     print("merge items scores back to df_filtered".center(100,'-'))
#     df_items=df_filtered.merge(df_zsc, left_on="host_about", right_on="text", how='left')
#     print(f"[INFO] len df items : {len(df_items)}\n"
#           f"no match rows stay NaN on items cols!!")

#     if save:
#         if output_folder ==None:
#             output_folder=os.path.dirname(path_df_filtered)
#         os.makedirs(output_folder, exist_ok=True)
        
#         filename=os.path.basename(path_df_filtered).replace("_filtered.csv", f"_items.csv")
        
#         outpath_df_items=os.path.join(output_folder,filename) 
#         df_items.to_csv(outpath_df_items, index=False)
        
#         print(f"✅[SAVE] df merged with items scores saved to {outpath_df_items}!\n"
#               f" ready for fa!")    
#     return df_items
