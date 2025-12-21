import pandas as pd
import numpy as np
import time, os, sys,re
from tqdm import tqdm
from datetime import datetime
from transformers import pipeline
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

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

def run_zsc(
    path_df_unique,
    dict_items=None,
    model_name="tasksource/ModernBERT-large-nli",#par défaut
    by_lang=True,
    outpath_zsc_result=None, 
    save_interval=1000,# save every 
    log=True):
    
    """
        无论如何都要输入原df_unique用于筛选除了中间结果中还有多少todo！
        自己定义输出路径：有中间结果，则更新原路径文件
        
        dict_items无输入时取默认；
        
        models :
        "MoritzLaurer/bge-m3-zeroshot-v2.0"
        "tasksource/ModernBERT-large-nli"
        
        outpath_zsc_result=output_zsc/model_name/listings_paris2406_items.csv
        
        返回df_zsc :id, host_about,lang, labels
        
    """
       
    if not dict_items:     
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
        dict_fr2en=dict(zip(labels_fr, labels_en))
        if len(labels_en)!=len(labels_fr):
            print(f"[CHECK] labels fr match labels en !")
    
    ## input    
    print("load df_unique & previous result".center(100,'-'))    
    df_unique=pd.read_csv(path_df_unique)

    # 已有结果，接着做没有处理过的：
    if not os.path.exists(outpath_zsc_result):
        print(f"[INFO] first zsc!")
        os.makedirs(os.path.dirname(outpath_zsc_result), exist_ok=True)
        df_done = pd.DataFrame()
        df_todo=df_unique.copy()
        
    else:
        df_done = pd.read_csv(outpath_zsc_result)
        done_ids = set(df_done["id"])
        df_todo = df_unique[~df_unique["id"].isin(done_ids)]
        print(f"[CHECK by id] {len(done_ids)} rows DONE,\n"
            f" {len(df_todo)} rows TO DO.")

    # df to zsc
    df=df_todo.copy()           

    # model
    print("load model".center(100,'-'))
    start_time_model=time.time()
    device=0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device
    )
    end_time_model=time.time()

    print(f"[MODEL] {model_name} loaded in {end_time_model-start_time_model:.2f} sec !")

    
    ## zsc 
    print("df_todo ==ZSC==> results_list".center(100,'-'))   
    
    def classify_row(row, by_lang):
        text = row["host_about"]            
        lang= row["lang"]
        
        try:
            # labels
            if by_lang :
                candidate_labels=dict_items[lang]
            else :
                candidate_labels=dict_items['en']

            #zsc
            res = classifier(text, candidate_labels, multi_label=True)#***
            
            # labels fr2en:
            if by_lang and lang == "fr":
                labels_en = [dict_fr2en[label] for label in res["labels"]]
            else:
                labels_en = res["labels"]  # already EN
            dict_scores = dict(zip(labels_en, res["scores"]))

            return  {
                    "id": row["id"],
                    "text":text,
                    # "lang": lang,
                    **dict_scores
                }
            
        except Exception as e:
            # 报错则将所有labels en初始化为nan
            all_en_labels = dict_fr2en.values() if lang == "fr" else dict_items["en"]
            return {
                "id": row["id"],
                "text":text,
                # "lang": lang,
                **{label: np.nan for label in all_en_labels}
            }
    
    results_list = []
    start_time = time.time()
    if by_lang:    
        print(f"[INFO] zsc by lang ! ")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="ZSC"):
        results_list.append(classify_row(row, by_lang=by_lang))
        
        # save_interval
        if (idx + 1) % save_interval == 0:
            ## 每次要和df_done合并保存，不然结果中只有上一轮论更新的，丢失上一次之前的！
            df_zsc = pd.concat([df_done, pd.DataFrame(results_list)], ignore_index=True)
            df_zsc.to_csv(outpath_zsc_result, index=False)
            print(f"✔ [SAVE] checkpoint {len(df_zsc)} / {len(df_unique)} saved to {outpath_zsc_result}!\n") 
    
    ## 最后一次保存!!!
    if results_list:
        # 合并done+todo :
        df_zsc = pd.concat([df_done, pd.DataFrame(results_list)], ignore_index=True)
        # save :有结果的更新原结果
        df_zsc.to_csv(outpath_zsc_result, index=False)
        print(f"✅ [FINAL SAVE] {len(df_zsc)} rows saved to {outpath_zsc_result}!")
    
    end_time = time.time()
    print(f"\n ⭐ [SUCCESS] ZSC sur {len(df)} textes avec {len(dict_items['en'])} EN labels/{len(dict_items['fr'])} FR labels \n"
          f"par {model_name} prend {(end_time - start_time)/3600:.2f} hours !\n")
    
    if log:# 中断无法储存？
        today=datetime.today()
        today_str=today.strftime('%Y-%m-%d %H:%M')
        # 和df_zsc存在同一个地方
        outpath_log=os.path.join(os.path.dirname(outpath_zsc_result), f"run.txt")
                
        with open(outpath_log, "a", encoding="utf-8") as f:
            f.write(
                f"{today_str.center(100, '-')}\n"
                f"df_unique:{path_df_unique}\n"
                f"zsc done :{len(results_list)}*{len(labels_en)}"
                f"labels EN: {labels_en}\n"
                f"labels FR :{labels_fr}\n"
                f"model : {model_name}\n"
                f"df_items :{outpath_zsc_result}\n"
                f"runtime: {(end_time - start_time)/3600:.2f} hours\n"
            )
            print(f"✅ [SAVE] log saved to {outpath_log}!")
    
    return pd.DataFrame(results_list)





def merge_by_host_about(path_df_zsc, path_df_filtered, 
                        save=False, output_folder=None):
    
    df_zsc=pd.read_csv(path_df_zsc)
    # print(f"[INFO] df_zsc columns :{df_zsc.columns}")
    
    items_cols=[c for c in df_zsc.columns if c not in ["id", 'text','lang']]
    
    df_filtered=pd.read_csv(path_df_filtered)
    print(f"[INFO] df_zsc :{len(df_zsc)}; df_filtered: {len(df_filtered)}!")
    print(f"[CHECK] items cols:{items_cols}\n")
    
    print("merge items scores back to df_filtered".center(100,'-'))
    df_items=df_filtered.merge(df_zsc, left_on="host_about", right_on="text", how='left')
    print(f"[INFO] len df items : {len(df_items)}\n"
          f"no match rows stay NaN on items cols!!")

    if save:
        if output_folder ==None:
            output_folder=os.path.dirname(path_df_filtered)
        os.makedirs(output_folder, exist_ok=True)
        
        filename=os.path.basename(path_df_filtered).replace("_filtered.csv", f"_items.csv")
        
        outpath_df_items=os.path.join(output_folder,filename) 
        df_items.to_csv(outpath_df_items, index=False)
        
        print(f"✅[SAVE] df merged with items scores saved to {outpath_df_items}!\n"
              f" ready for fa!")    
    return df_items
