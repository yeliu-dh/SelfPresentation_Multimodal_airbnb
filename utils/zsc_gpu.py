import pandas as pd
import numpy as np
import json
import time, os, sys,re
from tqdm import tqdm
from datetime import datetime
from transformers import pipeline
import torch
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
from concurrent.futures import ThreadPoolExecutor, as_completed


   
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



def classify_row(row,classifier):
    host_id=row['host_id']    
    text=row['host_about']
    lang=row["lang"]   
    
    """
    output =[
        {
            "host_id":host_id,
            "host_about":text,
            item:score
            
        },
        {
            
        }
    ]
    saved as .json!
        
    """
    
    if lang=="fr":
        candidate_labels=dict_items['fr']
    else :
        candidate_labels=dict_items['en']        
    #if not by_lang, fix lang=="en"?

    try:
        #zsc MULTILABEL!
        res = classifier(text, candidate_labels, multi_label=True)#***
        
        # labels fr2en:
        if  lang == "fr":
            labels_en = [dict_fr2en[label] for label in res["labels"]]
        else:
            labels_en = res["labels"]  # already EN
        dict_scores = dict(zip(labels_en, res["scores"]))
        # print("dict_scores:",dict_scores)

        return  {
                "host_id": host_id,
                "host_about":text,
                **dict_scores
            }
        
    except Exception as e:
        # 报错则将所有labels en初始化为nan
        all_en_labels = dict_fr2en.values() if lang == "fr" else dict_items["en"]
        return {
            "host_id": host_id,
            "host_about":text,
            **{label: np.nan for label in all_en_labels}
        }
        
        
def merge_from_results_json(df_input,results):
    df=df_input.copy()
    df_results=pd.DataFrame(results)
    
    df_results = df_results.drop_duplicates(subset=['host_id','host_about'])
      
    
    df_merged=df.merge(df_results, on=["host_about","host_id"], how='left')
    print(f"[check] {len(df_merged)} ==? {len(df_input)}")
    return df_merged

def run_zsc(
    df_input, 
    path_json="results_zsc.json", 
    model_name="MoritzLaurer/bge-m3-zeroshot-v2.0",#"tasksource/ModernBERT-large-nli",#par défaut
    batch_size=8,
    save_interval=1000,
    ):

    # ---input---
    df=df_input.copy()
    print(f"[info] INPUT df: {df.shape}")
    
    # ---filter---
    df_unique=df.dropna(subset="host_about").drop_duplicates(subset="host_about")
    print(f"[info] dropna & drop_duplicates on 'host_about': {df_unique.shape}\n")


    # ---load old results from json--
    if os.path.exists(path_json):
        with open(path_json, 'r', encoding='utf-8') as f:
            results=json.load(f)
    else :
        dir_json=os.path.dirname(path_json)
        os.makedirs(dir_json, exist_ok=True)
        results=[]
        
    
    # ---no repetition--
    # host_id : type : int
    # res 中host_id是str
    processed=set(int(res['host_id']) for res in results)        
    pending_df = df_unique[~df_unique['host_id'].isin(processed)]
    print(processed)
    print(pending_df['host_id'])
    print(f"[info] {len(processed)}/{len(df_unique)} already zsced!\n"
          f"{len(pending_df)}/{len(df_unique)} to run zsc!\n")  
    
    # ---load mode---
    device=0 if torch.cuda.is_available() else -1
    classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device
        )    
    
    # ---zsc---
    
    start_time = time.time()
    for idx, row in tqdm(pending_df.iterrows(), total=len(df), desc="ZSC"):
        results.append(classify_row(row,classifier))
        
        # ---interval save---
        if (idx + 1) % save_interval == 0:
            with open(path_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"[INTERVAL SAVE] Saved {idx+1} / {len(pending_df)} texts")

    # --- final save ---
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[FINAL SAVE] {len(results)}/{len(df_unique)} texts saved!")

    
    # ---show---
    end_time = time.time()
    runtime_h=(end_time-start_time)/3600
    print(f"\n[DONE] ZSC sur {len(df)} textes avec {len(dict_items['en'])} EN labels/{len(dict_items['fr'])} FR labels \n"
          f"par {model_name} prend {runtime_h:.2f} hours !\n")

    # ---merge---
    df_merged=merge_from_results_json(df_input=df,results=results)
    
    return df_merged






# def parse_output(row, output, dict_fr2en):
#     host_id = row["host_id"]
#     text = row["host_about"]
#     lang = row.get("lang", "en")  # 如果有语言字段

#     try:
#         # 👉 labels 统一转英文
#         if lang == "fr":
#             labels_en = [dict_fr2en.get(label, label) for label in output["labels"]]
#         else:
#             labels_en = output["labels"]

#         # 👉 label-score 映射
#         dict_scores = dict(zip(labels_en, output["scores"]))

#         return {
#             "host_id": host_id,
#             "host_about": text,
#             **dict_scores
#         }

#     except Exception as e:
#         print(f"[ERROR] host_id={host_id} : {e}")
#         return {
#             "host_id": host_id,
#             "host_about": text,
#             "error": str(e)
#         }
    
    
    
    # # --- zsc --- batch + tqdm（GPU推荐写法）
    # start_time = time.time()
    # processed_count = 0
    
    # rows = pending_df.to_dict("records")
    # texts = [row["host_about"] for row in rows]
    # print(f"[info] process by batch : {batch_size}")

    # for i in tqdm(range(0, len(texts), batch_size), desc="ZSC on host_about..."):
    #     batch_texts = texts[i:i+batch_size]
    #     batch_rows = rows[i:i+batch_size]

    #     # batch
    #     outputs = classifier(
    #         batch_texts,
    #         candidate_labels=dict_items['en'] + dict_items['fr'],  # 你的labels
    #         multi_label=True
    #     )

    #     # ---parse
    #     for row_input, output in zip(batch_rows, outputs):
    #         row_result = parse_output(row_input, output, dict_fr2en)
    #         results.append(row_result)
    #         processed_count += 1

    #     # ---interval save---
    #     if processed_count % save_interval == 0:
    #         with open(path_json, "w", encoding="utf-8") as f:
    #             json.dump(results, f, indent=2, ensure_ascii=False)
    #         print(f"[INTERVAL SAVE] Saved {processed_count} / {len(texts)} texts")
    
    
    

# #---zsc---多线程 + tqdm 
    # start_time = time.time()

    # processed_count = 0
    # # classify_one(text, lang, classifier)
    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     futures = {executor.submit(classify_row, row, classifier): row for _, row in pending_df.iterrows()}
    #     for future in tqdm(as_completed(futures), total=len(futures), desc="ZSC on host_about..."):
    #         row = future.result()
    #         results.append(row)
    #         processed_count += 1

    #         # interval save
    #         if processed_count % save_interval == 0:
    #             with open(path_json, "w", encoding="utf-8") as f:
    #                 json.dump(results, f, indent=2, ensure_ascii=False)
    #             print(f"[INTERVAL SAVE] Saved {processed_count} / {len(pending_df)} texts")
    
    # # -------- 最终保存 --------
    # with open(path_json, "w", encoding="utf-8") as f:
    #     json.dump(results, f, indent=2, ensure_ascii=False)
    #     print(f"[FINAL SAVE] Saved {processed_count} / {len(pending_df)} texts!")
    
    