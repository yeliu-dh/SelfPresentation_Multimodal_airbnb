import pandas as pd
import time
import os, sys
import requests
from requests.exceptions import RequestException
import time
import shutil
from tqdm import tqdm
tqdm.pandas()  # 这行让 pandas 的 apply() 支持进度条

from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

##==================================== STRATIFIED SAMPLING=================================================##

def stratified_sampling(df, groupby=["is_changed", "has_host_about", "lang"],N_total=2000):
    print(f"*******************STRATIFIED SAMPLING***************************\n"
          f"METHODS:\n"
          f"regroupe df by {','.join(groupby)};\n"
          f"sampling proportionally {N_total} pics;\n"
          f"=> get a list of set (host_id, host_picture_id)\n")
    
 
    print(f"===============INPUT INFO=================\n"
          f"len before deduplication by host_id:{len(df)}")
    df=df.drop_duplicates(subset=['host_id'])
    cols=['host_id', "host_picture_url"]+groupby
    df=df[cols]
    print(f"after:{len(df)}\n")#-5K

    strata = (
        df
        .groupby(["is_changed", "has_host_about", "lang"])
        .size()
        .reset_index(name="N_s")
    )
    # 每个类别按照原df比例在smaple中应该拥有的数量 = 按比例 * 总数
    strata["n_sample"] = (strata["N_s"] / strata["N_s"].sum() * N_total).round().astype(int)
    print(f"strata :\n {strata}\n")


    sampled_list = []
    for _, row in strata.iterrows():
        cond = (
            (df["is_changed"] == row["is_changed"]) &
            (df["has_host_about"] == row["has_host_about"]) &
            (df["lang"] == row["lang"])
        )
        subset = df[cond]

        # 如果某层数量不足，直接全取
        n = min(row["n_sample"], len(subset))

        sampled = subset.sample(n=n, random_state=42)
        sampled_list.append(sampled)

    sampled_df = pd.concat(sampled_list, ignore_index=True)
    # print(sampled_df,"\n")

    id_url_df=sampled_df.copy()[:][['host_id','host_picture_url']]
    id_url_list= list(zip(id_url_df["host_id"], id_url_df["host_picture_url"]))
    print(f"len SAMPLE/'id_url_list': {len(id_url_list)}")
    display(sampled_df.head())
    return sampled_df, id_url_list



##===========================================DOWNLOAD==============================================##
# URL 校验
def check_pic_url(url, timeout=3):
    if not isinstance(url, str) or url.strip() == '':
        return False
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return response.status_code == 200
    except RequestException:
        return False

# 单张图片下载函数
def download_image(id, url, out_dir='images_raw', timeout=10):
    if not check_pic_url(url, timeout=timeout):
        print(f"[WARNING] 无效 URL: {url}")
        return None

    os.makedirs(out_dir, exist_ok=True)

    filename= f"{str(id)}.jpg"
    out_path = os.path.join(out_dir, filename)


    # 防止覆盖已有文件
    if os.path.exists(out_path):
        print(f"[INFO] host {id} pic 已存在，跳过： {out_path}")
        #CONTINUE只能在循环中使用
        return out_path

    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        with open(out_path, 'wb') as f:
            for chunk in response.iter_content(1024*8):
                if chunk:
                    f.write(chunk)
        return out_path
    except RequestException as e:
        print(f"[ERROR] 下载失败: {url}, 错误: {e}")
        return None



# 批量并行下载函数
def download_images_batch(id_url_list, out_dir='images_raw', max_workers=12):
    """
    id_url_list: list of tuples [(host_id, url), ...]
    返回: dict {host_id: 保存路径 or None}
    """
    start_time=time.time()
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        """
        字典 futures
        key = Future 对象
        value = host_id
        这样我们在任务完成后，可以知道这个 Future 对应的是哪个 host_id，方便把结果写回字典。
        """       
        futures = {executor.submit(download_image, id, url, out_dir): id for id, url in id_url_list}
        
        # for fut in as_completed(futures):
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading images..."):
            host_id = futures[fut]
            try:
                path = fut.result()
                results[host_id] = path #获取id对应的结果image_out_path
            except Exception as e:
                results[host_id] = None
                # print(f"[ERROR] 下载出错: host_id={host_id}, 错误: {e}")
    end_time=time.time()
    print(f"\n[SUCCES] downloaded {len(id_url_list)} pics : {end_time-start_time:.2f} sec!\n")
    #try1:[SUCCES] downloaded 2001 : 673.04 sec!

    # report:
    print(f"success: {len([p for p in results.values() if p])} pics!")
    print(f"fails: {len([u for u, p in results.items() if not p])} pics!")#无效url/下载失败！

    return results



import os
import json
import pandas as pd
import time
from tqdm import tqdm


def ls_import_data(df,folder='images_TEST'):
    start_time=time.time()

    # 读取 images_TEST 文件名
    # folder = "images_TEST"
    filenames = sorted(os.listdir(folder))

    records = []
    for file in tqdm(filenames, desc="Generating import data for label studio..."):
        # 跳过非图片文件
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        # 文件名：41925864.jpg → 41925864
        host_id = os.path.splitext(file)[0]

        # host_id 通常是整数，所以这里尝试转成 int
        try:
            host_id_int = int(host_id)
        except:
            print(f"[WARN] Cannot parse host_id from file: {file}")
            continue

        # 查找该 host_id 对应的行
        row = df[df["host_id"] == host_id_int]

        if row.empty:
            print(f"[WARN] host_id {host_id_int} not found in DataFrame!")
            continue

        row = row.iloc[0]

        # 构造 Label Studio task 数据
        record = {
            # "folder": folder,                # 保留原始文件名作为额外信息
            "host_id": int(row["host_id"]),
            "filename":file,
            "image": row["host_picture_url"],      # 自动加载远程 URL
        }
        records.append(record)

    # 保存成 JSON
    output_path = os.path.join(folder, "000filename_url_records.json")
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    end_time=time.time()
    print(f"[SUCCESS] JSON filename:url saved to {output_path}, total {len(records)} items: {end_time-start_time:.2f} sec!\n")
    
    return 






def split_copy (sampled_df, RAW_DIR = "images_raw", TEST_DIR = "images_TEST", TRAIN_DIR = "images_TRAIN", POOL_DIR = "images_POOL"):
    from sklearn.model_selection import train_test_split

    #---------------------split no stratify---------------------
    test_ratio = 0.15
    train_ratio = 0.35
    pool_ratio = 0.50   # 不用直接用，最后自动得到

    # Step 1：POOL vs REST
    pool_df, rest_df = train_test_split(
        sampled_df,
        test_size=pool_ratio,
        # stratify=strat,
        random_state=42
    )

    train_ratio_updated = train_ratio / (1 - pool_ratio) 
    train_df, test_df = train_test_split(
        rest_df,
        train_size=train_ratio_updated,
        # stratify=strat,
        random_state=42
    )
    print(f"TEST vs TRAIN vs POOL :{len(test_df)}, {len(train_df)}, {len(pool_df)}")


    #----------------------COPY---------------------------
    # 自动创建文件夹
    for d in [TEST_DIR, TRAIN_DIR, POOL_DIR]:
        os.makedirs(d, exist_ok=True)

    def copy_images(df, target_dir):
        for id in df["host_id"]:
            id=str(id)
            filename=id+'.jpg'
            src = os.path.join(RAW_DIR, filename)
            dst = os.path.join(target_dir, filename)
            
            i=0
            if os.path.exists(src):
                if os.path.exists(dst):
                    print(f"[INFO] host {id} pic already copied: {dst}")
                else:
                    shutil.copy(src, dst)
            else:
                i+=1
                print(f"[WARNING {i}] host {id} pic not found: {src}")
    
    start_time=time.time()
    copy_images(test_df, TEST_DIR)
    copy_images(train_df, TRAIN_DIR)
    copy_images(pool_df, POOL_DIR)
    end_time=time.time()
    print(f"\n[SUCCES] Images copied from {RAW_DIR}to {TEST_DIR} /{TEST_DIR} /{POOL_DIR} in {end_time-start_time:.2f} sec!\n")
    
    #ls的输入json：
    for folder in [TEST_DIR, TRAIN_DIR, POOL_DIR]:
        ls_import_data(sampled_df,folder)

    return
















##=========================================clip EMBEDDINGS==========================================================##
import os
import json
import numpy as np
import torch
# import clip
from PIL import Image
import time
from tqdm import tqdm

# -----------------------------------------
# Load existing npz or create empty dict
# -----------------------------------------
def load_npz(path):
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        return dict(data.items())
    return {}

# -----------------------------------------
# Save npz safely
# -----------------------------------------
def save_npz(path, data_dict):
    np.savez(path, **data_dict)

# -----------------------------------------
# Vectorize one folder
# -----------------------------------------
def embed_folder(model, processor, device, folder_path, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        embeddings = load_npz(save_path)
    except FileNotFoundError:
        embeddings = {}

    files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    
    for fname in tqdm(files, desc=f"Embedding '{os.path.basename(folder_path)}' => '{save_path}'"):
        if fname in embeddings:
            print(f"[INFO] {fname} already embedded! SKIPPING!")
            continue
        
        img_path = os.path.join(folder_path, fname)
        try:
            img = Image.open(img_path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(device)
            img_input = inputs['pixel_values']

            with torch.no_grad():
                emb = model.get_image_features(img_input).cpu().numpy()[0]
                emb = emb.astype("float32")
                emb = emb / np.linalg.norm(emb)
            
            embeddings[fname] = emb
        except Exception as e:
            import traceback
            print(f"[ERROR] Failed: {fname}, error = {e}")
            traceback.print_exc()

    save_npz(save_path, embeddings)
    return list(embeddings.keys())

def embed_images_save_mapping(model, processor, device, images_folders=["images_TEST","images_TRAIN","images_POOL"], embeddings_folder='embeddings'):
    """
    folder : embeddings

    - embeddings_XXX.npz
    {
    "host_id01.jpg": [0.12, 0.51, ...],   # 512-d embedding
    "host_id02.jpg": [...],
    }

    - mapping.json
    {
    "TEST": ["hos_id.jpg", "host_id.jpg", ...],
    "TRAIN": [...],
    "POOL": [...]
    }

    """
    start_time=time.time()

    os.makedirs(embeddings_folder, exist_ok=True)

    mapping = {}
    for folder_path in images_folders:
        foldername=folder_path.split("_")[-1] 
        save_path= os.path.join(embeddings_folder, f"emb_{foldername}.npz")
        mapping[foldername]=embed_folder(model, processor, device, folder_path, save_path)
 
    with open(os.path.join(embeddings_folder, "mapping.json"), "w") as f:
        json.dump(mapping, f, indent=2)
    end_time=time.time()
    print(f"[SUCCES] All embeddings saved and mapped: {end_time-start_time:.2f} sec!")




##=============================PARSE MANUEL ANNOTATIONS================================================##
import json
import pandas as pd

# with open("label_studio_export.json") as f:
#     data = json.load(f)

# rows = []
# for item in data:
#     filename = item["data"]["filename"]
#     res = item["annotations"][0]["result"]
#     row = {"filename": filename}
#     for r in res:
#         row[r["from_name"]] = r["value"]["choices"][0]
#     rows.append(row)

# df_labels = pd.DataFrame(rows)
# print(df_labels.head())




