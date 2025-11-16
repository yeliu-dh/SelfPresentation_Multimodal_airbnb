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

## STRATIFIED SAMPLING


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








def split_copy (test_df, train_df, pool_df, RAW_DIR = "images_raw", TEST_DIR = "images_TEST", TRAIN_DIR = "images_TRAIN", POOL_DIR = "images_POOL"):
    # 自动创建文件夹
    for d in [TEST_DIR, TRAIN_DIR, POOL_DIR]:
        os.makedirs(d, exist_ok=True)

    def copy_images(df, target_dir):
        for img in df["host_id"]:
            src = os.path.join(RAW_DIR, img)
            dst = os.path.join(target_dir, img)

            if os.path.exists(src):
                shutil.copy(src, dst)
            else:
                print(f"[WARNING] File not found: {src}")

    copy_images(test_df, TEST_DIR)
    copy_images(train_df, TRAIN_DIR)
    copy_images(pool_df, POOL_DIR)

    print(f"[SUCCES] Images copied from {RAW_DIR}to {TEST_DIR} /{TEST_DIR} /{POOL_DIR}!")
    return















##=====================================================================================================##

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128):
        super().__init__()
        # shared layers
        self.shared_fc = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

        # heads for each task
        self.type_head = nn.Linear(hidden_dim, 3)       # Type: life/pro/else
        self.quality_head = nn.Linear(hidden_dim, 1)    # Quality: high/low
        self.smile_head = nn.Linear(hidden_dim, 1)      # Is_smiling: yes/no

    def forward(self, x):
        x = self.relu(self.shared_fc(x))
        type_out = self.type_head(x)          # logits for softmax
        quality_out = torch.sigmoid(self.quality_head(x))  # probability 0-1
        smile_out = torch.sigmoid(self.smile_head(x))      # probability 0-1
        return type_out, quality_out, smile_out

# 示例训练 loop 的损失
def multi_task_loss(type_logits, type_labels, quality_pred, quality_labels, smile_pred, smile_labels):
    type_loss = F.cross_entropy(type_logits, type_labels)
    quality_loss = F.binary_cross_entropy(quality_pred, quality_labels.float())
    smile_loss = F.binary_cross_entropy(smile_pred, smile_labels.float())
    return type_loss + quality_loss + smile_loss
