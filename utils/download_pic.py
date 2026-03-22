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
    # 正常下载，返回path_img, 无效url返回None

    if not check_pic_url(url, timeout=timeout):
        print(f"\n[WARNING] 无效 URL: {url}")
        return None

    os.makedirs(out_dir, exist_ok=True)

    filename= f"{str(id)}.jpg"
    out_path = os.path.join(out_dir, filename)


    # 防止覆盖已有文件
    if os.path.exists(out_path):
        # print(f"[INFO] host {id} pic EXISTS, SKIPPING: {out_path}")
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


def get_id_url_list(df):
    """
    host_id:先int再str


    """
    for c in ["host_id",'host_picture_url']:
        if c not in df.columns:
            print(f"[warning!] {c} not in df!!!\n")
            break
    id_url_list=[(str(int(row['host_id'])), row["host_picture_url"]) for _, row in df.iterrows()]

    return id_url_list



# 批量并行下载函数
def download_images_batch(path_df,out_dir='images/paris_2306', max_workers=10):
    """

    """
    start_time=time.time()
    os.makedirs(out_dir, exist_ok=True)

    # ---data---
    df=pd.read_csv(path_df)
    df_pic=df.copy()
    df_pic=df_pic[df_pic['host_has_profile_pic']=="t"]
    df_pic=df_pic.dropna(subset=["host_picture_url"]).drop_duplicates(subset="host_picture_url")
    
    if "host_pic_is_valid" in df.columns:
      print(f"[check]{df_pic.host_pic_is_valid.value_counts(dropna=False)}")
      print(f"[info] filtered by 'host_pic_is_valid'!")
      df_pic=df_pic[df_pic['host_pic_is_valid'].isna()]
    print(f"[check] has pic+ dropna + drop_duplicates: {len(df)} => {len(df_pic)}!")
    
    if len(df_pic)>0:
      # ---pending---
      # 图片文件名和df中的host_id必须都是仅整数INT！
      processed=[int(os.path.splitext(f)[0]) for f in os.listdir(pic_folder) if os.path.splitext(f)[0].isdigit()]#存在重复副本！
      pending_df=df_pic[~df_pic['host_id'].isin(processed)]

      id_url_list=get_id_url_list(pending_df)# get str(int(host_id)):url
      print(f"[info] {len(processed)}/{len(df_pic)} downloaded; {len(pending_df)} /{len(df_pic)} to download!\n")

      # ---save---
      # oi适合多线程
      failed=list(df[df['host_has_profile_pic']=='f']['host_id'])+list(df[df['host_pic_is_valid'].isna()]['host_id'])
      print(f"[info] no (valid) host pic: {len(failed)}/{len(df_pic)} failed\n")
      
      with ThreadPoolExecutor(max_workers=max_workers) as executor:
          futures = {executor.submit(download_image, id, url, out_dir): id for id, url in id_url_list}#***

          for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading images..."):
              host_id = futures[fut]
              try:
                  path = fut.result()
                  if path:
                    processed.append(int(host_id))
                  else :
                    failed.append(int(host_id))

              except Exception as e:
                  failed.append(host_id)
                  
      end_time=time.time()
      print(f"\n[DONE] {len(id_url_list)} pics downloaded : {end_time-start_time:.2f} sec!\n")
    

      # ---host_pic_is_valid---
      # 每次下载一部分更新is_valid标记
      df['host_pic_is_valid'] = df['host_id'].apply(
          lambda x: True if x in processed else (False if x in failed else None)
      )

      print(f"[update] {df['host_pic_is_valid'].value_counts(dropna=False)}")

      # df['host_pic_is_valid']=df['host_id'].apply(lambda x : True if x in processed elif  )
      df.to_csv(path_df, index=False)
      print(f"[save] df with 'host_pic_is_valid' updated saved to {path_df}.\n")
    
    else :
      print(f'[info] all downloaded!')

    return df
