import pandas as pd
import time
import os, sys
import requests
from requests.exceptions import RequestException
import time
from tqdm import tqdm
tqdm.pandas()  # 这行让 pandas 的 apply() 支持进度条

from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed




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

    #filename
    # parsed = urlparse(url)
    # filename = os.path.basename(parsed.path)
    # if not filename:
    #     filename = f"{hash(url)}.jpg"
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
    print(f"\n [SUCCES] downloaded {len(id_url_list)} : {end_time-start_time:.2f} sec!\n")

    return results



