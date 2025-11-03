import pandas as pd
import os 
import time
import numpy as np

import time


def unzip_csv_gz(folder='raw_data', output_folder='data'):
    os.makedirs(output_folder, exist_ok=True)

    for i, filename in enumerate(os.listdir(folder)):
        file_path = os.path.join(folder, filename)

        # 只处理 .gz 文件
        if not filename.endswith('.gz'):
            print(f"{i} {filename} is not a gzip file, skipped.\n")
            continue

        filename_clean = os.path.splitext(filename)[0]  # 去掉 .gz
        file_outpath = os.path.join(output_folder, filename_clean)

        # 如果输出文件已存在，跳过
        if os.path.exists(file_outpath):
            print(f"{i} {filename_clean} already exists in {output_folder}!\n")
            continue

        try:
            df = pd.read_csv(file_path, compression='gzip', encoding='utf-8')
            df.to_csv(file_outpath, index=False)
            print(f"{i} ✔ {filename_clean} converted and saved in {output_folder}!\n")
        
        except Exception as e:
            print(f"{i} ❌ Error converting {filename}: {e}\n")


def split_change_stable(df6, df9, year='2025'):
    start_time=time.time()

    print("len df MARS:",len(df6))
    print("len df JUIN:",len(df9))
    
    #合并数据集：
    df6['scraped_date'] = f"{year}Q1"
    df9['scraped_date'] = f"{year}Q2"
    dfparis = pd.concat([df6, df9], ignore_index=True)
    dfparis['host_since']=pd.to_datetime(dfparis['host_since'], errors="coerce")
    print(f"len df total : {len(dfparis)} \n")

    #增加status列：
    # 仍然要统计每一个季度的listings数量，来识别新进入和流出房东
    hosts6 = df6.groupby('host_id').size().reset_index(name='Q1')
    hosts9 = df9.groupby('host_id').size().reset_index(name='Q2')
    merged = hosts6.merge(hosts9, on='host_id', how='outer')
    

    # 分类:新房东，消失房东，listings增加/减少/不变
    merged['status'] = merged.apply(
        lambda r:
            'reactive_host' if pd.isna(r['Q1']) else
            'deactived_host' if pd.isna(r['Q2']) else
            "old_host",
            # ('expanded' if r['Q2'] > r['Q1'] else
            # 'reduced' if r['Q2'] < r['Q1'] else
            # 'no_change'),
        axis=1
    )

    # print(f"HOST STATUTS: {merged.status.value_counts()}\n")
    dfparis=dfparis.merge(merged[['host_id','status']], left_on='host_id', right_on='host_id',how='left')

    # 根据host_since 细分new和reactive：纯向量操作（快十几倍，不需要 apply），同下效果。
    mask = (
        (dfparis['status'] == 'reactive_host') &
        (dfparis['host_since'].between('2024-01-01', '2024-06-30'))
    )
    dfparis.loc[mask, 'status'] = 'new_host'
    
    # start_date, end_date = pd.to_datetime('2025-04-01'), pd.to_datetime('2025-06-30')
    # dfparis['status'] = dfparis.apply(
    #     lambda r: (
    #         "new_host"
    #         if (r['status'] == "reactive_host")
    #         and (pd.notna(r['host_since']))
    #         and (start_date <= r['host_since'] <= end_date)
    #         else r['status']
    #     ),
    #     axis=1
    # )

    # 统计房东状态，但只看Q2避免，Q1Q2重复记录
    print("※ HOST STAUTS CHANGE (Q2): \n", dfparis[dfparis['scraped_date']==f'{year}Q2']['status'].value_counts(dropna=False),"\n")


    #描述房源数量变化
    print("LISTINGS CHANGE:")
    hosts_in = set(df9['id']) - set(df6['id']) # 9月有，但6月没有
    hosts_out = set(df6['id']) - set(df9['id']) #6月有，但9月没有
    hosts_io= set(df9['id']) ^ set(df6['id']) # 属于 6月 或 9月，但不同时属于两者;所有变化的房东（新增 + 消失）
    print(f'new listings after JO : {len(hosts_in)}({len(hosts_in)*100/len(df9):.2f}%)')
    print(f'listings dispeared after JO : {len(hosts_out)}({len(hosts_out)*100/len(df9):.2f}%)')
    print(f"listings changed during JO :{len(hosts_io)}\n")



    #================增加host_about_q2, host_about_change列=========================
    if 'host_about_q1' not in dfparis:
        df_q1 = dfparis[dfparis['scraped_date'] == f"{year}Q1"][['host_id', 'id', 'host_about']]
        df_q1 = df_q1.rename(columns={'host_about': 'host_about_q1'})
        dfparis = dfparis.merge(df_q1, on=['host_id', 'id'], how='left')
        # print(f"q1 : {dfparis.host_about_q1.notna().value_counts()}")

    # 计算文本相似度（使用 difflib.SequenceMatcher）
    from difflib import SequenceMatcher
    def text_similarity(a, b):
        if pd.isna(a) or pd.isna(b):
            return None
        return SequenceMatcher(None, str(a), str(b)).ratio()

    # 标记 Q2 相对于 Q1 的变化
    def host_about_change(row, threshold=0.85):
        """
        没改动/例外都被填为nan （包括所有Q1房东）
        
        新房东/新文本/新bio填1或sim
        """

        #虽然计算sim，但是所有标记成：变化1，不变0。仅用于筛选

        if row['scraped_date'] != f"{year}Q2":
            return 0  # 只对 Q2 标记，Q1均为0
        if pd.isna(row['host_about_q1']) and pd.notna(row['host_about']):
            return 1  # 新增文本; （新房东或之前未填写文本的房东）
        if pd.notna(row['host_about_q1']) and pd.notna(row['host_about']):#老房东&bio不为空# row['host_about'] != row['host_about_q2']:
            sim= text_similarity(row['host_about_q1'], row['host_about']) 
            if sim < threshold:## 老文本有明显变化
                return 1 
            else:
                return 0 #无明显改变是就填NAN
        return 0 # 未变&其他情况

    if "host_about_changed" not in dfparis:
        dfparis['host_about_changed'] = dfparis.apply(host_about_change, axis=1)

    # 查看 Q2 中变化统计：没改动都被填为nan，新房东/新文本/新bio填1或sim:
    print("HOST ABOUT CHANGE :")
    print(dfparis.host_about_changed.notna().value_counts())
    print(dfparis.host_about_changed.value_counts(),"\n")


    #================增加host_picture_url_q1, host_picture_url_change列=========================
    if "host_picture_url_q1" not in dfparis:
        df_q1 = dfparis[dfparis['scraped_date'] == f"{year}Q1"][['host_id', 'id', 'host_picture_url']]
        df_q1 = df_q1.rename(columns={'host_picture_url': 'host_picture_url_q1'})
        dfparis = dfparis.merge(df_q1, on=['host_id', 'id'], how='left')

    # 标记 Q2 相对于 Q1 的变化
    def host_picture_url_change(row):
        if row['scraped_date'] !=f"{year}Q2":
            return 0 # 只对 Q2 标记, Q1均为0
        if pd.isna(row['host_picture_url_q1']) and pd.notna(row['host_picture_url']):
            return 1  # 新增照片
        if pd.notna(row['host_picture_url_q1']) and pd.notna(row['host_picture_url']) and row['host_picture_url'] != row['host_picture_url_q1']:
            return 1  # 修改照片
        return 0 # 没变&其他情况

    if "host_picture_url_changed" not in dfparis:
        dfparis['host_picture_url_changed'] = dfparis.apply(host_picture_url_change, axis=1)

    # 查看 Q2 变化统计
    print("HOST PICTURE CHANGE :")
    print(dfparis.host_picture_url_changed.notna().value_counts())
    print(dfparis.host_picture_url_changed.value_counts(),"\n")


    #====================SPLIT=====================
    #筛选出Q2所有变化的房东：
    dfparisQ2=dfparis[dfparis['scraped_date'] == f"{year}Q2"]
    def presentation_change_level(row):
        # 数值计算更加简单、可靠!
        bio_change = 1 if row.get('host_about_changed') ==1 else 0
        pic_change = 1 if row.get('host_picture_url_changed') == 1 else 0
        return bio_change+pic_change  #记录为数组 或加减数值 0,1,2 分别代表不同程度变化

    #在整个数据上应用：
    dfparisQ2['presentation_change'] = dfparisQ2.apply(presentation_change_level, axis=1)
    print(f"HOST IM global : {dfparisQ2.presentation_change.value_counts()} \n")
    # print(f"HOST IM CHANGE :\n {dfparisQ2[['host_about_changed','host_picture_url_changed']].value_counts(dropna=False)}\n")


    # split

    dfparisQ2['is_changed'] = (
        (dfparisQ2['status'].isin(['new_host', 'reactive_host'])) |
        (dfparisQ2['presentation_change'] > 0)
    )    #返回T/F
    df_change = dfparisQ2[dfparisQ2['is_changed']]
    df_stable = dfparisQ2[~dfparisQ2['is_changed']]

    # df_change = dfparisQ2[
    #     (dfparisQ2['status'].isin(['new_host', 'reactive_host'])) |
    #     dfparisQ2['presentation_change']> 0 #改变bio/pic
    # ]
    # df_stable = dfparisQ2[~dfparisQ2['host_id'].isin(df_change['host_id'])]#.isin() 来做“是否在某个列表中”的列级比较，然后加 ~ 表示取反。
    #按照host_id筛选有风险，会排除掉一个房东的多个房源，只取第一个




    print(f"※ len change: {len(df_change)}")
    print(f"※ len stable: {len(df_stable)} \n")

    #房东市场行为变化统计：
    print("※ HOST STATUS CHANGE:")
    print(df_change.status.value_counts()/len(df_change),'\n')
    # print(df_stable.status.value_counts()/len(df_stable),"\n")#tjs old_host

    #房东自我展示变化统计：
    print("※ HOST IM CHANGE:")
    print(df_change .presentation_change.value_counts()/len(df_change))
    print(f"※ HOST IM CHANGE (exclude no_change):\n {df_change[['host_about_changed','host_picture_url_changed']].value_counts(dropna=False)}\n")

    # print(df_stable.presentation_change.value_counts()/len(df_stable),"\n")# tjs0



    # ================END====================
    end_time=time.time()
    print(f"Temps d'exécution :{end_time-start_time:.2f} sec.")
    return dfparis, df_change, df_stable








# def desc_stat(df):
    






























def compute_booking_rate(row):
    if row['has_availability'] == 'f' or row['availability_90'] == 0:#90 一个季度内不活跃
        return 0.0  # 下架或不活跃
    elif row['availability_30'] == 0:
        return 1.0  # 满房
    elif row['availability_30'] > 0:
        return min(row['number_of_reviews_l30d'] / row['availability_30'], 1.0)
    else:
        return None  # 其他缺失情况


def classify_activity(row):
    if row['has_availability'] == 'f' or row['availability_90'] == 0:
        return 'inactive'
    elif row['availability_30'] == 0:
        return 'full_booked'
    else:
        return 'active'
    




def pricestr2float(df):
    df["price"] = df["price"].str.replace(r'[$,]', '', regex=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    print(df.price.describe(include='all'))
    print(df.price.notna().value_counts())
    return


'''
room_type:
[Entire home/apt|Private room|Shared room|Hotel]

All homes are grouped into the following three room types:

Entire place
Private room
Shared room
Entire place
Entire places are best if you're seeking a home away from home. With an entire place, you'll have the whole space to yourself. This usually includes a bedroom, a bathroom, a kitchen, and a separate, dedicated entrance. Hosts should note in the description if they'll be on the property or not (ex: "Host occupies first floor of the home"), and provide further details on the listing.

Private rooms
Private rooms are great for when you prefer a little privacy, and still value a local connection. When you book a private room, you'll have your own private room for sleeping and may share some spaces with others. You might need to walk through indoor spaces that another host or guest may occupy to get to your room.

Shared rooms
Shared rooms are for when you don't mind sharing a space with others. When you book a shared room, you'll be sleeping in a space that is shared with others and share the entire space with other people. Shared rooms are popular among flexible travelers looking for new friends and budget-friendly stays.
'''



def categorize_property(ptype):
    if pd.isna(ptype) or str(ptype).strip() == "":
        return "others"
    ptype_lower = str(ptype).lower()

    # ENTIRE
    if any(word in ptype_lower for word in ["entire", "condo", "loft", "apartment"]):
        return "entire"

    # HOTEL
    elif "hotel" in ptype_lower:
        return "hotel"

    # SHARED
    elif any(word in ptype_lower for word in ["shared", "bed and breakfast", "boutique"]):
        return "shared"

    # PRIVATE
    elif "private" in ptype_lower:
        return "private"

    else:
        return "others"
    
    

"""
reviews_per_month: 	
The average number of reviews per month the listing has over the lifetime of the listing.

Psuedocoe/~SQL:

IF scrape_date - first_review <= 30 THEN number_of_reviews
ELSE number_of_reviews / ((scrape_date - first_review + 1) / (365/12))


"""