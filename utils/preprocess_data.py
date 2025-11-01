import pandas as pd
import os 
import time
import numpy as np



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
    print("len df MARS:",len(df6))
    print("len df JUIN:",len(df9))
    
    #合并数据集：
    df6['scraped_date'] = f"{year}Q1"
    df9['scraped_date'] = f"{year}Q2"
    dfparis = pd.concat([df6, df9], ignore_index=True)
    print(f"len df total : {len(dfparis)} \n")

    #增加status列：
    # 统计每个房东在每个时期的房源数量
    hosts6 = df6.groupby('host_id').size().reset_index(name='listings_06')
    hosts9 = df9.groupby('host_id').size().reset_index(name='listings_09')

    # 合并两个时期
    merged = hosts6.merge(hosts9, on='host_id', how='outer')

    # 分类:新房东，消失房东，listings增加/减少/不变
    merged['status'] = merged.apply(
        lambda r:
            'new_host' if pd.isna(r['listings_06']) else
            'disappeared_host' if pd.isna(r['listings_09']) else
            ('expanded' if r['listings_09'] > r['listings_06'] else
            'reduced' if r['listings_09'] < r['listings_06'] else
            'no_change'),
        axis=1
    )
    print(f"HOST STATUTS: {merged.status.value_counts()}\n")
    dfparis=dfparis.merge(merged[['host_id','status']], left_on='host_id', right_on='host_id',how='left')



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
        没改动/例外都被填为nan （包括所有Q2房东）
        
        新房东/新文本/新bio填1或sim

        """
        if row['scraped_date'] != f"{year}Q2":
            return None  # 只对 Q2 标记
        if pd.isna(row['host_about_q1']) and pd.notna(row['host_about']):
            return 0  #1?完全不一样 Q1 没有文本Q2 新增文本; 或填写文本的新房东
        if pd.notna(row['host_about_q1']) and pd.notna(row['host_about']):#老房东&bio不为空# row['host_about'] != row['host_about_q2']:
            sim= text_similarity(row['host_about_q1'], row['host_about']) 
            if sim < threshold:## 老文本有明显变化
                return sim 
            else:
                return None#无明显改变是就填NAN
        return None  # 其他情况

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

    # 标记 Q3 相对于 Q2 的变化
    def host_picture_url_change(row):
        if row['scraped_date'] !=f"{year}Q2":
            return None # 只对 Q2 标记
        if pd.isna(row['host_picture_url_q1']) and pd.notna(row['host_picture_url']):
            return 1  # 新增照片
        if pd.notna(row['host_picture_url_q1']) and pd.notna(row['host_picture_url']) and row['host_picture_url'] != row['host_picture_url_q1']:
            return 1  # 修改照片
        return None  # 其他情况

    if "host_picture_url_changed" not in dfparis:
        dfparis['host_picture_url_changed'] = dfparis.apply(host_picture_url_change, axis=1)

    # 查看 Q3 变化统计
    print("HOST PICTURE CHANGE :")
    print(dfparis.host_picture_url_changed.notna().value_counts())
    print(dfparis.host_picture_url_changed.value_counts(),"\n")


    #====================SPLIT=====================
    df_change=dfparis[(dfparis['status']=='new_host')|(dfparis['host_about_changed']==1)|(dfparis['host_picture_url_changed']==1)]
    df_stable = dfparis[~dfparis['host_id'].isin(df_change['host_id'])]#.isin() 来做“是否在某个列表中”的列级比较，然后加 ~ 表示取反。
    
    print(f"len change: {len(df_change)}")
    print(f"len stable: {len(df_stable)} \n")
    print("HOST STATUS IN CHANGE/STABLE:")
    print(df_change.status.value_counts()/len(df_change))
    print(df_stable.status.value_counts()/len(df_stable))

    return dfparis, df_change, df_stable









def desc_stat(df):
    






























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