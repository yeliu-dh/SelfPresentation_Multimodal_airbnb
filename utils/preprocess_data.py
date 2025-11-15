import pandas as pd
import os 
import time
import numpy as np
import time


##==================================UNZIP============================================##
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














##==================================SPLIT============================================##


def split_change_stable(df6, df9, year='2025'):
    start_time=time.time()
    print("===================================INPUT DATA===================================")

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
    print("===================================APRCU GLOBAL===================================")
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
    print("HOST ABOUT CHANGE (change:1, no_change:0) :")
    # print(dfparis.host_about_changed.notna().value_counts())#没有nan，例外也填0
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
    print("HOST PICTURE CHANGE (change:1, no_change:0):")
    print(dfparis.host_picture_url_changed.notna().value_counts())
    print(dfparis.host_picture_url_changed.value_counts(),"\n")


    #====================SPLIT=====================
    #筛选出Q2所有变化的房东：!!!!!!!!!
    dfparisQ2=dfparis[dfparis['scraped_date'] == f"{year}Q2"]
    def presentation_change_level(row):
        # 数值计算更加简单、可靠!
        bio_change = 1 if row.get('host_about_changed') ==1 else 0
        pic_change = 1 if row.get('host_picture_url_changed') == 1 else 0
        return bio_change+pic_change  #记录为数组 或加减数值 0,1,2 分别代表不同程度变化

    #在整个数据上应用：
    dfparisQ2['presentation_change'] = dfparisQ2.apply(presentation_change_level, axis=1)
    print(f"HOST IM global (no_change:0, 1change :1, 2change:2): \n {dfparisQ2.presentation_change.value_counts()} \n")
    # print(f"HOST IM CHANGE :\n {dfparisQ2[['host_about_changed','host_picture_url_changed']].value_counts(dropna=False)}\n")


    # split
    # 统计new_host| reactive_host | host_about_change| host_pic_change的行：
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
    # 按照host_id筛选有风险，会排除掉一个房东的多个房源，只取第一个


    print("===================================SUMMARY ON CHANGE===================================")
    print("CHANGE == new_host| reactive_host | host_about_change| host_pic_change")
    print(f"※ len change: {len(df_change)}")
    print(f"※ len stable: {len(df_stable)} \n")

    #房东市场行为变化统计：
    print("※ HOST STATUS CHANGE dans df_change:")
    print(df_change.status.value_counts()/len(df_change),'\n')
    # print(df_stable.status.value_counts()/len(df_stable),"\n")#tjs old_host

    #房东自我展示变化统计：
    print("※ HOST IM CHANGE dans df_change:")
    print(df_change .presentation_change.value_counts()/len(df_change),'\n')
    print(f"※ DETAILS :\n {df_change[['host_about_changed','host_picture_url_changed']].value_counts(dropna=False)}\n")

    # print(df_stable.presentation_change.value_counts()/len(df_stable),"\n")# tjs0



    # ================END====================
    end_time=time.time()
    print(f"✅ Temps d'exécution :{end_time-start_time:.2f} sec.")
    return dfparisQ2, df_change, df_stable





def save_global_change_stable_csv(csv_files, year:str, location:str, output_folder="data_jo_processed"):
    
    print("NB. csv files enter strictly in order : global-change-stable!")
    print(f"default output folder : {output_folder}")

    os.makedirs(output_folder, exist_ok=True)

    for i, f in enumerate(csv_files):
        if i==0 :
            output_path=f"{output_folder}\listings_jo_{location}{year}.csv"
        elif i==1:
            output_path=f"{output_folder}\listings_jo_{location}{year}_change.csv"
        elif i==2:
            output_path=f"{output_folder}\listings_jo_{location}{year}_stable.csv"
        f.to_csv(output_path, index=False)
        print(f"✔ csv saved in {output_path}!")

    return 






















##=============================DESC STAT=====================================##

def desc_catORnum(df, vars):

    print("================= BALN PROCESSED VARIABLES =====================")
    for var in vars:
        col = df[var]
        # nunique = col.nunique(dropna=False)
        # print(f"\n>>> {var} ({col.dtype}), unique={nunique}")

        # ① 分类变量检测规则
        is_categorical = (
            col.dtype == "object" or                 # 字符串
            col.nunique(dropna=False) <= 5 or                          # 值太少
            set(col.unique()) <= {0,1}               # 明确是二元 dummy
        )

        # ② 输出方式
        if is_categorical:
            print(col.value_counts(dropna=False).sort_values(ascending=False), "\n")
        else:
            print(col.describe(include="all"), "\n")

        print("-----------------------------------------------------------")
    return 





##==================================HOST VARS============================================##

import langid #这个库速度更快、稳定性高。
def detect_language_langid(text):
    try:
        if isinstance(text, str) and text.strip():
            lang, _ = langid.classify(text)
            if lang=="en" or lang=="fr":
                return lang
            else :
                return "other_langs"
        else:
            return "no_text"  # 空值或非字符串
        

    except Exception:
        return "unk"  # 检测失败的情况





def preprocess_host_variables(df):
    
    var_ok=["host_has_profile_pic","host_picture_url","host_identity_verified","host_name","number_of_reviews"]

    var_toprocess=["host_is_superhost","review_scores_rating", "host_since","host_about",
                "host_response_time","host_response_rate","calculated_host_listings_count"]
    
    print(f"\n\n******************************HOST VARS******************************\n"
          f"PROCESS METHODS :\n"
          f"- host_is_sueprhost: fillna('f')\n"
          f"- review_scores_rating: 缺失严重，新增一列, 'has_rating:1/0'\n"
          f"- host_since: 新增1列years_since_host :float, 按照ab页面显示计算年数，0.5-1年填1， 0-0.5年填0 \n"
          f"- has_host_about:新增3列'has_host_about:1/0','lang:en/fr/other_langs','len:int',\n"
          f"- host_response_time: fillna('no_response_time') \n"
          f"- host_response_rate:缺失严重，增加一列has_response_rate:1/0， fillna(0)\n"
          f"- calculated_host_listings_count : 新增1列'professional_host:1/0'\n"
          )
    
    for var in var_toprocess:
        if not var in df.columns :
            print(f"[WARNING] {var} not found in df!")
        else :
            if var=='host_is_superhost':
                # host_is_superhost :对缺失superhost的填f
                df['host_is_superhost']=df['host_is_superhost'].fillna('f')

            elif var=='review_scores_rating':
                #review_scores_rating:对缺失评分（数值）增加一列has_rating：有为1，无为0;
                # 双变量法：相当于把有/无评分的分开，所以原列中可以fillna(0); 区分信号存在/强度
                df['has_rating'] = df['review_scores_rating'].notna().astype(int)
                df['review_scores_rating'] = df['review_scores_rating'].fillna(0)

            elif var=='host_since':
                # host_since:增加一列years_since_host，少于半年填0，少于1年填1
                def add_years_since_host(df, date_col='host_since', scrape_date='2024-06-12'):
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    today = pd.Timestamp(scrape_date) if scrape_date else pd.Timestamp.today()

                    # 计算距今天数
                    df['days_since_host'] = (today - df[date_col]).dt.days

                    # 分段换算成年限
                    def convert_days_to_years(days):
                        if pd.isna(days):
                            return None
                        years = days / 365.25
                        if years < 0.5:
                            return 0
                        elif years < 1:
                            return 1
                        else:
                            return round(years)

                    df['years_since_host'] = df['days_since_host'].apply(convert_days_to_years)

                    # 缺失信号与填充
                    if int(df.host_since.isna().sum())!=0:
                        df['has_since'] = df[date_col].notnull().astype(int)
                        df['years_since_host'] = df['years_since_host'].fillna(0)

                    return df
                df = add_years_since_host(df, date_col='host_since', scrape_date='2024-06-12')
            
            
            elif var=='host_about':
                # host_about：增加两列：has_host_about,lang,len
                df['has_host_about'] = df['host_about'].notna().astype(int)
                df['lang'] = df['host_about'].apply(detect_language_langid)

                # df['host_about'] = df['host_about'].fillna("nan")#无文本保留nan？
                df['len']=df['host_about'].apply(lambda x : len(x.split()) if isinstance(x, str) else 0)
            

            elif var=='host_response_time':
                # host_response_time（类别） 缺失值填 no_response_time?
                df["host_response_time"]=df["host_response_time"].fillna('no_response_time')
            
            elif var=='host_response_rate':
                # host_response_rate（数值） 缺失值新增一列has_response_rate：有为1，无为0;所有值变成float？
                df['has_response_rate'] = df['host_response_rate'].notna().astype(int)
                df['host_response_rate'] = df['host_response_rate'].fillna(0)
                df['host_response_rate'] = pd.to_numeric(df["host_response_rate"], errors="coerce")
            
            elif var=="calculated_host_listings_count":
                df["professional_host"]=df["calculated_host_listings_count"].apply(lambda x : 0 if x ==1 else 1)

            else :
                print(f"[INFO] {var} hasn't specific traitement!")
    
    # 手动整理添加新增cols
    var_processed=["host_is_superhost",
                "review_scores_rating","has_rating", 
                "host_since","years_since_host",
                "host_about","has_host_about", "lang","len",
                "host_response_time",
                "host_response_rate","has_response_rate",
                "calculated_host_listings_count", "professional_host"]
    

    print(f"==========================HOST VARS================================")
    impo_vars=["host_picture_url","host_identity_verified","number_of_reviews"]+var_processed
    print(f"{len(impo_vars)} IMPO VARS:{impo_vars}\n") 


    # print(f"===============BIALN PROCESSED VARIABLES=====================")
    desc_catORnum(df, vars=var_processed)

    # for var in var_processed:
    #     print(df[var].dtype)
    #     if df[var].dtype =="int64" or df[var].dtype =="float64":#!="object"
    #         print(df[var].describe(include='all'),"\n")
    #     else :             
    #         print(df[var].value_counts(dropna=False).sort_values(ascending=False),"\n")
    #     print("-----------------------------------------------------------")


    # delect intermidate cols:optionnal
    # intermediate_cols=[]
    df=df.drop(columns=['days_since_host'])#?
    print(f"Intermediate cols deleted: days_since_host")

    return df













##==================================PROXY============================================##

def filter_by_proxy(df,proxy_vars=['price',"availability_90"]):
    print(
        f"\n\n==============================PROXY=============================\n"
        # f"FILTRER METHODS :\n"
        # f"nettoyer + dropna sur les vars : {', '.join(proxy_vars)} \n"
        )

    ## process:
    for var in proxy_vars:
        if var == 'price' and var in df.columns :
            df['price'] = (
                df['price']
                .replace('[\$,]', '', regex=True)  # 去掉 $ 和 ,
                .astype(float)                     # 转成数值
            )
    # describe :
    print(f"---------------------PROXY (before filtrated)---------------------")

    for var in proxy_vars:
        print(df[var].dtype)
        if df[var].dtype =="int64" or df[var].dtype =="float64":#!="object"
            print(df[var].value_counts(dropna=False),"\n")
            print(df[var].describe(include='all'),"\n")

        else :             
            print(df[var].value_counts(dropna=False),"\n")
        print("-----------------------------------------------------------")

    # filtrer :
    df_filtred=df.copy()
    for var in proxy_vars :
        len_before=len(df_filtred)
        df_filtred=df_filtred[df_filtred[var].notna()]
        len_after=len(df_filtred)
        print(f"- {len_before-len_after} dropped in {var}")

    print(f"len avant de filtrer {', '.join(proxy_vars)}: {len(df)}")
    print(f"apres: {len(df_filtred)}\n")
   
    return df, df_filtred





##==================================LOCATION============================================##

def add_is_within_km(df, threshold_km=3):
    
    venues_df = pd.DataFrame([
        {"venue": "Stade de France", "lat": 48.9244, "lon": 2.3601},
        {"venue": "Paris Aquatic Centre", "lat": 48.9235, "lon": 2.3554},   # 根据维基坐标 :contentReference[oaicite:1]{index=1}
        {"venue": "Porte de La Chapelle Arena", "lat": 48.8970, "lon": 2.3620},  # 根据来源 :contentReference[oaicite:2]{index=2}
        {"venue": "Paris La Défense Arena", "lat": 48.8915, "lon": 2.2370},   # 来源示例 :contentReference[oaicite:3]{index=3}
        {"venue": "Bercy Arena", "lat": 48.8386, "lon": 2.3781},  # 来源 :contentReference[oaicite:4]{index=4}
        {"venue": "Champ de Mars (Eiffel Tower area)", "lat": 48.8558, "lon": 2.2983},  # 来源 :contentReference[oaicite:5]{index=5}
        {"venue": "Grand Palais", "lat": 48.8660, "lon": 2.3117},   # 来源 :contentReference[oaicite:6]{index=6}
        {"venue": "Les Invalides", "lat": 48.8565, "lon": 2.3124},   # 来源 :contentReference[oaicite:7]{index=7}
        {"venue": "Palace of Versailles", "lat": 48.8059, "lon": 2.1162},   # 来源 :contentReference[oaicite:8]{index=8}
        # …你可以继续补充其他场馆
    ])

    print(f"\n\n ==============================LOCATION=============================\n"
            # f"CALCULATION METHODS :\n"
            # f"-'latitude','longitude': \n 计算房源到各大主要venue的距离，果最小值<=5km, 则在is_within_5km上填't',反之'f'"
            # f"venues_df :\n {venues_df}\n"
            )
    
    # 确保坐标数值化
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    venues_df['lat'] = pd.to_numeric(venues_df['lat'], errors='coerce')
    venues_df['lon'] = pd.to_numeric(venues_df['lon'], errors='coerce')

    # 定义 haversine 函数（公里）
    def haversine(lat1, lon1, lat2, lon2):
        if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
            return np.nan
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        return R * 2 * np.arcsin(np.sqrt(a))

    # 计算每个房源到所有场馆的最小距离
    def min_distance(lat, lon):
        distances = [
            haversine(lat, lon, v_lat, v_lon)
            for v_lat, v_lon in zip(venues_df['lat'], venues_df['lon'])
            if not pd.isna(v_lat) and not pd.isna(v_lon)
        ]
        return np.nanmin(distances) if len(distances) > 0 else np.nan

    df['dist_to_venue_min'] = df.apply(lambda row: min_distance(row['latitude'], row['longitude']), axis=1)
    df[f'is_within_{threshold_km}km'] = (df['dist_to_venue_min'] <= threshold_km).astype(int)
    
    # inter
    df=df.drop(columns=['dist_to_venue_min'])

    # desc
    print(df[f'is_within_{threshold_km}km'].value_counts(dropna=False))

    return df


##==================================OBJ VAR ============================================##
# 集合proxy+location处理
# 描述！

def preprocess_obj_vars(df, proxy_vars=['price',"availability_90"], obj_vars=["room_type", "minimum_nights","instant_bookable"], threshold_km=3):
    # obj_vars=["room_type", "minimum_nights","instant_bookable"]#all ok,无缺失/异常
    # proxy_vars=['price',"availability_90"]
    
    print(f"\n\n**************************PROXY + OBJ VARS******************************\n"
          f"PROCESS METHODS :\n"
          f"1) process proxies :{', '.join(proxy_vars)}: \n"
          f"  nettoyer + dropna sur les vars : {', '.join(proxy_vars)} \n"
          f"  ==> get df_filtered \n\n"
          f"2) desc statistique :{', '.join(obj_vars)} \n\n"
          f"3) location :'latitude','longitude': 新增一列'is_within_Xkm'\n"
          f"  计算房源到各大主要venue的距离，果最小值<=5km, 则在is_within_3km上填1,反之0 \n"
        )
                
    #proxy
    df, df_filtered=filter_by_proxy(df, proxy_vars=['price',"availability_90"])

    # 4 obj vars:
    # print(f"\n\n=============================OBJ VARS=============================")
    desc_catORnum(df, vars=obj_vars)

    # for var in obj_vars:
    #     # print(f"var : {var}, type : {df[var].dtype}")
    #     if df[var].dtype =="int64" or df[var].dtype =="float64":#!="object"数字型
    #         print(f"NAN:{df[var].isna().sum()}")
    #         print(df[var].describe(include='all'),"\n")
    #     else :
    #         print(f"NAN:{df[var].isna().sum()}")             
    #         print(df[var].value_counts().sort_values(ascending=False),"\n")
    #     print("-----------------------------------------------------------")

    # location:
    df_filtered=add_is_within_km(df_filtered,threshold_km=3)
    return df_filtered













# def compute_booking_rate(row):
#     if row['has_availability'] == 'f' or row['availability_90'] == 0:#90 一个季度内不活跃
#         return 0.0  # 下架或不活跃
#     elif row['availability_30'] == 0:
#         return 1.0  # 满房
#     elif row['availability_30'] > 0:
#         return min(row['number_of_reviews_l30d'] / row['availability_30'], 1.0)
#     else:
#         return None  # 其他缺失情况


# def classify_activity(row):
#     if row['has_availability'] == 'f' or row['availability_90'] == 0:
#         return 'inactive'
#     elif row['availability_30'] == 0:
#         return 'full_booked'
#     else:
#         return 'active'
    




# def pricestr2float(df):
#     df["price"] = df["price"].str.replace(r'[$,]', '', regex=True)
#     df["price"] = pd.to_numeric(df["price"], errors="coerce")
#     print(df.price.describe(include='all'))
#     print(df.price.notna().value_counts())
#     return


# '''
# room_type:
# [Entire home/apt|Private room|Shared room|Hotel]

# All homes are grouped into the following three room types:

# Entire place
# Private room
# Shared room
# Entire place
# Entire places are best if you're seeking a home away from home. With an entire place, you'll have the whole space to yourself. This usually includes a bedroom, a bathroom, a kitchen, and a separate, dedicated entrance. Hosts should note in the description if they'll be on the property or not (ex: "Host occupies first floor of the home"), and provide further details on the listing.

# Private rooms
# Private rooms are great for when you prefer a little privacy, and still value a local connection. When you book a private room, you'll have your own private room for sleeping and may share some spaces with others. You might need to walk through indoor spaces that another host or guest may occupy to get to your room.

# Shared rooms
# Shared rooms are for when you don't mind sharing a space with others. When you book a shared room, you'll be sleeping in a space that is shared with others and share the entire space with other people. Shared rooms are popular among flexible travelers looking for new friends and budget-friendly stays.
# '''



# def categorize_property(ptype):
#     if pd.isna(ptype) or str(ptype).strip() == "":
#         return "others"
#     ptype_lower = str(ptype).lower()

#     # ENTIRE
#     if any(word in ptype_lower for word in ["entire", "condo", "loft", "apartment"]):
#         return "entire"

#     # HOTEL
#     elif "hotel" in ptype_lower:
#         return "hotel"

#     # SHARED
#     elif any(word in ptype_lower for word in ["shared", "bed and breakfast", "boutique"]):
#         return "shared"

#     # PRIVATE
#     elif "private" in ptype_lower:
#         return "private"

#     else:
#         return "others"
    
    

# """
# reviews_per_month: 	
# The average number of reviews per month the listing has over the lifetime of the listing.

# Psuedocoe/~SQL:

# IF scrape_date - first_review <= 30 THEN number_of_reviews
# ELSE number_of_reviews / ((scrape_date - first_review + 1) / (365/12))


# """