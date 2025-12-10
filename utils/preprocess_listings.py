## unzip + split + host_vars + proxy_obj_vars + desc + profile 


import pandas as pd
import os 
import time
import numpy as np
import time



##=============================DESC STAT=====================================##

def print_nan_ratio(df, col):
    ratio = (df[col].value_counts(dropna=False) / len(df)).round(2)
    nan_ratio = ratio.get(np.nan, 0)   # 如果没有 NaN，就返回 0
    # print(f"- ratio nan in '{col}': {nan_ratio}!")
    return nan_ratio

def print_zero_ratio(df, col):
    return np.round((len(df[df[col]==0])/len(df)),2)


def desc_catORnum(df, vars):

    print("\n================= BALN PROCESSED VARIABLES =====================\n"
        f"[INFO] statictic description on df ({len(df)} lines):\n"
        f"{' ; '.join(vars)}\n")
    
    
    for var in vars:
        if not var in df:
            print(f"[WARNING] {var} not in df!!!")
        else :
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
                if np.nan in col.value_counts(dropna=False):
                    print(f"[WARINNG] NaN in {var}!!")
                else :
                    print(f"no NaN in {var}:)")
                print(col.value_counts(dropna=False).sort_values(ascending=False), "\n")
                
            else:# 数值型，自动滤过了nan！！要手动打印在返回process处理！
                print(f"- {var}: {print_nan_ratio(df, col=var)*100}% NaN !\n"
                    f"- {var}: {print_zero_ratio(df, col=var)*100}% 0 !\n")
                
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






def preprocess_host_variables(df_raw):
    start_time=time.time()
    df=df_raw.copy()
    
    var_ok=["host_has_profile_pic","host_picture_url","host_identity_verified","host_name","number_of_reviews"]
    var_toprocess=["host_is_superhost","review_scores_rating", "host_since","host_about",
                "host_response_time","host_response_rate","calculated_host_listings_count"]
    print(f"\n\n******************************HOST VARS******************************\n"
          f"PROCESS PIPELINE :\n"
          f"- host_is_sueprhost: fillna('f')\n"
          f"- review_scores_rating: {print_nan_ratio(df, col='review_scores_rating')*100}% NaN; fillna(0), to numeric; ADD 'has_rating' :1/0'\n"
          f"- host_since: ADD 'years_since_host' :float, 0.5-1 year=>1， 0-0.5 year => 0 \n"
          f"- has_host_about: ADD 'has_host_about':1/0','lang:en/fr/other_langs','len:int',\n"
          f"- host_response_time:{print_nan_ratio(df, col='host_response_time')*100}% NaN, fillna('no_response_time') \n"
          f"- host_response_rate:{print_nan_ratio(df, col='host_response_rate')*100}% NaN, ADD 'has_response_rate' :1/0， fillna(0)\n"
          f"- calculated_host_listings_count : ADD 'professional_host:1/0'\n"
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
                df['review_scores_rating'] = pd.to_numeric(df['review_scores_rating'], errors='coerce')#有文本？？先转数字再填？
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
                # host_response_rate（数值）+%
                # 缺失值新增一列 'has_response_rate'：有为1，无为0; 所有值去掉百分号变成float
                df['has_response_rate'] = df_raw['host_response_rate'].notna().astype(int)
                
                # 转成字符串以便检查，但不修改原来的 NaN
                df['host_response_rate'] = df_raw['host_response_rate'].astype("string")
                mask_percent =df['host_response_rate'].str.contains('%', na=False)
                if mask_percent.any():
                    df['host_response_rate'] = df['host_response_rate'].str.rstrip('%')

                df['host_response_rate'] = pd.to_numeric(df['host_response_rate'], errors='coerce')
                df['host_response_rate'] = df['host_response_rate'] / 100
                df['host_response_rate'] = df['host_response_rate'].fillna(0)
                
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
    # desc_catORnum(df, vars=var_processed)
    desc_catORnum(df, vars=impo_vars)

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
    end_time=time.time()
    print(f"\n✅[SUCCES] Process host variables : {end_time-start_time:.2f} sec!\n")
    

    return df












##==================================PROXY============================================##

def filter_by_proxy(df,proxy_vars=['price',"availability_90"], get_boooking_rate_l30d=True):
    print(
        f"\n\n==============================PROXY=============================\n")

    
    for var in proxy_vars:
        # price
        if var == 'price' and var in df.columns :
            print(f"- price: {print_nan_ratio(df, col='price')*100}% NaN'\n")
            df['price'] = (
                df['price']
                .replace('[\$,]', '', regex=True)  # 去掉 $ 和 ,
                .astype(float)                     # 转成数值
            )            
            
    if get_boooking_rate_l30d==True:
        print(
            f"- number_of_reviews_l30d: {print_nan_ratio(df, col='number_of_reviews_l30d')*100}% NaN'\n"
            f"- availability_30: {print_nan_ratio(df, col='availability_30')*100}% NaN'\n"
        )
        df['booking_rate_l30d'] = df.apply(
            lambda row: min(row['number_of_reviews_l30d'] / row['availability_30'], 1.0)
            if row['availability_30'] > 0 else None,
            axis=1
        )
            
        
    # filtrer :
    # init df_filtered:
    df_filtred=df.copy()
    # if get_boooking_rate_l30d==True:
    #     proxy_vars.append('booking_rate_l30d')
        
    for var in proxy_vars :
        len_before=len(df_filtred)
        df_filtred=df_filtred[df_filtred[var].notna()]
        len_after=len(df_filtred)
        print(f"[INFO]{len_before-len_after} nan dropped in {var}")

    print(f"\nlen BEFORE filtrage by {'; '.join(proxy_vars)}: {len(df)}\n"
        f"len AFTER: {len(df_filtred)}\n")
   
    return df, df_filtred





##==================================LOCATION============================================##

def add_is_within_km(df, threshold_km):
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

def preprocess_obj_vars(df, proxy_vars=['price',"availability_30","availability_90"], 
                        get_boooking_rate_l30d=False, filtrate_by_booking_rate=False,#无输入时默认不按照booking筛选 
                        obj_vars=["room_type", "minimum_nights","instant_bookable"], 
                        threshold_km:int=None, 
                        output_folder="mod_results", filename=None):
    # obj_vars=["room_type", "minimum_nights","instant_bookable"]#all ok,无缺失/异常
    # proxy_vars=['price',"availability_90"]
    
    print(f"\n\n**********************************PROXY + OBJ VARS**********************************\n"
        f"PROCESS PIPELINE :\n"
        f"1) process proxies : \n"
        f" clean + dropna  : {', '.join(proxy_vars)} \n"
        f" ==> get df_filtered \n\n"
        
        f" 2) if get_boooking_rate_l30d==True, calculation method:\n"
        f"- booking_rate_l30d = number_of_reviews_l30d / availability_30 \n"
        f" note that many 'number_of_reviews_l30d' is 0!\n"
        f" if availability_30 = 0, take NaN.\n"
        f" if booking_rate_l30d > 1, take 1.\n\n"
        
        f"3) desc statistique :{', '.join(obj_vars)} \n\n"
        
        f"4) if enter 'threshold_km':\n"
        f"location :'latitude','longitude': ADD 'is_within_Xkm'\n"
        f"calculate  distance bewtween listing and its cloest venue. if it's under {threshold_km} km, 'is_within_{threshold_km} km' ==1, else 0.\n"
        )
    all_vars=[]
    
                
    # ---------------------------proxy-----------------------
    df, df_filtered=filter_by_proxy(df, proxy_vars=proxy_vars, get_boooking_rate_l30d=get_boooking_rate_l30d)
    all_vars.extend(proxy_vars)
    
    
    ## 单独写？
    if get_boooking_rate_l30d==True and filtrate_by_booking_rate==True:
        print(f"len BEFORE filtrage of nan (availability==0) by 'booking_rate_l30d': {len(df_filtered)}")
        df_filtered=df_filtered[df_filtered['booking_rate_l30d'].notna()]
        print(f"len AFTER: {len(df_filtered)}\n")           
    
    if get_boooking_rate_l30d==True:
        all_vars.extend(['number_of_reviews_l30d',"booking_rate_l30d"])#?
    
    #---------------------- obj vars ------------------------
    if "instant_bookable" in obj_vars:
        df["instant_bookable"]=df["instant_bookable"].fillna("f")
   
    all_vars.extend(obj_vars)
    
    # -----------------------location----------------------
    if threshold_km!=None:    
        df_filtered=add_is_within_km(df_filtered,threshold_km=3)
        all_vars.append(f'is_within_{threshold_km}km')
        
    # desc
    desc_catORnum(df_filtered, vars=all_vars) 
    
    # save
    os.makedirs(output_folder, exist_ok=True)
    if filename is None:
        filename="listings_filtered.csv"
    outpath_df_filtered=os.path.join(output_folder, filename)
    df_filtered.to_csv(outpath_df_filtered, index=False)
    print(f"\n✅[SAVE] df_filtered saved to {outpath_df_filtered}!")
    
    return df_filtered






def save_csv_as_latex(table_csv, output_path,caption, label):
    latex_code = table_csv.round(3).to_latex(
            caption=caption,
            label=label,
            index=True,        # 是否保留行索引（item 名称）
            escape=False       # False 可以保留 LaTeX 特殊字符，比如 _ 等
        )
    # print(f"LATEX : \n {latex_code}")
    with open(output_path, 'w') as f:
        f.write(latex_code)
    print(f"[SAVE] table latex saved to {output_path}!\n")   
    
    return 




## =================================profil comparaison=======================================##

def group_mean_table(df, cols, group_col='host_is_superhost'):
    """
    生成一个表格，对指定cols在group_col的两组之间取均值。
    
    df: pandas DataFrame
    cols: list of column names to observe
    group_col: 分组列名，默认 'host_is_superhost'
    
    返回: DataFrame，index=cols, 列=[Superhôte, Autres]
    """
    # 创建空DataFrame
    result = pd.DataFrame(index=cols, columns=['Superhôte', 'Autres'])
    
    for col in cols:
        # 两组均值
        result.loc[col, 'Superhôte'] = df[df[group_col]=='t'].get(col).mean()
        result.loc[col, 'Autres']   = df[df[group_col]!='t'].get(col).mean()
    
    return result



def group_mean_table_ttest(df, cols, group_col='host_is_superhost', output_folder="desc_results"):

    """
    返回均值表 + t-test p-value列
    """
    from scipy.stats import ttest_ind
    
    print(f"[INFO] ttest on {len(df)} lines.\n"
          f"group by : {df[group_col].value_counts(dropna=False)}\n")
    
    # filter valid cols:
    cols_valid=[col for col in cols if col in df.columns]
    cols_missing=[col for col in cols if col not in cols_valid]

    print(f"[WARNING]{len(cols_missing)} missing cols in df_input :\n {'; '.join(cols_missing)}\n")
    
    # 或者只筛选非数值列
    non_numeric_cols = [c for c in cols_valid if not pd.api.types.is_numeric_dtype(df[c])]
    if len(non_numeric_cols)>0:
        print(f"[WARNING] non_numeric_cols:\n {'; '.join(non_numeric_cols)}\n")
    
    numeric_cols=[c for c in cols_valid if c not in non_numeric_cols]
    print(f"Table of Superhost and others in {len(numeric_cols)} dimensions:\n")
    
    
    ## init res df
    result = pd.DataFrame(index=numeric_cols, columns=['Superhôte', 'Autres', 'ttest_p'])
    for col in numeric_cols:
        group1 = df[df[group_col]=='t'][col].dropna()
        group2 = df[df[group_col]!='t'][col].dropna()
        
        result.loc[col, 'Superhôte'] = group1.mean()
        result.loc[col, 'Autres']   = group2.mean()
        
        # t-test
        _, p = ttest_ind(group1, group2, equal_var=False)  # Welch t-test
        result.loc[col, 'ttest_p'] = p
    
  
        # 显著性星号
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = ''
        result.loc[col, 'significance'] = sig
    # result=result.sort_values(by="ttest_p", ascending=True)
    
    
    
    # save csv and latex:
    os.makedirs(output_folder, exist_ok=True)
    outpath_latex=os.path.join(output_folder, 'table_host_latex.tex')
    
    outpath_csv=os.path.join(output_folder, 'table_host.csv')    
    result.to_csv(outpath_csv, index=False)
    print(f"[SAVE] table host csv saved to {outpath_csv}!\n")  
    
    save_csv_as_latex(table_csv=result, 
                      output_path=outpath_latex, 
                      caption="Tableau du profil des Superhôtes et des Autres",
                      label="tab:table_host")

    
    return result




















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