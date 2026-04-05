## unzip + split + host_vars + proxy_obj_vars + desc + profile 


import pandas as pd
import os 
import time
import numpy as np
import time
from utils.io import save_csv_as_latex

##=============================DESC STAT=====================================##

def print_nan_ratio(df, col):
    ratio = (df[col].value_counts(dropna=False) / len(df)).round(2)
    nan_ratio = ratio.get(np.nan, 0)   # 如果没有 NaN，就返回 0
    # print(f"- ratio nan in '{col}': {nan_ratio}!")
    return nan_ratio

def print_zero_ratio(df, col):
    return np.round((len(df[df[col]==0])/len(df)),2)


def desc_catORnum(df, vars):
    if isinstance(vars, str):
        vars=[vars]
    print("\n================= BALN PROCESSED VARIABLES =====================\n"
        f"[INFO] statictic description on df ({len(df)} lines):\n"
        f"{' ; '.join(vars)}\n")
    
    
    for var in vars:
        if not var in df:
            print(f"[WARNING] {var} not in df!!!\n")
            print("-----------------------------------------------------------")           
            
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
                if print_nan_ratio(df, col=var)!=0:
                    # df[var].isna().any()==True:
                    print(f"- {var}: {print_nan_ratio(df, col=var)*100}% NaN !\n"
                        f"- {var}: {print_zero_ratio(df, col=var)*100}% 0 !\n")
                else:
                    print(f"no NaN in {var}:)\n"
                          f"{print_zero_ratio(df, col=var)*100}% 0\n")
                    
                print(col.describe(include="all"), "\n")

            print("-----------------------------------------------------------")
    return 




##==================================HOST VARS============================================##
import langid #这个库速度更快、稳定性高。

def is_valid_text(text):
    """
    filtrer les textes:
        pas de nan,
        pas que les ponctuation,
        pas que les liens url
        pas que les caractères
    renvoyer T/F
    """
    if pd.isna(text):  # ==dropnaNaN
        return False
    text = str(text).strip()
    if len(text) == 0:  # strip
        return False
    # effacer ce qui n'a que de ponctuation
    if not re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", text):
        return False
        # effacer url
    if re.fullmatch(r"https://\S+", text):
        return False
    return True



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



def preprocess_host_variables(df_raw, 
                            save=False, output_folder="data_processed", 
                            filename=f"listings_processed.csv"):
    start_time=time.time()
   
    df=df_raw.copy()
    

    var_toprocess=["host_has_profile_pic", "host_identity_verified","host_is_superhost","review_scores_rating", "host_since","host_about",
                "host_response_time","host_response_rate","calculated_host_listings_count"]
    print(f"\n\n******************************HOST VARS******************************\n"
          f"PROCESS PIPELINE :\n"
          f"- host_has_profile_pic : fillna('f') if not f/t\n"
          f"- host_identity_verified: fillna('f') if not f/t \n"
          f"- host_is_sueprhost: check ONLY 't'/'f'; fillna('f')\n"
          f"- review_scores_rating: {print_nan_ratio(df, col='review_scores_rating')*100}% NaN; fillna(mean), to numeric; ADD 'has_rating' \n"
          f"- host_since: ADD 'years_since_host' :float, 0.5-1 year=>1， 0-0.5 year => 0 \n"
          f"- has_host_about: 'lang:en/fr/other_langs','len:int',\n"
        #   ADD 'has_host_about':1/0',
          f"- host_response_time:{print_nan_ratio(df, col='host_response_time')*100}% NaN, fillna('no_response_time') \n"
          f"- host_response_rate:{print_nan_ratio(df, col='host_response_rate')*100}% NaN, ADD 'has_response_rate' :1/0， fillna(0)\n"
          f"- calculated_host_listings_count : ADD 'professional_host:1/0', count>1\n"
          )
    
    
    for var in var_toprocess:
        if not var in df.columns :
            print(f"[WARNING] {var} not found in df!")
        else :
            if var=="host_has_profile_pic":
                df['host_has_profile_pic']=df['host_has_profile_pic'].apply(lambda x : x if x in ['t','f'] else 'f')
            
            elif var=="host_identity_verified":
                df['host_identity_verified']=df['host_identity_verified'].apply(lambda x : x if x in ['t','f'] else 'f')                
                
            elif var=='host_is_superhost':
                # host_is_superhost :对缺失superhost的填f
                df['host_is_superhost']=df['host_is_superhost'].apply(lambda x : x if x in ['t', 'f'] else None)
                df['host_is_superhost']=df['host_is_superhost'].fillna('f')

            elif var=='review_scores_rating':
                #review_scores_rating:对缺失评分（数值）增加一列has_rating：有为1，无为0;
                # 双变量法：相当于把有/无评分的分开，所以原列中可以fillna(0); 区分信号存在/强度
                df['has_rating'] = df['review_scores_rating'].notna().astype(int)
                df['review_scores_rating'] = pd.to_numeric(df['review_scores_rating'], errors='coerce')#有文本？？先转数字再填？
                # 填0太极端，会破坏模型，在筛选后has_rating ==0在20%以下，可以选择平均值
                median_rating = df['review_scores_rating'].median()
                mean_rating = df['review_scores_rating'].mean()

                print(f"[INFO] missing review scores are filled by mean :{mean_rating}\n")
                df['review_scores_rating'] = df['review_scores_rating'].fillna(median_rating)


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
                # df['has_host_about'] = df['host_about'].notna().astype(int)
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
    var_processed=[
        "host_identity_verified",
        "host_has_profile_pic",
        "host_is_superhost",
        "review_scores_rating","has_rating", 
        "host_since","years_since_host",
        "host_about", "lang","len", #"has_host_about"包括在len中！
        "host_response_time",
        "host_response_rate","has_response_rate",
        "calculated_host_listings_count", "professional_host"
        ]
    

    print(f"==========================HOST VARS================================")
    vars_ok=["host_picture_url","number_of_reviews","number_of_reviews_nextQ"]
    # 若先filter，number_of_reviews会被重命名为number_of_reviews_nextQ
    #"host_name",
    impo_vars=vars_ok + var_processed
    print(f"{len(impo_vars)} IMPO VARS:{impo_vars}\n") 


    # print(f"===============BIALN PROCESSED VARIABLES=====================")
    # desc_catORnum(df, vars=var_processed)
    desc_catORnum(df, vars=impo_vars)

    # delect intermidate cols:optionnal
    # intermediate_cols=[]
    
    df=df.drop(columns=['days_since_host'])#?
    print(f"Intermediate cols deleted: days_since_host")
    end_time=time.time()
    print(f"\n✅[SUCCES] Process host variables : {end_time-start_time:.2f} sec!\n")
    
    if save ==True:
        os.makedirs(output_folder, exist_ok=True)
        outpath_df_filtered=os.path.join(output_folder, filename)
        df.to_csv(outpath_df_filtered, index=False)
        print(f"\n✅[SAVE] {len(df)} lines df_filtered saved to '{outpath_df_filtered}'!")
    
    
    return df










##==================================PROXY============================================##

def filter_df(df, vars, 
              filtrate_by_booking_rate_l30d=False, 
              filtrate_by_booking_rate_l90d=False,
              ):
    
    df_filtered=df.copy()

    # check vars :
    vars_valid=[v for v in vars if v in df.columns ]
    missings_vars=[v for v in vars if v not in vars_valid]
    if len(missings_vars)>0:
        print(f"[WARNING] missing vars in df : {'; '.join(missings_vars)}!!")
    
    if filtrate_by_booking_rate_l30d==False :
        vars_valid=[ v for v in vars_valid if v not in [ 'number_of_reviews_l30d','availability_30','booking_rate_l30d']]
        print("[CHECK] no filter on 'number_of_reviews_l30d','availability_30','booking_rate_l30d'!")
    
    if filtrate_by_booking_rate_l90d==False :
        vars_valid=[ v for v in vars_valid if v not in [ 'number_of_reviews_nextQ','availability_90','booking_rate_l90d']]
        print("[CHECK] no filter on  'number_of_reviews_nextQ','availability_90','booking_rate_l90d'!")
        
            
    # filter df
    for var in vars_valid :
        len_before=len(df_filtered)
        df_filtered=df_filtered[df_filtered[var].notna()]
        len_after=len(df_filtered)
        print(f"[INFO]{len_before-len_after} nan dropped in {var}")

    print(
        f"\nfilter by: {'; '.join(vars)}\n"
        f"len BEFORE: {len(df)}\n"
        f"len AFTER: {len(df_filtered)}\n"
        )
   
    return df_filtered




def check_proxy_vars(df,proxy_vars=['price',"availability_90"]):
   
    for var in proxy_vars:
        # price
        if var == 'price' and var in df.columns :
            print(f"- price: {print_nan_ratio(df, col='price')*100}% NaN'\n")
            df['price'] = (
                df['price']
                .replace('[\$,]', '', regex=True)  # 去掉 $ 和 ,
                .astype(float)                     # 转成数值
            )            
                    
    # filtrer :
    # init df_filtered:
    # if get_boooking_rate_l30d==True:
    #     proxy_vars.append('booking_rate_l30d')

    return df, proxy_vars



def add_booking_rate_l30d(df):
    print(
            f"- number_of_reviews_l30d: {print_nan_ratio(df, col='number_of_reviews_l30d')*100}% NaN'\n"
            f"- availability_30: {print_nan_ratio(df, col='availability_30')*100}% NaN'\n"
        )
    df['booking_rate_l30d'] = df.apply(
        lambda row: min(row['number_of_reviews_l30d'] / row['availability_30'], 1.0)
        if row['availability_30'] > 0 else None,
        axis=1
    )      

    return df 
    
  
def add_booking_rate_l90d(df, df_nextQ):
    """
    
    
    """
    
    # no match / neg value stay nan in number_of_reviewsQ3
    
    # 本季度为止累计评论数：number_of_reviews_till_Q
    df_reviews=df.copy()
    df_reviews=df_reviews.rename(columns={"number_of_reviews":'number_of_reviews_till_Q'})

    # 下季度为止累计评论数：number_of_reviews_till_nextQ
    reviews_nextQ=df_nextQ[['id','number_of_reviews']]
    reviews_nextQ.columns=['id','number_of_reviews_till_nextQ']
    
    # merge
    df_reviews=df_reviews.merge(reviews_nextQ, left_on='id', right_on="id", how="left")

    # 计算本季度获得的评论数：number_of_reviews_nextQ
    # 下季度无匹配或差值小于0 则number_of_reviews_nextQ为nan    
    df_reviews['number_of_reviews_nextQ']=df_reviews['number_of_reviews_till_nextQ']-df_reviews["number_of_reviews_till_Q"]
    df_reviews['number_of_reviews_nextQ']=df_reviews['number_of_reviews_nextQ'].apply(lambda x : np.nan if x<0 else x)
        
    print(f"[CHECK] no match on reviews_nextQ : {len(df_reviews[df_reviews['number_of_reviews_nextQ'].isna()])};\n"
          f"NO ava90 this Q: {len(df_reviews[df_reviews['availability_90']==0])}")
    
    #若nextQ为nan，则booking自动为nan
    # df_reviews['booking_rate_l90d'] = df_reviews.apply(
    #             lambda row: min(row['number_of_reviews_nextQ'] / row['availability_90'], 1.0)
    #             if row['availability_90'] > 0 else None, #直接去掉了不开放房东的行！不需要再在ava上drop0
    #             axis=1
    #         )
    
    #向量化写法，速度更快：
    df_reviews['booking_rate_l90d'] = (
    df_reviews['number_of_reviews_nextQ'] / df_reviews['availability_90']
        ).clip(upper=1.0)

    # 对 availability_90 == 0 或 number_of_reviews_nextQ 为 NaN 的行，也会自然是 NaN
    df_reviews.loc[df_reviews['availability_90'] == 0, 'booking_rate_l90d'] = np.nan

    return df_reviews




##=======================================LOCATION============================================##

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


##property type
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




###======================================MAIN of=================================================###
##==================================PROXY +OBJ+LOCATION ======================================##

def preprocess_obj_vars(df, df_nextQ, proxy_vars=['price',"availability_30","availability_90"], 
                        get_booking_rate_l30d=False, filtrate_by_booking_rate_l30d=False,#无输入时默认不按照booking筛选 
                        get_booking_rate_l90d=True, filtrate_by_booking_rate_l90d=True,
                        obj_vars=["room_type", 'property_type',"minimum_nights","instant_bookable"], 
                        threshold_km:int=None, 
                        save=False, output_folder="data_processed", filename="listings_filtered.csv"):
    
    # obj_vars=["room_type", "minimum_nights","instant_bookable"]#all ok,无缺失/异常
    # proxy_vars=['price',"availability_90"]
    
    print(f"\n\n==========================PROXY + OBJ VARS============================\n"
        f"PROCESS PIPELINE :\n"
        f"1) process proxies : \n"
        f"- price : delete '$', to_numeric\n\n"
        
        f" 2) if get_boooking_rate_l30d==True, calculation method:\n"
        f"- booking_rate_l30d = number_of_reviews_l30d / availability_30 \n"
        f" note that many 'number_of_reviews_l30d' is 0!\n"
        f" if availability_30 = 0, take NaN.\n"
        f" if booking_rate_l30d > 1, take 1.\n\n"
        
        
        f"3) if 'add_booking_rate_l90d':\n"
        f" ADD number_of_reviews_nextQ, booking_rate_l90d \n"
        f" if host no longer exists in df_nextQ / substraction get negative value/ ava_90_thisQ==0, then number_of_reviews_nextQ=> nan \n"
        f" 1 >= booking_rate_l30d  = number_of_reviews_nextQ (Q3)/ availability_90 (Q2)\n"
        f" if filter False: no filter on booking_rate, ava, nb_reviews\n\n"

        
        f"4) obj vars :\n "
        f"- instant_bookable : fillna('f')\n"
        f"- minimum_nights : to_numeric, fillna(0)\n"
        f"- property_type : clean col : entire, hotel, shared, private, others.\n\n"

        
        f"5) filter : dropna on vars ==> desc df_filtered \n\n"
        f"desc statistique :{', '.join(obj_vars)} \n\n"
                
        f"6) if enter 'threshold_km':\n"
        f"location :'latitude','longitude': ADD 'is_within_Xkm'\n"
        f"calculate  distance bewtween listing and its cloest venue. if it's under {threshold_km} km, 'is_within_{threshold_km} km' ==1, else 0.\n\n"
        
        f"7) cols to keep : only keeps cols needed!\n\n")
    
    
    vars_to_dropna=[]    
    
    print("# ---------------------------proxy---------------------------")
    
    df, proxy_vars=check_proxy_vars(df, proxy_vars=proxy_vars)
    
    vars_to_dropna.extend(proxy_vars)
    
    if get_booking_rate_l30d==True:
        df=add_booking_rate_l30d(df)
        vars_to_dropna.extend(["number_of_reviews_l30d","availability_30","booking_rate_l30d"])
    
    if get_booking_rate_l90d==True :
        if  df_nextQ is  None or df_nextQ.empty:
            print(f"[CHECK] need df_nextQ!!!")
        else :
            print(f"[CHECK] get booking_rate_l90d!")
            df=add_booking_rate_l90d(df, df_nextQ)
        vars_to_dropna.extend(["number_of_reviews_nextQ","availability_90","booking_rate_l90d"])

    print("#------------------------ obj vars ---------------------------")
    if "instant_bookable" in obj_vars:
        df["instant_bookable"]=df["instant_bookable"].fillna("f")
        
    if "minimum_nights" in obj_vars:# 又不是数字的值: 2023-12-12
        df['minimum_nights'] = pd.to_numeric(df.minimum_nights, errors='coerce')#无法转换的会变成nan
        df["minimum_nights"]=df["minimum_nights"].fillna(0)

    if "room_type" in obj_vars:
        """
        
        4 room_type
        Entire home/apt    32396
        Private room        3819
        Hotel room           530
        Shared room          268
        2.1                    1
        Name: count, dtype: int64        
        """
        df['room_type']=df['room_type'].apply(lambda x : x if x in ['Entire home/apt',"Private room","Hotel room", "Shared room"] else None)           
        
    if "property_type" in obj_vars:
        df["property_type"] = df["property_type"].apply(categorize_property)
        
    vars_to_dropna.extend(obj_vars)
    

    if threshold_km!=None:    
        print('#-------------------------location------------------------')
        df=add_is_within_km(df,threshold_km=threshold_km)
        vars_to_dropna.append(f'is_within_{threshold_km}km')# extend (list)!        
        
        
    print("# -----------------------filter & desc-------------------------")
    vars_to_dropna=list(set(vars_to_dropna))
    print(f"[INFO] vars to dropna:{'; '.join(vars_to_dropna)}")
    
    df_filtered=filter_df(df,vars=vars_to_dropna, 
                          filtrate_by_booking_rate_l30d=filtrate_by_booking_rate_l30d,
                          filtrate_by_booking_rate_l90d=filtrate_by_booking_rate_l90d)
    
    desc_catORnum(df=df_filtered, vars=vars_to_dropna) 


    if save ==True:
        os.makedirs(output_folder, exist_ok=True)
        outpath_df_filtered=os.path.join(output_folder, filename)
        df_filtered.to_csv(outpath_df_filtered, index=False)
        print(f"\n✅[SAVE] {len(df_filtered)} lines df_filtered saved to '{outpath_df_filtered}'!")
    
    return df_filtered







# --------------------------------------simplify------------------------------------------
def keep_cols(df, cols_to_keep, 
              save=True, output_folder=None, filename=None):

    df_simple=df.copy()
        
    cols_to_keep_valid = [c for c in cols_to_keep if c in df_simple.columns]
    cols_to_keep_missing = [c for c in cols_to_keep if c not in cols_to_keep_valid]
    
    if len(cols_to_keep_missing)>0:
        print(f"[INFO] skip missing cols to keep :{'; '.join(cols_to_keep_missing)}!")

    df_simple=df_simple[cols_to_keep_valid]
    print(f"[INFO] df only keeps {df_simple.columns}!")
    
    if save:
        os.makedirs(output_folder, exist_ok=True)
        if filename==None:# fallback!
            filename="listings_simple.csv"
        outpath_df_simple=os.path.join(output_folder, filename)
        df_simple.to_csv(outpath_df_simple, index=False)
        
        print(f"\n✅[SAVE] {len(df_simple)} df simple saved to '{outpath_df_simple}'!")
    
    return df_simple







## ======================================DESC================================================##


"""
# exemplaire :

cols_to_check=[
    # "host_identity_verified", "host_has_profile_pic",         
    "review_scores_rating",'number_of_reviews',
    "years_since_host", "professional_host",#'calculated_host_listings_count',
    # "lang", "len",
    "price", "availability_30", "room_type", "instant_bookable"
]   


"""


# def group_mean_table(df, cols, group_col='host_is_superhost'):
#     """
#     生成一个表格，对指定cols在group_col的两组之间取均值。
    
#     df: pandas DataFrame
#     cols: list of column names to observe
#     group_col: 分组列名，默认 'host_is_superhost'
    
#     返回: DataFrame，index=cols, 列=[Superhôte, Autres]
#     """
#     # 创建空DataFrame
#     result = pd.DataFrame(index=cols, columns=['Superhôte', 'Autres'])
    
#     for col in cols:
#         # 两组均值
#         result.loc[col, 'Superhôte'] = df[df[group_col]=='t'].get(col).mean()
#         result.loc[col, 'Autres']   = df[df[group_col]!='t'].get(col).mean()

#     return result





def group_mean_table_ttest(df, cols_to_check, group_col='host_is_superhost',
                           save=False, output_folder="mod_results", filename_noext=None):
    from scipy.stats import ttest_ind

    if save and output_folder:
        os.makedirs(output_folder, exist_ok=True)     
    
    if group_col in cols_to_check :
        cols_to_check.remove(group_col)
        print(f"[CHECK] remove group_col '{group_col}' from vars")


    # ------------------------------------check groups----------------------------------
    groups = df[group_col].unique()
    print(f"[CHECK] ttest takes only 2 groups! OR go to ANOVA!")

    if len(groups) != 2:
        # raise ValueError("ttest requires exactly 2 groups")
        print(f"[WARNING] more than 2 groups!")

    g1, g2 = groups[:2]
    print(f'[INFO] groups :{g1} vs {g2}')    
    
    #--------------------------------filter+ desc-----------------------------------------
    print(f"[INFO] ttest on {len(df)} lines.\n"
        #   f"group by : {df[group_col].value_counts(dropna=False)}\n"
    )        
    # 变量存在？
    cols_valid=[col for col in cols_to_check if col in df.columns]
    cols_missing=[col for col in cols_to_check if col not in cols_valid]
    if len(cols_missing)>0:
        print(f"[WARNING]{len(cols_missing)} missing cols in df_input :\n {'; '.join(cols_missing)}\n")
    
    # 变量为数值？
    print(f"[CHECK] ttest take ONLY numeric cols! \n")
    non_numeric_cols = [c for c in cols_valid if not pd.api.types.is_numeric_dtype(df[c])]
    numeric_cols=[c for c in cols_valid if c not in non_numeric_cols]

    if len(non_numeric_cols)>0:
        print(f"[WARNING] no numeric cols:\n {'; '.join(non_numeric_cols)}\n")
       
    
    #---------------------------------- init result df------------------------------------
    # 一次性声明清楚
    result = pd.DataFrame(
        index= ['proportion'] + list(numeric_cols),
        columns=[g1, g2, 'ttest_p', 'significance']
    )
    result.columns.name = group_col # 左上角标记分组变量

    # proportions
    result.loc['proportion', g1] = (df[group_col] == g1).mean()
    result.loc['proportion', g2] = (df[group_col] == g2).mean()

    # for col in non_numeric_cols:
    #   result.loc[col, g1]=(df[group_col] == g1)
    
    # loop over numeric columns
    for col in numeric_cols:
        x1 = df.loc[df[group_col] == g1, col].dropna()
        x2 = df.loc[df[group_col] == g2, col].dropna()

        result.loc[col, g1] = x1.mean()
        result.loc[col, g2] = x2.mean()

        _, p = ttest_ind(x1, x2, equal_var=False)
        result.loc[col, 'ttest_p'] = p

        # significance
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = ''
        result.loc[col, 'significance'] = sig
        
        
    if group_col=='host_is_superhost':
        
        result = result.rename(columns={'t':"Superhôte",
                                        'f':"Autres"}
                               )  
        desired_order=['Superhôte','Autres','ttest_p','significance']
        result=result[desired_order]    
        
    # display(result)
         
    if save :  
        if filename_noext==None:
            filename_csv= f'table_groupby_{group_col}.csv'
            filename_tex=f'table_groupby_{group_col}.tex'
        else : 
            filename_csv=filename_noext+'.csv'
            filename_tex=filename_noext+'.tex'
            
        # as csv
        outpath_csv=os.path.join(output_folder,filename_csv)    
        result.to_csv(outpath_csv, index=True)# index TRUE!!!
        print(f"✅ [SAVE] table host csv saved to {outpath_csv}!\n")  

        # as latex 
        outpath_latex=os.path.join(output_folder,"latex", filename_tex)
        save_csv_as_latex(table_csv=result, 
                        output_path=outpath_latex, 
                        caption="Tableau du profil des Superhôtes et des Autres",
                        label="tab:table_host", 
                        ndigits=3)

    return result




# def plot_distribution(df, group_col=None, y_var='booking_rate_l30d', 
#                     save=False, output_folder="mod_results",filename=None):
#     import matplotlib.pyplot as plt
#     import seaborn as sns

#     # 单独使用时需要检查
#     os.makedirs(output_folder, exist_ok=True)     
#     # to numeric
#     df[y_var] = pd.to_numeric(df[y_var], errors='coerce')
    
#     plt.figure(figsize=(10,6))
#     # colors = ['#1f77b4', '#ff7f0e']  # 蓝色/橙色
#     colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

#     title= f"Distribution de {y_var}" 
#     if y_var=="booking_rate_l30d":
#         title=f"Distribution de Taux de réservation" # no touch to y_var!

#     if group_col:
#         groups = df[group_col].unique()
#         print(f'[INFO] groups of {group_col}:{groups}')
        
#         group_title=f" ({group_col} {groups[0]} vs {groups[1]})" #开头空一格
#         if group_col=='host_is_superhost':
#             group_title= " (Superhôtes vs Autres)"
#         title+=group_title


#         for i, val in enumerate(groups):
#             group_data = df[df[group_col]==val][y_var].dropna()
#             label=val
#             if group_col=="host_is_superhost":
#                 label = "Superhôtes" if val=="t" else "Autres"
            
#             # 画 KDE 曲线
#             sns.kdeplot(group_data, fill=True, alpha=0.3, label=label, color=colors[i])
#             sns.kdeplot(group_data, color=colors[i], lw=2)  
    
    
#     else :     
#         group_data=df[y_var].dropna()       
#         sns.kdeplot(group_data, fill=True, alpha=0.3, color=colors[0])
#         sns.kdeplot(group_data, color=colors[0], lw=2)  # 描边
      
#     plt.xlabel(f"{y_var}")
#     plt.ylabel("Densité")
#     plt.title(title)
#     plt.legend()
    
#     if save: 
#         if filename==None:
#             filename=f"host_performance_on_{y_var}.jpg"
#         outpath_kde=os.path.join(output_folder,filename)
#         plt.savefig(outpath_kde, dpi=300)
#         print(f"✅ [SAVE] plot distribution saved to {outpath_kde}!")
    
#     plt.show()
        
#     return  

def plot_distribution(
    df, 
    group_cols=None, 
    y_var='booking_rate_l30d',
    save=False, 
    output_folder="mod_results",
    filename=None
):
    import os
    import math
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_folder, exist_ok=True)
    df = df.copy()
    df[y_var] = pd.to_numeric(df[y_var], errors='coerce')

    # 标题
    title = f"Distribution de {y_var}"
    if y_var == "booking_rate_l30d":
        title = "Distribution de Taux de réservation"

    palette = sns.color_palette("tab10")
        
    # palette = {
    #     "london_2023": "#9ecae1",
    #     "london_2024": "#2171b5",
    #     "paris_2023": "#dadaeb",
    #     "paris_2024": "#6a51a3"
    # }

    # ======================
    # 👉 多列分组 → 子图模式
    # ======================
    if isinstance(group_cols, list):
        group_key = "_group_tmp"
        df[group_key] = df[group_cols].astype(str).agg("_".join, axis=1)

        groups = df[group_key].dropna().unique()
        n_groups = len(groups)

        print(f"[INFO] groups: {groups}")

        # 👉 每行4个
        ncols = 4
        nrows = math.ceil(n_groups / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows),sharey=True)#共享y轴
        
        axes = axes.flatten()

        for i, val in enumerate(groups):
            ax = axes[i]

            group_data = df[df[group_key] == val][y_var].dropna()
            if len(group_data) == 0:
                continue

            # label格式
            parts = val.split("_")
            label = ", ".join([
                f"{col}={v}" for col, v in zip(group_cols, parts)
            ])

            color = palette[i % len(palette)]
            sns.kdeplot(group_data, fill=True, alpha=0.3, ax=ax, color=color)
            sns.kdeplot(group_data, lw=2, ax=ax, color=color)

            ax.set_title(label)
            ax.set_xlabel(y_var)
            ax.set_ylabel("Densité")

        # 👉 删除多余子图
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

    # ======================
    # 👉 单列分组（保留原 overlay）
    # ======================
    elif isinstance(group_cols, str):

        plt.figure(figsize=(10,6))

        groups = df[group_cols].dropna().unique()
        print(f"[INFO] groups: {groups}")

        for i, val in enumerate(groups):
            group_data = df[df[group_cols] == val][y_var].dropna()
            if len(group_data) == 0:
                continue

            label = val
            if group_cols == "host_is_superhost":
                label = "Superhôtes" if val == "t" else "Autres"

            color = palette[i % len(palette)]

            sns.kdeplot(group_data, fill=True, alpha=0.3, label=label, color=color)
            sns.kdeplot(group_data, lw=2, color=color)

        plt.xlabel(y_var)
        plt.ylabel("Densité")
        plt.title(title)
        plt.legend()

    # ======================
    # 👉 不分组
    # ======================
    else:
        plt.figure(figsize=(10,6))

        group_data = df[y_var].dropna()
        sns.kdeplot(group_data, fill=True, alpha=0.3)
        sns.kdeplot(group_data, lw=2)

        plt.xlabel(y_var)
        plt.ylabel("Densité")
        plt.title(title)

    # ======================
    # 保存
    # ======================
    if save:
        if filename is None:
            filename = f"distribution_{y_var}.jpg"
        outpath = os.path.join(output_folder, filename)
        plt.savefig(outpath, dpi=300)
        print(f"✅ [SAVE] plot saved to {outpath}")

    plt.show()
    

def plot_violon(df_input, vars, to_fillna0=False, save=False, 
                output_folder=None, filename=None):
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    df=df_input.copy()
    if to_fillna0==True:
        df[vars]=df[vars].fillna(0)
    
    data=df[vars]   
    plt.figure(figsize=(6,4))
    sns.violinplot(data, palette='viridis')
    plt.title("violinplot of scores")
    plt.xticks(rotation=45)
    
    # save
    if save:    
        if filename==None:
            filename="violonplot.jpg"
        outpath_violon=os.path.join(output_folder,filename)
        plt.tight_layout()#必须在savefig前
        plt.savefig(outpath_violon, dpi=300)
        print(f"✅ [SAVE] violon plot saved to {outpath_violon}!")
        
    return 