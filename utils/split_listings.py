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

def split_change_stable(path_Q1, path_Q2, year:int,threshold_text_sim=0.85, 
                        save=False, output_folder=None, filename=None):
    # return dfQ2_split
    print(f"[IO] input : 'dfQ2'.csv => outupt: 'dfQ2'_split.csv")
      
    Q1=pd.read_csv(path_Q1)
    Q2=pd.read_csv(path_Q2)
    
    start_time=time.time()
    print("===================================INPUT DATA===================================")

    print("len df Q1:",len(Q1))
    print("len df Q2:",len(Q2))
    
    #标记Q，合并数据集：
    Q1['scraped_date'] = f"{year}Q1"# no espace between year and Q!!
    Q2['scraped_date'] = f"{year}Q2"
    df = pd.concat([Q1, Q2], ignore_index=True)
    df['host_since']=pd.to_datetime(df['host_since'], errors="coerce")
    print(f"len df total : {len(df)} \n")

    # 增加status列：
    # 同级host_id在某一个季度出现的次数，按照host_id合并Q1Q2
    hosts_Q1 = Q1.groupby('host_id').size().reset_index(name='Q1')
    hosts_Q2 = Q2.groupby('host_id').size().reset_index(name='Q2')
    merged = hosts_Q1.merge(hosts_Q2, on='host_id', how='outer')
    """
    merged:
            host_id   Q1   Q2          status
    0            275  1.0  NaN  deactived_host
    1           2626  6.0  3.0        old_host

    """

    # 标记每个host_id在Q1>Q2的状态:
    # reactive_host(这里包括了new host), desactive_host, old_host : listings增加/减少/不变
    merged['status'] = merged.apply(
        lambda r:
            'reactive_host' if pd.isna(r['Q1']) else#不在Q1中
            'deactived_host' if pd.isna(r['Q2']) else#不在Q2中
            "old_host",
        axis=1
    )
    
    # 按照host_id merge back
    df=df.merge(merged[['host_id','status']], left_on='host_id', right_on='host_id',how='left')

    
    # 把是reactive_host，且host_since在今年6以前的标记为new_host, 做成mask
    # 仅用host_since只能识别出很小一部分的房东，可能很早注册房东，但没有出租，受到JO刺激才进入市场！
    # mtd：根据host_since 细分new和reactive：纯向量操作（快十几倍，不需要 apply），同下效果。
    mask = (
        (df['status'] == 'reactive_host') &
        (df['host_since'].between(f'{year}-01-01', f'{year}-06-30'))# faut int ici!!!
    )
    df.loc[mask, 'status'] = 'new_host'
   
 
 
    # 统计房东状态，但只看Q2避免，Q1Q2重复记录!
    print('listings count change'.center(100,"-"))
    #描述房源数量变化
    listings_in = set(Q2['id']) - set(Q1['id']) # 9月有，但6月没有
    listings_out = set(Q1['id']) - set(Q2['id']) #6月有，但9月没有
    listings_io= set(Q2['id']) ^ set(Q1['id']) # 属于 6月 或 9月，但不同时属于两者;所有变化的房东（新增 + 消失）
    print(f'new listings in Q2 {len(listings_in)}({len(listings_in)*100/len(Q2):.2f}%)')
    print(f'listings dispeared in Q2 : {len(listings_out)}({len(listings_out)*100/len(Q2):.2f}%)')
    print(f"listings changed (in+out) in Q2:{len(listings_io)}\n")


    #=============================增加host_about_q2, host_about_change列=========================
    print("BIO change".center(100, '-'))
    if 'host_about_q1' not in df:
        df_q1 = df[df['scraped_date'] == f'{year}Q1'][['host_id', 'id', 'host_about']]
        df_q1 = df_q1.rename(columns={'host_about': 'host_about_q1'})
        df = df.merge(df_q1, on=['host_id', 'id'], how='left')

    # 计算文本相似度（使用 difflib.SequenceMatcher）
    from difflib import SequenceMatcher
    def text_similarity(a, b):
        if pd.isna(a) or pd.isna(b):
            return None
        return SequenceMatcher(None, str(a), str(b)).ratio()

    # 标记 Q2 相对于 Q1 的变化
    def host_about_change(row, threshold=threshold_text_sim):
        """
        没改动/例外都被填为nan （包括所有Q1房东!）
        
        新房东/新文本/新bio填1或sim
        """
        #虽然计算sim，但是所有标记成：变化1，不变0。仅用于筛选
        if row['scraped_date'] != f"{year}Q2":
            return 0  # 只考虑 Q2 标记，Q1均为0
        
        if pd.isna(row['host_about_q1']) and pd.notna(row['host_about']):
            return 1  # 新增文本; （新房东或之前未填写文本的房东）
        if pd.notna(row['host_about_q1']) and pd.notna(row['host_about']):#老房东&bio不为空# row['host_about'] != row['host_about_q2']:
            sim= text_similarity(row['host_about_q1'], row['host_about']) 
            if sim < threshold:## 老文本有明显变化
                return 1 
            else:
                return 0 #无明显改变是就填NAN
        return 0 # 未变&其他情况

    if "host_about_changed" not in df:
        df['host_about_changed'] = df.apply(host_about_change, axis=1)


    #=======================增加host_picture_url_q1, host_picture_url_change列=========================
    print('PIC change'.center(100,'-'))
    if "host_picture_url_q1" not in df:
        df_q1 = df[df['scraped_date'] == f"{year}Q1"][['host_id', 'id', 'host_picture_url']]
        df_q1 = df_q1.rename(columns={'host_picture_url': 'host_picture_url_q1'})
        df = df.merge(df_q1, on=['host_id', 'id'], how='left')

    # 标记 Q2 相对于 Q1 的变化
    def host_picture_url_change(row):
        if row['scraped_date'] !=f"{year}Q2":
            return 0 # 只对 Q2 标记, Q1均为0
        if pd.isna(row['host_picture_url_q1']) and pd.notna(row['host_picture_url']):
            return 1  # 新增照片
        if pd.notna(row['host_picture_url_q1']) and pd.notna(row['host_picture_url']) and row['host_picture_url'] != row['host_picture_url_q1']:
            return 1  # 修改照片
        return 0 # 没变&其他情况

    if "host_picture_url_changed" not in df:
        df['host_picture_url_changed'] = df.apply(host_picture_url_change, axis=1)


    #=======================增加number_of_reviews_q1,booking_rate_l90d_previousQ =========================
    print('number of reviews till previousQ'.center(100,'-'))
    if "number_of_reviews_till_previousQ" not in df:
        df_q1 = df[df['scraped_date'] == f"{year}Q1"][['host_id', 'id', "availability_90",'number_of_reviews']]
        df_q1 = df_q1.rename(columns={"availability_90":"availability_90_previousQ",
                                      'number_of_reviews': 'number_of_reviews_till_previousQ'})
        df = df.merge(df_q1, on=['host_id', 'id'], how='left')

        #无匹配的reactive_host+new_host填0
        df["number_of_reviews_till_previousQ"]=df["number_of_reviews_till_previousQ"].fillna(0)
        # 0 表示上季度为止都没有评论，或上季度无匹配


    #clip 就是 裁剪数值的上下限:
    # + 本季度也没有评论/增长
    df["number_of_reviews_thisQ"] = (
        df["number_of_reviews"] - df["number_of_reviews_till_previousQ"]
    ).clip(lower=0)
    

    df['booking_rate_l90d_previousQ'] = df.apply(
                lambda row: min(row['number_of_reviews_thisQ'] / row['availability_90_previousQ'], 1.0)
                if row['availability_90_previousQ'] > 0 else None,#没有开放，不参与计算？
                axis=1
            )    
    
    ##增加一种reactive host:在Q1中ava90==0，Q2开放
    # old_host中分出新的一类reopen_host
    mask = (
        (df['availability_90_previousQ'] == 0) &
        (df['availability_90']!=0)
    )
    # print("before include ava90: ", len(df[df["status"]=="reactive_host"]))
    df.loc[mask, 'status'] = 'reopen_host'
    # print('after: ',len(df[df["status"]=="reactive_host"]))
    
    #==============================================SPLIT==============================================
    print("split  changed and stable hosts".center(100,'-'))
    print(f"[INFO] 4 criterions : \n"
        f"- NEW hosts enter between '{year}-01-01'~'{year}-06-30';\n"
        f"- REACTIVED hosts : host_since before Q1+Q2; active not in Q1, but in Q2 ;\n"
        f"- REOPEN host: ava90_previousQ==0, ava90!=0 "
        f"=> ADD 'status_changed'==1\n\n"
        
        f"-BIO change in Q2(include new host);\n"
        f"-PIC change in Q2(include new host);\n"
        f"=> ADD 'sp_changed','old_host_sp_changed', regardless of status\n\n"
        
        f"==> ADD 'is_changed'==1\n")

    #----------------------summary-----------------------
    dfQ2=df[df['scraped_date']==f'{year}Q2']   
    
    def get_change_levels(df):
        # status
        df['status_changed'] = (
            (df['status'].isin(['new_host', 'reactive_host','reopen_host']))
        ).astype(int) # return T/F=>1/0
        
        #SP 
        df['sp_changed'] = (
            (df['host_about_changed']==1)|
            (df['host_picture_url_changed']==1)
        ).astype(int)
        
        # sp_changed-new
        df['old_host_sp_changed'] = (
            (df['sp_changed']==1)&
            (df['status_changed']==0)
        ).astype(int)
        
        #overall
        df['is_changed'] = (
            (df['status_changed']==1)|
            (df['sp_changed']==1)
        ).astype(int)
        return df
    
    dfQ2=get_change_levels(dfQ2)
    

    print(f"[INFO] host status COUNT in Q2:\n {dfQ2.status.value_counts(dropna=False)}\n\n"
          f"RATIO : {dfQ2.status.value_counts(dropna=False)/len(dfQ2)}\n") 
      
    print(f"[INFO] host SP \n"
          f"{dfQ2.old_host_sp_changed.value_counts(dropna=False)}\n")

    print(f"[INFO] OVERALL changed : {len(dfQ2[dfQ2['is_changed']==1])}; stable : {len(dfQ2[dfQ2['is_changed']==0])}\n"
        f" changed : {len(dfQ2[dfQ2['is_changed']==1])/len(dfQ2):.2f}%; stable : {len(dfQ2[dfQ2['is_changed']==0])/len(dfQ2):.2f}%\n")
    print('-'*100)


    print(f"booking_rate_l90d :\n"
        f"[info] \n"
        f"no match/no ava(isna), include all status_changed_host: {dfQ2['booking_rate_l90d_previousQ'].isna().value_counts()/len(dfQ2)}\n"
        f"{dfQ2['booking_rate_l90d_previousQ'].describe(include='all')}\n")
  
         
    # ------------------save---------------------
    end_time=time.time()
    print(f"[RUNTIME] done in {end_time-start_time:.2f} sec!")
    
    if save :
        os.makedirs(output_folder, exist_ok=True)
        if filename ==None:
            filename=os.path.basename(path_Q2).replace('.csv', '_split.csv')
        outpath_df=os.path.join(output_folder,filename)
        dfQ2.to_csv(outpath_df,index=False)
        print(f"✅[SAVE] dfQ2 split saved to {outpath_df}!")
    return dfQ2


