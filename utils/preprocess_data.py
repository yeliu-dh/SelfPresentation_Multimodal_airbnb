import pandas as pd
import os 
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