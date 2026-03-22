
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from collections import Counter
from deepface import DeepFace




# ======================================DEEPFACE================================================
def age_to_class(age):
    if age == "unk" or age is None:
        return None

    elif age < 30:
        return "young"
    elif age < 50:
        return "middle"
    else:
        return "senior"


def deepface_pic(img_path):
    try :
        analysis = DeepFace.analyze(
        img_path=img_path,
        actions=['age','gender','emotion'],
        enforce_detection=False,
        detector_backend="opencv",
        )

        # DeepFace 可能返回单个 dict 或列表
        if isinstance(analysis, dict):
            analysis = [analysis]

        ages, smiles, emotions, genders = [], [], [], []

        for face_analysis in analysis:
            ages.append(face_analysis.get("age", 0))
            smiles.append(face_analysis.get("emotion", {}).get("happy", 0))
            emotions.append(face_analysis.get("dominant_emotion", "neutral"))
            genders.append(face_analysis.get("dominant_gender", "unk"))

        # 取平均年龄
        age= int(sum(ages)/len(ages))
        age_class = age_to_class(age)
        
        # 平均微笑
        smile_score = float(sum(smiles)/len(smiles))
        is_smiling = 1 if smile_score > 40 else 0

        # 性别投票
        gender_counts = Counter(genders)
        if len(gender_counts) == 1:
            gender = genders[0]
        else:
            gender = "multi"

        # 主要情绪
        emotion_counts = Counter(emotions)
        dominant_emotion = emotion_counts.most_common(1)[0][0]
            
    except Exception as e :
        print(f"[error] from 'deepface' : {e}!\n")  
          
    return age, age_class, gender, smile_score, is_smiling


#deepface     
    