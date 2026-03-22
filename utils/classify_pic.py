import os, sys, pathlib, importlib
import pandas as pd
import json
import time
from tqdm import tqdm

import cv2 # pip install opencv-python
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import torch

import numpy as np
from transformers import CLIPModel, CLIPProcessor
from PIL import Image


#==================================================FACE===================================================
# # 初始化一次即可（不要在循环里初始化）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

mtcnn = MTCNN(keep_all=True, device=device)


def analyze_background_semantic(img_rgb, bbox_list):
    """
    通过life/pro标签，用clip检测背景类型。
    
    不考虑人脸，仅考虑背景，
    计算和life/pro组标签相似度的平均值哪个更高。
    """
        
    h, w, _ = img_rgb.shape
    mask = np.ones((h, w), dtype=np.uint8)

    # -- faces zone ---
    for (x1, y1, x2, y2) in bbox_list:
        mask[y1:y2, x1:x2] = 0

    # ---black face zones---
    bg_img = img_rgb.copy()
    bg_img[mask == 0] = 0

    
    # ---prompt---
    clean_prompts = [
        "a close-up headshot with a neutral wall background",
        "a portrait photo focused on a person with no other objects",
        "a professional studio profile picture"
    ]
    lifestyle_prompts = [
        "a person traveling outdoors",
        "a person on vacation in a city or nature scene",
        "a person doing sports or leisure activities",
        "a family or social gathering indoors or outdoors"
    ]
    prompts = clean_prompts + lifestyle_prompts

    
    # ---classify type of bg---
    bg_pil = Image.fromarray(bg_img)
    inputs = clip_processor(
        text=prompts,
        images=bg_pil,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits = outputs.logits_per_image
        probs = logits.softmax(dim=1).cpu().numpy()[0]

    clean_score = probs[:len(clean_prompts)].mean()
    lifestyle_score = probs[len(clean_prompts):].mean()

    is_lifestyle = lifestyle_score > clean_score

    return {
        "clean_score": float(clean_score),
        "lifestyle_score": float(lifestyle_score),
        "is_lifestyle_background": bool(is_lifestyle)
    }




def classify_pic_type(img_path):
    host_id = os.path.splitext(os.path.basename(img_path))[0]
    host_id=int(host_id)
    img=cv2.imread(img_path)
    
    # --init--
    # face：
    has_face, nb_face, face_area_ratio, avg_face_prob = 0, 0, 0, 0
    bbox_list=[]
    clean_score, lifestyle_score, host_picture_type=0, 0, "no_person" 
    # deepface:
    age, age_class, gender, dominant_emotion=None,None,None,None
    smile_score, is_smiling=0,0
    

    if not img is None:# 若为none，不更新
        img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _=img_rgb.shape
        img_area=h * w
        
        # ---face--- 
        boxes, probs = mtcnn.detect(img_rgb, landmarks=False) 
        
        # 若检测到人脸, 覆盖初始化结果:
        if boxes is not None and len(boxes) > 0: # nb_face!=0 
            has_face = 1 
            nb_face = len(boxes) 
            total_face_area = 0 
            valid_probs = [] 
            
            for box, prob in zip(boxes, probs): 
                if prob is None: 
                    continue
                x1, y1, x2, y2 = box.astype(int) # 防止越界 
                x1 = max(0, x1) 
                y1 = max(0, y1) 
                x2 = min(w, x2) 
                y2 = min(h, y2)
                area = (x2 - x1) * (y2 - y1) 
                total_face_area += area
                
                bbox_list.append([int(x1), int(y1), int(x2), int(y2)]) 
                valid_probs.append(prob) 
                face_area_ratio = total_face_area / img_area 
                avg_face_prob = sum(valid_probs) / len(valid_probs) if valid_probs else 0
            
            if nb_face>1:
                host_picture_type='life_style'
            
            else :# nb_face==1
                bg_semantic = analyze_background_semantic(img_rgb, bbox_list)
                clean_score=bg_semantic['clean_score']
                lifestyle_score=bg_semantic['lifestyle_score']
                host_picture_type="pro_style" if clean_score > lifestyle_score else "life_style"
        
    # --- save ---
    res = {
        "img_path": img_path,
        "host_id": host_id,
        "has_face": has_face,
        "nb_face": nb_face,
        "face_area_ratio": face_area_ratio,
        "avg_face_prob": avg_face_prob,
        "bbox_list": bbox_list,
        "clean_score":clean_score,
        "lifestyle_score": lifestyle_score,
        "host_picture_type":host_picture_type,

        # deepface
        "age": age,
        "age_class": age_class,
        "gender": gender,
        "smile_score": smile_score,
        "is_smiling": is_smiling,
        "dominant_emotion": dominant_emotion       
        }
    
    return res
    
  


def run_clf(pic_folder, path_results,save_interval=1000, n_sample=10):
    # ---detector---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)

    # ---no repetition---
    if os.path.exists(path_results):
        
        with open(path_results, 'r', encoding='utf-8')as f:
            results=json.load(f)
        
        processed=[int(result['host_id']) for result in results]
        pending_files=[os.path.join(pic_folder, f) for f in os.listdir(pic_folder) if f.endswith('.jpg') and int(os.path.splitext(f)[0]) not in processed]

        print(f"[info] {len(processed)} / {len(os.listdir(pic_folder))}  already classified; {len(pending_files)} / {len(os.listdir(pic_folder))} pic to classify")
    else :
        results=[]
        pending_files=[os.path.join(pic_folder, f) for f in os.listdir(pic_folder) if f.endswith('.jpg')]
        print(f"[info] {len(pending_files)} / {len(os.listdir(pic_folder))} pic to classify")
        

    # ---n_sample---
    if n_sample:
        pending_files=pending_files[:n_sample]
        
    if len(pending_files)>0:
        
        # ---clf---
        start_time=time.time()
        for i, img_path in enumerate(tqdm(pending_files, total=len(pending_files), desc="processing images...")):
            # ---save interval---
            if (i+1) % save_interval==0:
                with open(path_results, 'w', encoding="utf-8")as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                    print(f"[interval save] {i+1} / {len(pending_files)} results saved!\n")
                    
            results.append(classify_pic_type(img_path))
        end_time=time.time()
                
        # ---final save---
        with open(path_results, 'w', encoding="utf-8")as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"[final save] results saved to {path_results}\n")
        print(f"[DONE] {len(results)} images classified : {end_time-start_time:.2f} sec!")

    else :
        print(f"[info] all classified!")
        
    return results

