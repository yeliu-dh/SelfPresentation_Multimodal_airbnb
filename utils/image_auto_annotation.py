import json
import pandas as pd
import torch
import pickle
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import os,sys
import time
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image
import cv2
from sklearn.metrics import f1_score, classification_report


##======================================AUTOCLF=======================================================
##=====================================2_test_autoclf.ipynb===========================================

def read_json(json_path):
    with open(json_path, 'r', encoding='utf-8')as f:
        data=json.load(f)
    return data

def save_json(data, json_path):
    with open(json_path, 'w', encoding='utf-8')as f:
        json.dumps(data, f, ensure_ascii=False, indent=2)
    return 


##================================ EMBEDDINGS + FEW SHOT LABELS================================== 
# device='cuda' if torch.cuda.is_available() else "cpu" 
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_text_by_clip(text_list, device, model, processor):
    """
    text_list: list of strings
    return: np.array of shape (len(text_list), embedding_dim)
    label 被省去，只留下具体描述
    """       
    with torch.no_grad():
        inputs = processor(text=text_list, return_tensors="pt", padding=True).to(device)
        text_features = model.get_text_features(**inputs)  # (N, 512)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().astype("float32")



def embed_labels_txt_img(labels_prompt_path='labels/labels_prompt.json',
                         image_emb_path="embeddings_SAMPLE/emb_SAMPLE.npz",
                         labels_emb_path="labels/labels_emb_txt-img.pkl"):
    print(f"======================EMBEDDING TEXT + FEW SHOT IMG=======================\n") 
    start_time_prompt=time.time()
    # ---- 0) 初始化 CLIP ----
    device='cuda' if torch.cuda.is_available() else "cpu" 
    model_name="openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    # ---- 1) 读取 labels prompt ----
    labels_prompt = read_json(labels_prompt_path)

    # ---- 2) 加载 image embeddings ----
    img_embs = np.load(image_emb_path)

    # ---- 3) 构建最终 labels_emb ----
    labels_emb = {}
    for category, cls_dict in labels_prompt.items():
        labels_emb[category] = {}

        for cls_name, content in cls_dict.items():
            text = content["text"]
            images = content.get("images", [])

            # 文本 embedding
            text_emb = embed_text_by_clip([text], device, model, processor)  # shape (1,512)

            # 图片 embedding
            img_emb_list = []
            for fname in images:
                if fname in img_embs:
                    img_emb_list.append(img_embs[fname][None])  # shape (1,512)
                else:
                    print(f"[WARNING] {fname} not found in {image_emb_path}")

            if img_emb_list:
                img_emb_stack = np.vstack(img_emb_list)
                all_emb = np.vstack([text_emb, img_emb_stack])  # text + images
            else:
                all_emb = text_emb

            labels_emb[category][cls_name] = all_emb

    # ---- 4) 保存为 pkl ----
    os.makedirs(os.path.dirname(labels_emb_path), exist_ok=True)
    with open(labels_emb_path, "wb") as f:
        pickle.dump(labels_emb, f)
    end_time_prompt=time.time()


    print(f"[SUCCES] labels text-image prompt embeded by '{model_name}' saved to '{labels_emb_path}' : {end_time_prompt-start_time_prompt:.2f} sec!\n")
    return labels_emb


##==========================================PREDICTION========================================

# ------------------------
# 预测函数
# ------------------------
def is_default_pic(image_emb, default_pic_emb, threshold=0.95):
    """
    image_emb: np.array (512,)
    default_emb: np.array (512,)
    threshold: cosine similarity threshold
    """
    sim = image_emb @ default_pic_emb  # cosine similarity, embeddings 已经 L2-normalized
    return "1" if sim >= threshold else "0"


def few_shot_predict(img_emb, support_dict):
    """
    img_emb: (512,)
    support_dict: dict[class_name -> (N,512)]  
        每个类别可以有文本embedding和示例图片embedding
    返回：类别名
    """
    best_cls = None
    best_sim = -999

    for cls_name, emb_set in support_dict.items():  # emb_set shape = (K,512)
        sims = img_emb @ emb_set.T                  # → (K,)
        score = sims.max()                          # 取最大相似度

        if score > best_sim:
            best_sim = score
            best_cls = cls_name

    return best_cls

def predict_defaut_type_smile_sex(image_emb_path = "embeddings_SAMPLE/emb_SAMPLE.npz",
                                labels_emb_path = "labels/labels_emb_txt-img.pkl",
                                prediction_path = "annotations_SAMPLE/autoclf_predictions.json"
                                ):
    
    print(f"============================PREDICTION==============================\n"
          f"-is_default_pic:'1'/'0'\n"
          f"-type : life/pro/UNK (text prompt+few-shot)\n"
          f"-is_smiling: '1'/'0'/'UNK'\n"
          f"-sex: :M/F/MIX/UNK \n")
    start_time=time.time()

    # ------------------------
    # 读取 embeddings
    # ------------------------
    image_embs = np.load(image_emb_path)  # keys: "host_id.jpg"

    with open(labels_emb_path, "rb") as f:
        labels_emb = pickle.load(f)

    # with open(text_json_path, "r") as f:
    #     labels_text = json.load(f)
    default_pic_emb=image_embs["336591839.jpg"]

    # ------------------------
    # 遍历每张图片
    predictions = {}
    for fname in tqdm(image_embs.files, desc='predict on images...'):
        img_emb = image_embs[fname]

        pred = {}
        # ---- 1) 默认头像判断 ----
        pred["is_default_pic"] = is_default_pic(img_emb, default_pic_emb)

        # 如果是默认头像，直接覆盖其他标签
        if pred["is_default_pic"] == "1":
            pred.update({
                "type": "UNK",
                "is_smiling": "UNK",
                "sex": "UNK"
            })
            predictions[fname] = pred
            continue

        # ---- 2) few-shot 预测其他类别 ----
        for category in labels_emb.keys():
            if category == "is_default_pic":
                continue #跳过当前循环的剩余部分，直接进入下一次循环

            support_dict = labels_emb[category]  # dict[class_name -> (N,512)]
            best_cls = few_shot_predict(img_emb, support_dict)

            pred[category] = best_cls

        predictions[fname] = pred
  
    # 保存 prediction.json
    os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
    with open(prediction_path, "w") as f:
        json.dump(predictions, f, indent=2)

    end_time=time.time()
    print(f"[SUCCES] Auto predictions on {len(image_embs)} images saved → {prediction_path}: {end_time-start_time:.2f} sec!\n")
    return 




##==========================================DETECTION========================================

# ------------------------
# 识别 & 判断函数
# ------------------------
def get_quality_label(img_path, blur_threshold=700.0):
    """
    根据图像是否模糊判断质量
    img_path: 图片路径
    blur_threshold: Laplacian 方差阈值，值越小越模糊
    return: "high" 或 "low"
    """
    try:
        # 用 PIL 读取并转为灰度
        img = Image.open(img_path).convert("L")  
        img_np = np.array(img)

        # Laplacian 方差
        lap_var = cv2.Laplacian(img_np, cv2.CV_64F).var()

        if lap_var >= blur_threshold:
            return "high"
        else:
            return "low"
    except Exception as e:
        print(f"[ERROR] Failed to process {img_path}: {e}")
        return "low"

def detect_has_person(img_path):
    model = YOLO("yolov8n.pt")

    try:
        # results = model(img_path)[0]  # first result
        results = model(img_path, verbose=False)[0]  # 👈 关闭所有日志输出
        for box in results.boxes:
            cls = int(box.cls[0])
            if results.names[cls] == "person":
                return "1"
        return "0"
    except:
        return "UNK"

def detect_person_and_quality(
        blur_threshold,
        images_folder="images_SAMPLE",
        detection_path="annotations_SAMPLE/detections.json"
    ):
    print(f"============================DETECTION===============================")
    print(f"- has_person by yolo:'1'/'0',\n"
          f"- quality by Laplacian : high/low/UNK\n")
    start_time=time.time()

    os.makedirs(os.path.dirname(detection_path), exist_ok=True)
    records = {}

    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith((".jpg",".jpeg",".png"))]

    # print(f"[INFO] Found {len(image_files)} images in {images_folder}")

    for fname in tqdm(image_files, desc="detecting has_person and picture quality..."):
        fpath = os.path.join(images_folder, fname)

        # YOLO: detect person
        has_person = detect_has_person(fpath)

        # quality: size-based
        quality = get_quality_label(fpath, blur_threshold)

        records[fname] = {
            "has_person": has_person,
            "quality": quality
        }

        # print(f"[OK] {fname} → has_person={has_person}, quality={quality}")

    # save json
    os.makedirs(os.path.dirname(detection_path), exist_ok=True)

    with open(detection_path, "w") as f:
        json.dump(records, f, indent=2)

    end_time=time.time()
    print(f"[SUCCES] {len(image_files)} detection results saved to {detection_path} {end_time-start_time:.2f} sec!\n")
    return 







##==========================================MERGE + OVERRIDE========================================
def apply_override_rules (annos):
    """
    {
    "is_default_pic": "1",
    "type": "UNK",
    "is_smiling": "UNK",
    "sex": "UNK",
    "has_person": "0",
    "quality": "UNK"
    },
    {...}

    """
    if annos['is_default_pic']==1:
        return {
            "is_default_pic": "1",
            "type": "UNK",
            "is_smiling": "UNK",
            "sex": "UNK",
            "has_person": "0",
            "quality": "low"
            }       

    elif annos['has_person']==0:#no person
        return { 
            "is_default_pic": "0",
            "type": "UNK",
            "is_smiling": "UNK",
            "sex": "UNK",
            "has_person": "0",
            "quality": annos['quality']
        }
        
    else :
        return annos


def merge_annotations(detection_path="annotations_SAMPLE/detections.json",
                    prediction_path="annotations_SAMPLE/autoclf_predictions.json",
                    auto_annotations_path="annotations_SAMPLE/auto_annotations.json"
                ):
    print(f"==============================MERGE=============================\n" 
          f"merging detections and predictions...\n")
    
    detections=read_json(detection_path)
    predictions=read_json(prediction_path)
    print(f"[CHECK] keys alignement: {predictions.keys()==detections.keys()}")# check

    merged = {}
    for fname in predictions.keys():
        merged[fname] = {}  # 新建子字典

        # 1. 先放 CLIP/few-shot 预测结果
        if fname in predictions:
            merged[fname].update(predictions[fname])

        # 2. 再放 YOLO detections
        if fname in detections:
            merged[fname].update(detections[fname])

        # 3. OVERRIDE:
        updated_annos=apply_override_rules(merged[fname])
        merged[fname]=updated_annos

    # 4. 保存为 json
    os.makedirs(os.path.dirname(auto_annotations_path), exist_ok=True)

    with open(auto_annotations_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"[SAVE] merged and overrided auto-annotations saved in {auto_annotations_path}!\n")
    return 


##==========================================EVALUATION========================================
# def ls_to_dict(ls_json):
#     """
#     ls_json: list, LabelStudio 导出 json
#     return: dict {filename: {dim:value}}
#     """
#     out = {}
#     for item in ls_json:
#         # 找到图片名
#         fname = item.get("data", {}).get("filename") or item.get("file_upload")
#         if not fname:
#             continue
        
#         # 取第一条标注（annotations 可能有多人标注，取第一个即可）
#         ann_list = item.get("annotations", [])
#         if not ann_list:
#             continue
        
#         ann_result = ann_list[0].get("result", [])
#         labels = {}
#         for r in ann_result:
#             dim = r.get("from_name")
#             value_dict = r.get("value", {})
#             # choices 是 LabelStudio 默认格式
#             if "choices" in value_dict and len(value_dict["choices"]) > 0:
#                 labels[dim] = value_dict["choices"][0]
#             else:
#                 labels[dim] = "UNK"  # 没有标注就填 UNK

#         out[fname] = labels
#     return out


def check_label_consistency(ls_annos, auto_annos):
    """
    检查 ground truth (ls_annos) 与 auto_annos 之间的分类标签是否一致。
    
    ls_annos: dict {filename: {dim:label}}
    auto_annos: dict {filename: {dim:label}}

    功能：
    - 自动收集每个 dimension 在两者中的所有类别集合
    - 如果不一致，打印 warning
    - 返回 True(一致) or False(发现不一致)
    """
    print("=== Checking label consistency between ls_annos and auto_annos ===")

    all_dims = set()
    for d in ls_annos.values():
        all_dims.update(d.keys())
    for d in auto_annos.values():
        all_dims.update(d.keys())

    has_issue = False

    for dim in sorted(all_dims):
        # 收集 ls 中该维度的所有类别
        ls_labels = set(
            ann.get(dim, "UNK") for ann in ls_annos.values()
        )

        # 收集 auto 中该维度的所有类别
        auto_labels = set(
            ann.get(dim, "UNK") for ann in auto_annos.values()
        )

        # 检查是否一致
        if ls_labels != auto_labels:
            has_issue = True
            print(f"\n [WARNING] Label mismatch in dimension: '{dim}'")
            print(f"   LS annotations labels:   {sorted(ls_labels)}")
            print(f"   Auto annotations labels: {sorted(auto_labels)}")

            # 计算差异
            only_in_ls = ls_labels - auto_labels
            only_in_auto = auto_labels - ls_labels

            if only_in_ls:
                print(f"   → Labels only in LS (missing in auto): {sorted(only_in_ls)}")
            if only_in_auto:
                print(f"   → Labels only in Auto (missing in LS): {sorted(only_in_auto)}")

            print("   Suggestion: Please check label names / mapping or CLIP text prompts.\n")

    if not has_issue:
        print("✓ All label sets match perfectly across all dimensions.\n")

    return not has_issue



def ls_to_dict(ls_json):
    """
    ls_json: list, LabelStudio 导出 json
    return: dict {filename: {dim:value}}
    """

    # ------- NEW: type 标签映射 -------
    type_mapping = {
        "pro": "identity_style",
        "life": "lifestyle",
        "UNK": "UNK",
        None: "UNK"
    }

    out = {}
    for item in ls_json:
        # 获取文件名
        fname = item.get("data", {}).get("filename") or item.get("file_upload")
        if not fname:
            continue
        
        ann_list = item.get("annotations", [])
        if not ann_list:
            continue
        
        ann_result = ann_list[0].get("result", [])
        labels = {}

        for r in ann_result:
            dim = r.get("from_name")
            value_dict = r.get("value", {})

            # 默认 UNK
            if "choices" in value_dict and len(value_dict["choices"]) > 0:
                raw_label = value_dict["choices"][0]
            else:
                raw_label = "UNK"

            # ------- NEW: 类型特定映射 -------
            if dim == "type":
                mapped_label = type_mapping.get(raw_label, "UNK")
                labels[dim] = mapped_label
            else:
                labels[dim] = raw_label

        out[fname] = labels

    return out



from sklearn.metrics import classification_report, f1_score
import pandas as pd

def summarize_classification(auto_annos, ls_annos, label_dims=None, verbose=True, low_flag_v=0.7):
    """
    auto_annos: dict {filename: {dim: label}}
    ls_annos: dict {filename: {dim: label}}  # ground truth
    label_dims: list of dimensions to evaluate
    verbose: if True, print detailed classification report when low_performance_flag=True

    F1:
    宏平均 (macro)：每个类别 F1 平均 → 类别不平衡敏感
    微平均 (micro)：全局 TP/FP/FN 计算 → 对样本量敏感
    返回 DataFrame，可直接打印或保存 CSV

    """
    if label_dims is None:
        label_dims = list(next(iter(ls_annos.values())).keys())

    records = []

    for dim in label_dims:
        y_true = []
        y_pred = []
        unk_count = 0

        for fname in ls_annos:
            true_label = ls_annos[fname].get(dim, "UNK")
            pred_label = auto_annos.get(fname, {}).get(dim, "UNK")

            y_true.append(true_label)
            y_pred.append(pred_label)

            if true_label == "UNK":
                unk_count += 1

        # Metrics
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
        acc = sum([t == p for t, p in zip(y_true, y_pred)]) / len(y_true)

        low_flag = micro_f1 < low_flag_v

        # 如果某个维度表现不好 → 打印详细分类情况
        if verbose and low_flag:
            print(f"\n----------------------------------------------")
            print(f"⚠️  Low performance detected for DIMENSION: **{dim}**")
            print(f"Accuracy={acc:.3f}, Macro-F1={macro_f1:.3f}, Micro-F1={micro_f1:.3f}")
            print("----------------------------------------------")
            print(classification_report(y_true, y_pred, zero_division=0))
        
        records.append({
            "dimension": dim,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "UNK_count": unk_count,
            "low_performance_flag": low_flag
        })

    return pd.DataFrame(records)


# def evalate_clf(ls_annotations_path="annotations_SAMPLE/ls_annotations.json",
#             auto_annotations_path="annotations_SAMPLE/auto_annotations.json"#y_pred
#             ):
#     ## evaluation of clip:
#     print(f"==============================EVALUATION============================\n"
#           f"ls_annotations : y_true,\n"
#           f"auto_annotations:y_pred\n")

#     #load annotations :
#     ls_annotations_brut=read_json(ls_annotations_path)#y_true
#     auto_annotations=read_json(auto_annotations_path)#y_pred

#     # parse ls_annotations:
#     ls_annotations = ls_to_dict(ls_annotations_brut)

#     #summary:
#     dims = ["is_default_pic","has_person","type", "quality","is_smiling","sex"]
#     df_summary=summarize_classification(auto_annos=auto_annotations, ls_annos=ls_annotations, label_dims=dims)
#     print("classification report:")
#     display(df_summary)
#     return 

def evalate_clf(low_flag_v,
                ls_annotations_path="annotations_SAMPLE/ls_annotations.json",
                auto_annotations_path="annotations_SAMPLE/auto_annotations.json"):

    print(f"==============================EVALUATION============================\n"
          f"ls_annotations : y_true,\n"
          f"auto_annotations:y_pred\n")

    ls_annotations_brut = read_json(ls_annotations_path)
    auto_annotations = read_json(auto_annotations_path)

    ls_annotations = ls_to_dict(ls_annotations_brut)
    check_label_consistency(ls_annotations, auto_annotations)

    dims = ["is_default_pic","has_person","type", "quality","is_smiling","sex"]
    df_summary = summarize_classification(low_flag_v=low_flag_v, 
                                          auto_annos=auto_annotations,
                                          ls_annos=ls_annotations,
                                          label_dims=dims,
                                          verbose=True
                                          )

    print(f"Classification summary: LOW performance flag <{low_flag_v}")
    display(df_summary)

    return 


def autoclf(images_folder,
            image_emb_path,
            labels_prompt_path, 
            labels_emb_path,
            prediction_path,
            detection_path,
            auto_annotations_path,
            ls_annotations_path,
            blur_threshold=700,
            low_flag_v=0.7, 
            update_prompt=False):
    n_images=len([f for f in os.listdir(images_folder) if f.endswith('.jpg')])

    all_start_time=time.time()    
    # 无修改则不需要重新embedding:
    if update_prompt==True:
        labels_emb=embed_labels_txt_img(labels_prompt_path,
                            image_emb_path,
                            labels_emb_path)

    predict_defaut_type_smile_sex(image_emb_path,
                                labels_emb_path,
                                prediction_path)
    
    detect_person_and_quality(
                            blur_threshold,
                            images_folder,
                            detection_path)
    
    merge_annotations(
                        detection_path,
                        prediction_path,
                        auto_annotations_path)
    
    if ls_annotations_path is not None:    
        evalate_clf(low_flag_v,
                    ls_annotations_path,
                    auto_annotations_path
                    )    
    all_end_time=time.time()
    #     # labels_prompt_path, 


    print(f"[PROCESS] Embeddings-> Prediction-> Detection-> Merge-> Evaluation \n"
        f"{n_images} images in {images_folder} :{all_end_time-all_start_time:.2f} sec! \n")
    return 

























##=========================================CLF============================================================##

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128):
        super().__init__()
        # shared layers
        self.shared_fc = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

        # heads for each task
        self.type_head = nn.Linear(hidden_dim, 3)       # Type: life/pro/else
        self.quality_head = nn.Linear(hidden_dim, 1)    # Quality: high/low
        self.smile_head = nn.Linear(hidden_dim, 1)      # Is_smiling: yes/no

    def forward(self, x):
        x = self.relu(self.shared_fc(x))
        type_out = self.type_head(x)          # logits for softmax
        quality_out = torch.sigmoid(self.quality_head(x))  # probability 0-1
        smile_out = torch.sigmoid(self.smile_head(x))      # probability 0-1
        return type_out, quality_out, smile_out

# 示例训练 loop 的损失
def multi_task_loss(type_logits, type_labels, quality_pred, quality_labels, smile_pred, smile_labels):
    type_loss = F.cross_entropy(type_logits, type_labels)
    quality_loss = F.binary_cross_entropy(quality_pred, quality_labels.float())
    smile_loss = F.binary_cross_entropy(smile_pred, smile_labels.float())
    return type_loss + quality_loss + smile_loss
