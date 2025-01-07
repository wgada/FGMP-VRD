from clip import clip
from clip.clip import  _transform
from dataloaders.gqa import GQADataLoader, GQA
import numpy as np
from torch import optim
import torch
import time
import os
import sys
from config import ModelConfig, BOX_SCALE, IM_SCALE
from torch.nn import functional as F
from lib.pytorch_misc import optimistic_restore, de_chunkize, clip_grad_norm
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator, calculate_mR_from_evaluator_list
from lib.pytorch_misc import print_para
from torch.optim.lr_scheduler import ReduceLROnPlateau
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
conf = ModelConfig()
from tqdm import tqdm
import torch.nn as nn
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from PIL import Image
import pickle
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
import cv2
import matplotlib
import matplotlib.pyplot as plt
idx_to_predicate= {'1': 'on', '2': 'wearing', '3': 'of', '4': 'near', '5': 'in', '6': 'behind', '7': 'in front of', '8': 'holding', '9': 'next to', '10': 'above', '11': 'on top of', '12': 'below', '13': 'by', '14': 'with', '15': 'sitting on', '16': 'on the side of', '17': 'under', '18': 'riding', '19': 'standing on', '20': 'beside', '21': 'carrying', '22': 'walking on', '23': 'standing in', '24': 'lying on', '25': 'eating', '26': 'covered by', '27': 'looking at', '28': 'hanging on', '29': 'at', '30': 'covering', '31': 'on the front of', '32': 'around', '33': 'sitting in', '34': 'parked on', '35': 'watching', '36': 'flying in', '37': 'hanging from', '38': 'using', '39': 'sitting at', '40': 'covered in', '41': 'crossing', '42': 'standing next to', '43': 'playing with', '44': 'walking in', '45': 'on the back of', '46': 'reflected in', '47': 'flying', '48': 'touching', '49': 'surrounded by', '50': 'covered with', '51': 'standing by', '52': 'driving on', '53': 'leaning on', '54': 'lying in', '55': 'swinging', '56': 'full of', '57': 'talking on', '58': 'walking down', '59': 'throwing', '60': 'surrounding', '61': 'standing near', '62': 'standing behind', '63': 'hitting', '64': 'printed on', '65': 'filled with', '66': 'catching', '67': 'growing on', '68': 'grazing on', '69': 'mounted on', '70': 'facing', '71': 'leaning against', '72': 'cutting', '73': 'growing in', '74': 'floating in', '75': 'driving', '76': 'beneath', '77': 'contain', '78': 'resting on', '79': 'worn on', '80': 'walking with', '81': 'driving down', '82': 'on the bottom of', '83': 'playing on', '84': 'playing in', '85': 'feeding', '86': 'standing in front of', '87': 'waiting for', '88': 'running on', '89': 'close to', '90': 'sitting next to', '91': 'swimming in', '92': 'talking to', '93': 'grazing in', '94': 'pulling', '95': 'pulled by', '96': 'reaching for', '97': 'attached to', '98': 'skiing on', '99': 'parked along', '100': 'hang on'}
predicate_to_idx={}
for k,v in idx_to_predicate.items():
    predicate_to_idx[v]=int(k)
predicate_labels_26=['holding', 'sitting on', 'riding', 'standing on', 'carrying', 'walking on', 'standing in', 'lying on', 'eating', 'covered by', 'looking at', 'hanging on', 'covering', 'sitting in', 'parked on', 'watching', 'flying in', 'hanging from', 'using', 'sitting at', 'covered in', 'standing next to', 'playing with', 'walking in', 'reflected in', 'flying']
id_predicate_labels_26=[]
for i in predicate_labels_26:
    id_predicate_labels_26.append(predicate_to_idx[i])
idx_to_label={'1': 'window', '2': 'tree', '3': 'man', '4': 'shirt', '5': 'wall', '6': 'building', '7': 'person', '8': 'ground', '9': 'sky', '10': 'leg', '11': 'sign', '12': 'hand', '13': 'head', '14': 'pole', '15': 'grass', '16': 'hair', '17': 'car', '18': 'ear', '19': 'eye', '20': 'woman', '21': 'clouds', '22': 'shoe', '23': 'table', '24': 'leaves', '25': 'wheel', '26': 'door', '27': 'pants', '28': 'letter', '29': 'people', '30': 'flower', '31': 'water', '32': 'glass', '33': 'chair', '34': 'fence', '35': 'arm', '36': 'nose', '37': 'number', '38': 'floor', '39': 'rock', '40': 'jacket', '41': 'hat', '42': 'plate', '43': 'tail', '44': 'leaf', '45': 'face', '46': 'bush', '47': 'shorts', '48': 'road', '49': 'bag', '50': 'sidewalk', '51': 'tire', '52': 'helmet', '53': 'snow', '54': 'boy', '55': 'umbrella', '56': 'logo', '57': 'roof', '58': 'boat', '59': 'bottle', '60': 'street', '61': 'plant', '62': 'foot', '63': 'branch', '64': 'post', '65': 'jeans', '66': 'mouth', '67': 'cap', '68': 'girl', '69': 'bird', '70': 'banana', '71': 'box', '72': 'bench', '73': 'mirror', '74': 'picture', '75': 'pillow', '76': 'book', '77': 'field', '78': 'glove', '79': 'clock', '80': 'dirt', '81': 'bowl', '82': 'bus', '83': 'neck', '84': 'trunk', '85': 'wing', '86': 'horse', '87': 'food', '88': 'train', '89': 'kite', '90': 'paper', '91': 'shelf', '92': 'airplane', '93': 'sock', '94': 'house', '95': 'elephant', '96': 'lamp', '97': 'coat', '98': 'cup', '99': 'cabinet', '100': 'street light', '101': 'cow', '102': 'word', '103': 'dog', '104': 'finger', '105': 'giraffe', '106': 'mountain', '107': 'wire', '108': 'flag', '109': 'seat', '110': 'sheep', '111': 'counter', '112': 'skis', '113': 'zebra', '114': 'hill', '115': 'truck', '116': 'bike', '117': 'racket', '118': 'ball', '119': 'skateboard', '120': 'ceiling', '121': 'motorcycle', '122': 'player', '123': 'surfboard', '124': 'sand', '125': 'towel', '126': 'frame', '127': 'container', '128': 'paw', '129': 'feet', '130': 'curtain', '131': 'windshield', '132': 'traffic light', '133': 'horn', '134': 'cat', '135': 'child', '136': 'bed', '137': 'sink', '138': 'animal', '139': 'donut', '140': 'stone', '141': 'tie', '142': 'pizza', '143': 'orange', '144': 'sticker', '145': 'apple', '146': 'backpack', '147': 'vase', '148': 'basket', '149': 'drawer', '150': 'collar', '151': 'lid', '152': 'cord', '153': 'phone', '154': 'pot', '155': 'vehicle', '156': 'fruit', '157': 'laptop', '158': 'fork', '159': 'uniform', '160': 'bear', '161': 'fur', '162': 'license plate', '163': 'lady', '164': 'tomato', '165': 'tag', '166': 'mane', '167': 'beach', '168': 'tower', '169': 'cone', '170': 'cheese', '171': 'wrist', '172': 'napkin', '173': 'toilet', '174': 'desk', '175': 'dress', '176': 'cell phone', '177': 'faucet', '178': 'blanket', '179': 'screen', '180': 'watch', '181': 'keyboard', '182': 'arrow', '183': 'sneakers', '184': 'broccoli', '185': 'bicycle', '186': 'guy', '187': 'knife', '188': 'ocean', '189': 't-shirt', '190': 'bread', '191': 'spots', '192': 'cake', '193': 'air', '194': 'sweater', '195': 'room', '196': 'couch', '197': 'camera', '198': 'frisbee', '199': 'trash can', '200': 'paint'}
def load_clip_to_cpu():
    backbone_name = "ViT-B/32"
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url,'/home/ths/zero-shot-clip/neural-motifs')

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model, _transform(model.visual.input_resolution)

class CustomCLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model
        #self.logit_scale = clip_model.logit_scale
        
    def zero_shot_forward(self, image, prompts):
        with torch.no_grad():
            text_features = self.clip.encode_text(prompts)
            image_features = self.clip.encode_image(image)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        #logit_scale = self.logit_scale.exp()
        #similarity=logit_scale * image_features @ text_features.t()
        return similarity

def crop_union_region(image, bbox1, bbox2):
    # bbox 格式为 (x1, y1, x2, y2)，其中 (x1, y1) 为左上角坐标，(x2, y2) 为右下角坐标
    x1_union = int(min(bbox1[0], bbox2[0]))
    y1_union = int(min(bbox1[1], bbox2[1]))
    x2_union = int(max(bbox1[2], bbox2[2]))
    y2_union = int(max(bbox1[3], bbox2[3]))
    #print(x1_union,y1_union,x2_union,y2_union)
    # 裁剪并返回并集区域的图像
    cropped_image = image[y1_union:y2_union, x1_union:x2_union]
    #print(cropped_image.shape)
    return cropped_image
def mask_sub_obj_region(image,bbox1,bbox2):
    mask = np.zeros_like(image[:,:,0])
    mask[int(bbox1[1]):int(bbox1[3]),int(bbox1[0]):int(bbox1[2])] = 1 #bbox1(x1,y1,x2,y2)
    mask[int(bbox2[1]):int(bbox2[3]),int(bbox2[0]):int(bbox2[2])] = 1
    masked_img = cv2.bitwise_and(image,image,mask=mask)
    return masked_img
def dark_sub_obj_region(image,bbox1,bbox2):
    image_dark=image*0.4
    image_dark = cv2.convertScaleAbs(image_dark)
    image_dark[int(bbox1[1]):int(bbox1[3]),int(bbox1[0]):int(bbox1[2])] = image[int(bbox1[1]):int(bbox1[3]),int(bbox1[0]):int(bbox1[2])] #bbox1(x1,y1,x2,y2)
    image_dark[int(bbox2[1]):int(bbox2[3]),int(bbox2[0]):int(bbox2[2])] = image[int(bbox2[1]):int(bbox2[3]),int(bbox2[0]):int(bbox2[2])]
    #cv2.rectangle(image_dark, (int(bbox1[0]), int(bbox1[1])), (int(bbox1[2]), int(bbox1[3])), (0,0,255), 1)
    #cv2.rectangle(image_dark, (int(bbox2[0]), int(bbox2[1])), (int(bbox2[2]), int(bbox2[3])), (255,0,0), 1)
    return image_dark
sam_checkpoint = "/home/ths/zero-shot-clip/sam_vit_h_4b8939.pth"
model_type = "default"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
def sam_sub_obj_region(image,bbox1,bbox2): 
    image_copy = np.copy(image)
    predictor.set_image(image_copy)
    input_boxes = torch.tensor([bbox1,bbox2], device=predictor.device)  
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image_copy.shape[:2])
    masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
    mask=torch.sum(masks,dim=0)
    mask=mask.reshape(mask.shape[1],mask.shape[2])
    mask=mask.cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    blurred_image = cv2.GaussianBlur(image_copy, (0, 0), sigmaX=5)  # 调整sigmaX参数以控制模糊程度
    image_copy[mask == 0] = blurred_image[mask == 0] 
    return image_copy

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = load_clip_to_cpu( )
clip_model.to(device)
model = CustomCLIP(clip_model)
model.to(device)
train, val, test = GQA.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                          use_proposals=conf.use_proposals,
                          filter_non_overlap=False)
if conf.test:
    val = test
train_loader, val_loader = GQADataLoader.splits(train, val, mode='rel',
                                               batch_size=1,
                                               num_workers=0,
                                               num_gpus=1)
model.eval()
evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=False)
evaluator_list = []
for index,name in zip(id_predicate_labels_26,predicate_labels_26):
    evaluator_list.append((index, name, BasicSceneGraphEvaluator.all_modes()))
for val_b, batch in enumerate(tqdm(val_loader)):
    img_input = batch["img"][0]
    img_input = img_input.permute(1, 2, 0)
    img_input = img_input.numpy()
    img_input = (img_input * 255).astype(np.uint8)
    image = cv2.cvtColor(img_input, cv2.COLOR_RGB2BGR)
    obj_label = batch["gt_classes"][0] #object labels
    obj_pair=batch["gt_relations"][0]  #关系对
    obj_boxes=batch['gt_boxes'][0]
    obj_boxes_resize=obj_boxes*IM_SCALE / BOX_SCALE
    sub_obj_img=[]
    obj_pair_del=[]
    pred_rel_inds=[]
    for m in range(obj_pair.shape[0]):
        if idx_to_predicate[str(obj_pair[m][2])] not in predicate_labels_26: 
            obj_pair_del.append(m)  
    obj_pair_new = np.delete(obj_pair, obj_pair_del, axis=0)
    if obj_pair_new.shape[0]!=0:    
        for i in range(obj_boxes.shape[0]):
            for j in range(obj_boxes.shape[0]):
                processed_image = crop_union_region(image, obj_boxes_resize[i], obj_boxes_resize[j])
                if 0 in processed_image.shape:
                    continue
                processed_image = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)) 
                image_input = preprocess(processed_image).unsqueeze(0).to(device) 
                sub_obj_img.append(image_input)
                pred_rel_inds.append([i,j])
        pred_rel_inds=np.array(pred_rel_inds)
        sub_obj_img_tensor=torch.cat(sub_obj_img,dim=0) #一张图像所有可能主宾对的图像输入   
        prompts =torch.cat([clip.tokenize(f"{pred}") for pred in predicate_labels_26]).to(device) 
        similarity=model.zero_shot_forward(sub_obj_img_tensor,prompts)
        rel_cores=similarity.to('cpu').numpy()
        gt_entry = {
            'gt_classes': obj_label,
            'gt_relations': obj_pair_new,
            'gt_boxes': obj_boxes,
        }
    
        pred_entry = {
            'pred_rel_inds': pred_rel_inds,
            'rel_scores': rel_cores,
        }
        evaluator[conf.mode].evaluate_scene_graph_entry(
            gt_entry,
            pred_entry,
            predicate_labels_26,
            predicate_to_idx,

        )
        for (pred_id, _, evaluator_rel) in evaluator_list:
            gt_entry_rel = gt_entry.copy()
            mask = np.in1d(gt_entry_rel['gt_relations'][:, -1], pred_id)
            gt_entry_rel['gt_relations'] = gt_entry_rel['gt_relations'][mask, :]
            if gt_entry_rel['gt_relations'].shape[0] == 0:
                continue
            evaluator_rel[conf.mode].evaluate_scene_graph_entry(
                gt_entry_rel,
                pred_entry,
                predicate_labels_26,
                predicate_to_idx,
        )

recall=evaluator[conf.mode].print_stats()
mean_recall = calculate_mR_from_evaluator_list(evaluator_list, conf.mode)
result={}
result['recall']=recall
result['mean_recall']=mean_recall
#with open('/home/ths/zero-shot-clip/neural-motifs/result/GQA/cls_crop_baseline.pickle', 'wb') as file:
#    pickle.dump(result, file)


