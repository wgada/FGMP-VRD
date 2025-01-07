from clip import clip
from clip.clip import  _transform,tokenize
from dataloaders.visual_genome import VGDataLoader, VG
import numpy as np
from torch import optim
import torch
import pandas as pd
import time
import os
import sys
from config import ModelConfig, BOX_SCALE, IM_SCALE
from torch.nn import functional as F
from lib.pytorch_misc import optimistic_restore, de_chunkize, clip_grad_norm
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator,calculate_mR_from_evaluator_list
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
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
import torch.backends.cudnn as cudnn
idx_to_predicate= {"1": "above", "2": "across", "3": "against", "4": "along", "5": "and", "6": "at", "7": "attached to", "8": "behind", "9": "belonging to", "10": "between", "11": "carrying", "12": "covered in", "13": "covering", "14": "eating", "15": "flying in", "16": "for", "17": "from", "18": "growing on", "19": "hanging from", "20": "has", "21": "holding", "22": "in", "23": "in front of", "24": "laying on", "25": "looking at", "26": "lying on", "27": "made of", "28": "mounted on", "29": "near", "30": "of", "31": "on", "32": "on back of", "33": "over", "34": "painted on", "35": "parked on", "36": "part of", "37": "playing", "38": "riding", "39": "says", "40": "sitting on", "41": "standing on", "42": "to", "43": "under", "44": "using", "45": "walking in", "46": "walking on", "47": "watching", "48": "wearing", "49": "wears", "50": "with"}
predicate_to_idx={}
for k,v in idx_to_predicate.items():
    predicate_to_idx[v]=int(k)
predicate_labels_24=["to", "carrying", "covered in", "covering", "eating", "flying in", "growing on", "hanging from", 
"holding", "laying on", "looking at", "lying on", "mounted on", "painted on", "parked on", "playing", "riding", "says", "sitting on", "standing on",
"using", "walking in", "walking on", "watching"]
id_predicate_labels_24=[]
for i in predicate_labels_24:
    id_predicate_labels_24.append(predicate_to_idx[i])
idx_to_label={"1": "airplane", "2": "animal", "3": "arm", "4": "bag", "5": "banana", "6": "basket", "7": "beach", "8": "bear", "9": "bed", "10": "bench", "11": "bike", "12": "bird", "13": "board", "14": "boat", "15": "book", "16": "boot", "17": "bottle", "18": "bowl", "19": "box", "20": "boy", "21": "branch", "22": "building", "23": "bus", "24": "cabinet", "25": "cap", "26": "car", "27": "cat", "28": "chair", "29": "child", "30": "clock", "31": "coat", "32": "counter", "33": "cow", "34": "cup", "35": "curtain", "36": "desk", "37": "dog", "38": "door", "39": "drawer", "40": "ear", "41": "elephant", "42": "engine", "43": "eye", "44": "face", "45": "fence", "46": "finger", "47": "flag", "48": "flower", "49": "food", "50": "fork", "51": "fruit", "52": "giraffe", "53": "girl", "54": "glass", "55": "glove", "56": "guy", "57": "hair", "58": "hand", "59": "handle", "60": "hat", "61": "head", "62": "helmet", "63": "hill", "64": "horse", "65": "house", "66": "jacket", "67": "jean", "68": "kid", "69": "kite", "70": "lady", "71": "lamp", "72": "laptop", "73": "leaf", "74": "leg", "75": "letter", "76": "light", "77": "logo", "78": "man", "79": "men", "80": "motorcycle", "81": "mountain", "82": "mouth", "83": "neck", "84": "nose", "85": "number", "86": "orange", "87": "pant", "88": "paper", "89": "paw", "90": "people", "91": "person", "92": "phone", "93": "pillow", "94": "pizza", "95": "plane", "96": "plant", "97": "plate", "98": "player", "99": "pole", "100": "post", "101": "pot", "102": "racket", "103": "railing", "104": "rock", "105": "roof", "106": "room", "107": "screen", "108": "seat", "109": "sheep", "110": "shelf", "111": "shirt", "112": "shoe", "113": "short", "114": "sidewalk", "115": "sign", "116": "sink", "117": "skateboard", "118": "ski", "119": "skier", "120": "sneaker", "121": "snow", "122": "sock", "123": "stand", "124": "street", "125": "surfboard", "126": "table", "127": "tail", "128": "tie", "129": "tile", "130": "tire", "131": "toilet", "132": "towel", "133": "tower", "134": "track", "135": "train", "136": "tree", "137": "truck", "138": "trunk", "139": "umbrella", "140": "vase", "141": "vegetable", "142": "vehicle", "143": "wave", "144": "wheel", "145": "window", "146": "windshield", "147": "wing", "148": "wire", "149": "woman", "150": "zebra"}
with open('/home/ths/zero-shot-clip/neural-motifs/train_samples/train_samples_to.pickle', 'rb') as file:
    train_samples_list=pickle.load(file)
with open('/home/ths/zero-shot-clip/neural-motifs/image_features/vg_to/crop_feature_vgtest_to.pickle', 'rb') as file:
    crop_feature_vgtest=pickle.load(file) 
class UPLDataset(Dataset):
    def __init__(self):
        self.train_samples = train_samples_list
    def __len__(self):
        return len(self.train_samples)
    def __getitem__(self, index):
        pred_label = self.train_samples[index][1]
        image_feature_crop=self.train_samples[index][9].reshape(1,512) #已经归一化 crop
        return pred_label, image_feature_crop
def my_collate(batch):
    batch = list(zip(*batch))
    res = {'pred_label': batch[0], 'image_feature_crop': batch[1]}
    del batch
    return res
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
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, x, tokenized_prompts):
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x
class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames) #类别数量
        n_ctx = 16
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype).cuda() #创建了一个形状为 (n_ctx, ctx_dim) 的未初始化的 PyTorch 张量 ctx_vectors，其中 n_ctx 是张量的行数，ctx_dim 是张量的列数，dtype 是张量的数据类型。
        nn.init.normal_(ctx_vectors, std=0.02)  #对张量 ctx_vectors 进行正态分布的初始化
        prompt_prefix = " ".join(["X"] * n_ctx) #创建了一个字符串 prompt_prefix，其中包含了由 "X" 重复 n_ctx 次形成
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized，requires_grad=True
        classnames = [name.replace("_", " ") for name in classnames]#将 classnames 列表中的每个字符串中的下划线（"_"）替换为空格（" "）
        name_lens = [len(_tokenizer.encode(name)) for name in classnames] #得到的 name_lens 列表包含了处理过的每个字符串的编码后的长度
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) 
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = 'end'
    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts
class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.clip = clip_model
    def forward(self, image_feature_crop):
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_feature_crop @ text_features.t()
        return logits
    def test(self, image_features_crop):
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features_crop = image_features_crop / image_features_crop.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        similarity = (logit_scale * image_features_crop @ text_features.T).softmax(dim=-1)
        return similarity
def cosine_schedule_warmup(total_step, value, final_value=0, warmup_step=0, warmup_value=0):
    if warmup_step > 0:
        warmup_schedule = np.linspace(warmup_value, value, warmup_step+2)[1:-1]
    else:
        warmup_schedule = np.array([])
    steps = np.arange(total_step - warmup_step)
    schedule = final_value + 0.5 * (value-final_value) * (1+np.cos(np.pi * steps / len(steps)))
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == total_step
    return schedule
class build_cosine_scheduler:
    def __init__(self, optimizer, lr, total_step, lr_warmup_step=0):
        init_lr = 0
        final_lr = lr * 1e-3
        self.lrs = cosine_schedule_warmup(total_step, lr, final_lr, lr_warmup_step, init_lr)
        self.optimizer = optimizer

    def step(self, idx):
        lr = self.lrs[idx] 
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = lr
        self.lr = lr

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
cudnn.deterministic = True
train_dataset = UPLDataset()
train_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True,
                                      num_workers=0, collate_fn=my_collate)
train, val, test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                          use_proposals=conf.use_proposals,
                          filter_non_overlap=False)
if conf.test:
    val = test
train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=1,
                                               num_workers=0,
                                               num_gpus=1)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = load_clip_to_cpu( )
clip_model.eval() #开启评估模式，梯度计算禁用
model = CustomCLIP(predicate_labels_24, clip_model)
clip_model.to(device)
model.to(device)
for name, param in model.named_parameters():
    if "prompt_learner" not in name:
        param.requires_grad_(False)
pytorch_total_params = sum(p.numel() for p in model.prompt_learner.parameters() if p.requires_grad)
print(f'Number of trainable params: {pytorch_total_params / 1000000}M.')
criterion = torch.nn.CrossEntropyLoss().cuda() #定义损失函数
per_epoch_steps = len(train_dataloader)
LR=1e-3
epochs=100
real_lr = LR  #学习率设置
param_dict = [{'params': [p for p in model.prompt_learner.parameters() if p.requires_grad]}]
optimizer = torch.optim.SGD(param_dict, lr=real_lr, weight_decay=0.0)
scheduler = build_cosine_scheduler(optimizer, lr=real_lr, total_step=epochs*per_epoch_steps)
if conf.ckpt == None:
    start_epoch = 0
else:
    print('load model')
    start_epoch = int(conf.ckpt.split("_")[-1][:-4]) + 1
    model.prompt_learner.load_state_dict(torch.load(conf.ckpt))
# main training process
for epoch in range(start_epoch, epochs):
    for idx, batch in enumerate(tqdm(train_dataloader)):
        image_feature_crop=[i.to(device) for i in batch['image_feature_crop']]
        image_feature_crop=torch.cat(image_feature_crop,dim=0) #[16,512]
        pred_label=batch['pred_label'] #类别索引0-23
        pred_label=torch.tensor(pred_label).to(device) #一维向量[16]
        cur_iter_idx = epoch*per_epoch_steps+idx
        scheduler.step(cur_iter_idx)
        output = model(image_feature_crop)
        loss = criterion(output, pred_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch %d : train_loss = %f' %(epoch, loss))
    torch.save(model.prompt_learner.state_dict(), '/home/ths/zero-shot-clip/neural-motifs/prompt-ckpt/vg_to/COOP/prompt_learner_params_%d.pth' % epoch)
    # evaluate on validation set
    model.eval()
    evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=False)
    evaluator_list = []
    for index,name in zip(id_predicate_labels_24,predicate_labels_24):
        evaluator_list.append((index, name, BasicSceneGraphEvaluator.all_modes()))
    for val_b, batch in tqdm(enumerate(val_loader)):
        obj_label = batch["gt_classes"][0] #object labels
        obj_pair=batch["gt_relations"][0]  #关系对
        impath = batch["fn"][0] 
        obj_boxes=batch['gt_boxes'][0]
        obj_pair_del=[]
        pred_rel_inds=[]
        for m in range(obj_pair.shape[0]):
            if idx_to_predicate[str(obj_pair[m][2])] not in predicate_labels_24: 
                obj_pair_del.append(m)  
        obj_pair_new = np.delete(obj_pair, obj_pair_del, axis=0)
        if obj_pair_new.shape[0]!=0:  
            sub_obj_img_features=crop_feature_vgtest[impath]   
            for i in range(obj_boxes.shape[0]):
                for j in range(obj_boxes.shape[0]):
                    pred_rel_inds.append([i,j])
            pred_rel_inds=np.array(pred_rel_inds) 
            similarity=model.test(sub_obj_img_features)
            rel_cores=similarity.to('cpu').detach().numpy()
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
                predicate_labels_24,
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
                    predicate_labels_24,
                    predicate_to_idx,
                )
    recall=evaluator[conf.mode].print_stats()
    mean_recall = calculate_mR_from_evaluator_list(evaluator_list, conf.mode)
    result={}
    result['recall']=recall
    result['mean_recall']=mean_recall
    with open('/home/ths/zero-shot-clip/neural-motifs/result/vg_to/COOP/single_prompt_epoch'+str(epoch)+'.pickle', 'wb') as file:
        pickle.dump(result, file)