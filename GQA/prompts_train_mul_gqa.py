from clip import clip
from clip.clip import  _transform,tokenize
from dataloaders.gqa import GQADataLoader, GQA
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
idx_to_predicate= {'1': 'on', '2': 'wearing', '3': 'of', '4': 'near', '5': 'in', '6': 'behind', '7': 'in front of', '8': 'holding', '9': 'next to', '10': 'above', '11': 'on top of', '12': 'below', '13': 'by', '14': 'with', '15': 'sitting on', '16': 'on the side of', '17': 'under', '18': 'riding', '19': 'standing on', '20': 'beside', '21': 'carrying', '22': 'walking on', '23': 'standing in', '24': 'lying on', '25': 'eating', '26': 'covered by', '27': 'looking at', '28': 'hanging on', '29': 'at', '30': 'covering', '31': 'on the front of', '32': 'around', '33': 'sitting in', '34': 'parked on', '35': 'watching', '36': 'flying in', '37': 'hanging from', '38': 'using', '39': 'sitting at', '40': 'covered in', '41': 'crossing', '42': 'standing next to', '43': 'playing with', '44': 'walking in', '45': 'on the back of', '46': 'reflected in', '47': 'flying', '48': 'touching', '49': 'surrounded by', '50': 'covered with', '51': 'standing by', '52': 'driving on', '53': 'leaning on', '54': 'lying in', '55': 'swinging', '56': 'full of', '57': 'talking on', '58': 'walking down', '59': 'throwing', '60': 'surrounding', '61': 'standing near', '62': 'standing behind', '63': 'hitting', '64': 'printed on', '65': 'filled with', '66': 'catching', '67': 'growing on', '68': 'grazing on', '69': 'mounted on', '70': 'facing', '71': 'leaning against', '72': 'cutting', '73': 'growing in', '74': 'floating in', '75': 'driving', '76': 'beneath', '77': 'contain', '78': 'resting on', '79': 'worn on', '80': 'walking with', '81': 'driving down', '82': 'on the bottom of', '83': 'playing on', '84': 'playing in', '85': 'feeding', '86': 'standing in front of', '87': 'waiting for', '88': 'running on', '89': 'close to', '90': 'sitting next to', '91': 'swimming in', '92': 'talking to', '93': 'grazing in', '94': 'pulling', '95': 'pulled by', '96': 'reaching for', '97': 'attached to', '98': 'skiing on', '99': 'parked along', '100': 'hang on'}
predicate_to_idx={}
for k,v in idx_to_predicate.items():
    predicate_to_idx[v]=int(k)
predicate_labels_26=['holding', 'sitting on', 'riding', 'standing on', 'carrying', 'walking on', 'standing in', 'lying on', 'eating', 'covered by', 'looking at', 'hanging on', 'covering', 'sitting in', 'parked on', 'watching', 'flying in', 'hanging from', 'using', 'sitting at', 'covered in', 'standing next to', 'playing with', 'walking in', 'reflected in', 'flying']
id_predicate_labels_26=[]
for i in predicate_labels_26:
    id_predicate_labels_26.append(predicate_to_idx[i])
idx_to_label={'1': 'window', '2': 'tree', '3': 'man', '4': 'shirt', '5': 'wall', '6': 'building', '7': 'person', '8': 'ground', '9': 'sky', '10': 'leg', '11': 'sign', '12': 'hand', '13': 'head', '14': 'pole', '15': 'grass', '16': 'hair', '17': 'car', '18': 'ear', '19': 'eye', '20': 'woman', '21': 'clouds', '22': 'shoe', '23': 'table', '24': 'leaves', '25': 'wheel', '26': 'door', '27': 'pants', '28': 'letter', '29': 'people', '30': 'flower', '31': 'water', '32': 'glass', '33': 'chair', '34': 'fence', '35': 'arm', '36': 'nose', '37': 'number', '38': 'floor', '39': 'rock', '40': 'jacket', '41': 'hat', '42': 'plate', '43': 'tail', '44': 'leaf', '45': 'face', '46': 'bush', '47': 'shorts', '48': 'road', '49': 'bag', '50': 'sidewalk', '51': 'tire', '52': 'helmet', '53': 'snow', '54': 'boy', '55': 'umbrella', '56': 'logo', '57': 'roof', '58': 'boat', '59': 'bottle', '60': 'street', '61': 'plant', '62': 'foot', '63': 'branch', '64': 'post', '65': 'jeans', '66': 'mouth', '67': 'cap', '68': 'girl', '69': 'bird', '70': 'banana', '71': 'box', '72': 'bench', '73': 'mirror', '74': 'picture', '75': 'pillow', '76': 'book', '77': 'field', '78': 'glove', '79': 'clock', '80': 'dirt', '81': 'bowl', '82': 'bus', '83': 'neck', '84': 'trunk', '85': 'wing', '86': 'horse', '87': 'food', '88': 'train', '89': 'kite', '90': 'paper', '91': 'shelf', '92': 'airplane', '93': 'sock', '94': 'house', '95': 'elephant', '96': 'lamp', '97': 'coat', '98': 'cup', '99': 'cabinet', '100': 'street light', '101': 'cow', '102': 'word', '103': 'dog', '104': 'finger', '105': 'giraffe', '106': 'mountain', '107': 'wire', '108': 'flag', '109': 'seat', '110': 'sheep', '111': 'counter', '112': 'skis', '113': 'zebra', '114': 'hill', '115': 'truck', '116': 'bike', '117': 'racket', '118': 'ball', '119': 'skateboard', '120': 'ceiling', '121': 'motorcycle', '122': 'player', '123': 'surfboard', '124': 'sand', '125': 'towel', '126': 'frame', '127': 'container', '128': 'paw', '129': 'feet', '130': 'curtain', '131': 'windshield', '132': 'traffic light', '133': 'horn', '134': 'cat', '135': 'child', '136': 'bed', '137': 'sink', '138': 'animal', '139': 'donut', '140': 'stone', '141': 'tie', '142': 'pizza', '143': 'orange', '144': 'sticker', '145': 'apple', '146': 'backpack', '147': 'vase', '148': 'basket', '149': 'drawer', '150': 'collar', '151': 'lid', '152': 'cord', '153': 'phone', '154': 'pot', '155': 'vehicle', '156': 'fruit', '157': 'laptop', '158': 'fork', '159': 'uniform', '160': 'bear', '161': 'fur', '162': 'license plate', '163': 'lady', '164': 'tomato', '165': 'tag', '166': 'mane', '167': 'beach', '168': 'tower', '169': 'cone', '170': 'cheese', '171': 'wrist', '172': 'napkin', '173': 'toilet', '174': 'desk', '175': 'dress', '176': 'cell phone', '177': 'faucet', '178': 'blanket', '179': 'screen', '180': 'watch', '181': 'keyboard', '182': 'arrow', '183': 'sneakers', '184': 'broccoli', '185': 'bicycle', '186': 'guy', '187': 'knife', '188': 'ocean', '189': 't-shirt', '190': 'bread', '191': 'spots', '192': 'cake', '193': 'air', '194': 'sweater', '195': 'room', '196': 'couch', '197': 'camera', '198': 'frisbee', '199': 'trash can', '200': 'paint'}
with open('/home/ths/zero-shot-clip/neural-motifs/train_samples/gqa/train_samples_gqa.pickle', 'rb') as file:
    train_samples_list=pickle.load(file)
with open('/home/ths/zero-shot-clip/neural-motifs/image_features/GQA/sam_crop_feature_gqatest.pickle', 'rb') as file:
    crop_sam_feature_gqatest=pickle.load(file) 
with open('/home/ths/zero-shot-clip/neural-motifs/image_features/GQA/crop_feature_gqatest.pickle', 'rb') as file:
    crop_feature_gqatest=pickle.load(file) 
class UPLDataset(Dataset):
    def __init__(self):
        self.train_samples = train_samples_list
    def __len__(self):
        return len(self.train_samples)
    def __getitem__(self, index):
        pred_label = self.train_samples[index][1]
        image_feature = self.train_samples[index][2].reshape(1,512)
        image_feature_crop=self.train_samples[index][3].reshape(1,512) #已经归一化 
        return pred_label, image_feature, image_feature_crop
def my_collate(batch):
    batch = list(zip(*batch))
    res = {'pred_label': batch[0], 'image_feature': batch[1], 'image_feature_crop': batch[2]}
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
    def __init__(self, predicate_labels, clip_model, n_ctx=16, n_prompt=32, prompt_bsz=4):
        super().__init__()
        ctx_dim = clip_model.ln_final.weight.shape[0] #上下文向量的维度
        dtype = clip_model.dtype #模型数据类型
        n_cls = len(predicate_labels) #谓词类别数量
        self.dtype = dtype
        ctx_vectors = torch.empty(n_prompt, n_ctx, ctx_dim, dtype=self.dtype).cuda()  #创建了32个 (16, ctx_dim) 的未初始化的 PyTorch 张量 ctx_vectors
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        assert n_prompt % prompt_bsz == 0
        self.n_iter = int(n_prompt/prompt_bsz)
        prompt_prefix = ' '.join(['X'] * n_ctx)
        prompts = [prompt_prefix + ' ' +pred+'.' for pred in predicate_labels]
        predicate_labels = [name.replace("_", " ") for name in predicate_labels]
        self.name_lens = [len(_tokenizer.encode(pred)) for pred in predicate_labels]
        if n_prompt >1:
            self.pos = [0 for _ in range(n_prompt//4)] + [1 for _ in range(n_prompt//4)] + [2 for _ in range(n_prompt//2)]
        else:
            self.pos = [2 for _ in range(n_prompt)]
        self.pos = torch.tensor(self.pos, device='cuda')
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts])
        self.tokenized_prompts = tokenized_prompts
        with torch.no_grad(): 
            embedding = clip_model.token_embedding(tokenized_prompts.cuda()).type(self.dtype)
        self.register_buffer('token_prefix', embedding[:, :1, :]) # SOS, [n_cls, 1, ctx_dim]
        self.register_buffer('token_suffix', embedding[:, 1+n_ctx:, :]) # CLS, EOS, [n_cls, -1, ctx_dim] 
        nc_prompts = [prompt_prefix + '.' ]
        nc_tokenized_prompts = torch.cat([tokenize(p) for p in nc_prompts])
        self.nc_tokenized_prompts = nc_tokenized_prompts
        with torch.no_grad():
            embedding = clip_model.token_embedding(nc_tokenized_prompts.cuda()).type(self.dtype)
        self.register_buffer('nc_token_prefix', embedding[:, :1, :]) # SOS, [n_cls, 1, ctx_dim] 
        self.register_buffer('nc_token_suffix', embedding[:, 1+n_ctx:, :]) # EOS, [n_cls, -1, ctx_dim]
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.n_prompt = n_prompt
        self.ctx_dim = ctx_dim
        self.prompt_bsz = prompt_bsz
        self.prompt_build_mode = 'end'
        self.iter_idx = 0

    def forward(self, test):
        if self.n_iter > 1 and (not test):
            if self.iter_idx == 0:
                self.select_idx = torch.randperm(self.n_prompt, device='cuda') #提示索引0-31随机打乱
            batch_idx = self.select_idx[self.iter_idx*self.prompt_bsz: (self.iter_idx+1)*self.prompt_bsz] #选择当前迭代要使用的提示索引，通过切片操作实现
            ctx = self.ctx[batch_idx]
            pos = self.pos[batch_idx]

            self.iter_idx += 1
            if self.iter_idx == self.n_iter:
                self.iter_idx = 0
        else:
            ctx = self.ctx
            pos = self.pos

        prompt_size = ctx.shape[0]
        tokenized_prompts = self.tokenized_prompts.unsqueeze(1).repeat(1, prompt_size, 1).view(self.n_cls*prompt_size, -1)

        n_cls = self.n_cls

        ctx_end = ctx[pos==2]
        n_end = ctx_end.shape[0]
        prefix = self.token_prefix.unsqueeze(1).repeat(1, n_end, 1, 1)
        suffix = self.token_suffix.unsqueeze(1).repeat(1, n_end, 1, 1)
        ctx_end = ctx_end.unsqueeze(0).repeat(n_cls, 1, 1, 1)
        prompts_end = torch.cat([prefix, ctx_end, suffix], dim=2)

        ctx_middle = ctx[pos==1]
        n_middle = ctx_middle.shape[0]
        prompts_middle = []
        half_n_ctx = self.n_ctx // 2
        for i in range(n_cls):
            name_len = self.name_lens[i]
            prefix_i = self.token_prefix[i:i+1, :, :].unsqueeze(1).repeat(1, n_middle, 1, 1)
            class_i = self.token_suffix[i:i+1, :name_len, :].unsqueeze(1).repeat(1, n_middle, 1, 1)
            suffix_i = self.token_suffix[i:i+1, name_len:, :].unsqueeze(1).repeat(1, n_middle, 1, 1)
            ctx_i_half1 = ctx_middle[:, :half_n_ctx, :].unsqueeze(0)
            ctx_i_half2 = ctx_middle[:, half_n_ctx:, :].unsqueeze(0)
            prompt = torch.cat([
                prefix_i, # (1, n_middle, 1, dim)
                ctx_i_half1, # (1, n_middle, n_ctx//2, dim)
                class_i, # (1, n_middle, name_len, dim)
                ctx_i_half2, # (1, n_middle, n_ctx//2, dim)
                suffix_i # (1, n_middle, *, dim)
            ], dim=2)
            prompts_middle.append(prompt)
        prompts_middle = torch.cat(prompts_middle, dim=0)

        ctx_front = ctx[pos==0]
        n_front = ctx_front.shape[0]
        prompts_front = []
        for i in range(self.n_cls):
            name_len = self.name_lens[i]
            prefix_i = self.token_prefix[i:i+1, :, :].unsqueeze(1).repeat(1, n_front, 1, 1)
            class_i = self.token_suffix[i:i+1, :name_len, :].unsqueeze(1).repeat(1, n_front, 1, 1)
            suffix_i = self.token_suffix[i:i+1, name_len:, :].unsqueeze(1).repeat(1, n_front, 1, 1)
            ctx_i = ctx_front.unsqueeze(0)
            prompt = torch.cat([
                prefix_i, # (1, n_front, 1, dim)
                class_i, # (1, n_front, name_len, dim)
                ctx_i, # (1, n_front, n_ctx, dim)
                suffix_i # (1, n_front, *, dim)
            ], dim=2)
            prompts_front.append(prompt)
        prompts_front = torch.cat(prompts_front, dim=0)

        prompts = torch.cat([prompts_end,prompts_middle, prompts_front], dim=1).view(prompt_size*n_cls, -1, self.ctx_dim)
        
        if test:
            return prompts, tokenized_prompts
        else:
            nc_prompts, nc_tokenized_prompts = self.only_prefix( )
            return prompts, tokenized_prompts, nc_prompts, nc_tokenized_prompts

    def only_prefix(self):
        ctx = self.ctx
        prompt_size = ctx.shape[0]
        nc_tokenized_prompts = self.nc_tokenized_prompts.repeat(prompt_size, 1)
        prefix = self.nc_token_prefix.repeat(prompt_size, 1, 1)
        suffix = self.nc_token_suffix.repeat(prompt_size, 1, 1)
        nc_prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return nc_prompts, nc_tokenized_prompts

class CustomCLIP(nn.Module):
    def __init__(self, predicate_labels, clip_model, n_ctx=16, n_prompt=32, prompt_bsz=4):
        super().__init__()
        self.n_predclass=len(predicate_labels)
        self.n_prompt = n_prompt
        # text enoder
        self.text_encoder = TextEncoder(clip_model)
        # prompt learner
        self.prompt_learner = PromptLearner(predicate_labels, clip_model, n_ctx=n_ctx, n_prompt=n_prompt, prompt_bsz=prompt_bsz)
        # image encoder
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image_feature, image_feature_crop, pred_label):
        n_predclass = self.n_predclass
        text_prompt, tokenized_prompts, nc_prompts, nc_tokenized_prompts  = self.prompt_learner(test=False)
        n_prompt = text_prompt.shape[0]//n_predclass
        text_features = self.text_encoder(text_prompt, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.view(n_predclass, n_prompt, -1)
        text_mean = text_features.mean(dim=1)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_feature @ text_mean.t()
        logits1 = logit_scale * image_feature_crop @ text_mean.t()
        logits=0.6*logits+0.4*logits1

        
        batch_size = pred_label.shape[0] 
                
        text_features = text_features - text_mean.unsqueeze(1)
        diag_cov_martix = text_features.permute(2,0,1) @ text_features.permute(2,1,0)
        diag_cov_martix /= n_prompt + 1
        refined_logits = torch.einsum("bd, dik -> bik", [image_feature**2, diag_cov_martix])
        refined_logits1 = torch.einsum("bd, dik -> bik", [image_feature_crop**2, diag_cov_martix])
        refined_logits=0.6*refined_logits+0.4*refined_logits1

        sigma = refined_logits[torch.arange(batch_size), pred_label, pred_label].unsqueeze(-1) + \
                    refined_logits[:, torch.arange(n_predclass), torch.arange(n_predclass) ] - \
                    2 * refined_logits[torch.arange(batch_size), pred_label, : ]

        logits += 0.5*(logit_scale**2)*sigma.view(-1, n_predclass)

        loss_m = None
        nc_text_features = self.text_encoder(nc_prompts, nc_tokenized_prompts)
        nc_text_features = nc_text_features / nc_text_features.norm(dim=-1, keepdim=True)
        dis = nc_text_features @ nc_text_features.permute(1, 0)
        loss_m = dis[~torch.eye(self.n_prompt, dtype=torch.bool, device='cuda')].abs().mean()

        return logits, loss_m
    def test(self, image_features,add_img_features):
        with torch.no_grad():
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            add_img_features=add_img_features/add_img_features.norm(dim=-1, keepdim=True)
        text_prompt, tokenized_prompts = self.prompt_learner(test=True)
        text_features = self.text_encoder(text_prompt, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.view(self.n_predclass, self.n_prompt, -1)
        text_features = text_features.mean(dim=1)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        logits1 = logit_scale * add_img_features @ text_features.t()
        logits=0.6*logits+0.4*logits1
        return logits

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
train, val, test = GQA.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                          use_proposals=conf.use_proposals,
                          filter_non_overlap=False)
if conf.test:
    val = test
train_loader, val_loader = GQADataLoader.splits(train, val, mode='rel',
                                               batch_size=1,
                                               num_workers=0,
                                               num_gpus=1)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = load_clip_to_cpu( )
clip_model.to(device)
clip_model.eval() #开启评估模式，梯度计算禁用
N_CTX=16
N_PROMPT=32
PROMPT_BSZ=4
model = CustomCLIP(predicate_labels_26, 
                   clip_model, 
                   N_CTX, #16，一个提示里有16个向量
                   N_PROMPT, #32 一共有32个不同的提示
                   PROMPT_BSZ, #4 每个批次采样提示数量
                   )
model.to(device)
for name, param in model.named_parameters():
    if "prompt_learner" not in name:
        param.requires_grad_(False)
pytorch_total_params = sum(p.numel() for p in model.prompt_learner.parameters() if p.requires_grad)
print(f'Number of trainable params: {pytorch_total_params / 1000000}M.')
criterion = torch.nn.CrossEntropyLoss().cuda() #定义损失函数
iter_per_batch = 8 #每批次迭代8次，为了在一个批次遍历所有prompts
per_epoch_steps = len(train_dataloader) * iter_per_batch
LR=1e-3
epochs=100
real_lr = LR #* PROMPT_BSZ *img_batch_size /20 #学习率设置
param_dict = [{'params': [p for p in model.prompt_learner.parameters() if p.requires_grad]}]
optimizer = torch.optim.SGD(param_dict, lr=real_lr, weight_decay=0.0)
scheduler = build_cosine_scheduler(optimizer, lr=real_lr, total_step=epochs*per_epoch_steps)
sam_checkpoint = "/home/ths/zero-shot-clip/sam_vit_h_4b8939.pth"
model_type = "default"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
if conf.ckpt == None:
    start_epoch = 0
else:
    print('load model')
    start_epoch = int(conf.ckpt.split("_")[-1][:-4]) + 1
    model.prompt_learner.load_state_dict(torch.load(conf.ckpt))
# main training process
for epoch in range(start_epoch, epochs):
    for idx, batch in enumerate(tqdm(train_dataloader)):
        image_feature=[i.to(device) for i in batch['image_feature']]
        image_feature=torch.cat(image_feature,dim=0) #[16,512]
        image_feature_crop=[i.to(device) for i in batch['image_feature_crop']]
        image_feature_crop=torch.cat(image_feature_crop,dim=0) #[16,512]
        pred_label=batch['pred_label'] #类别索引0-23
        pred_label=torch.tensor(pred_label).to(device) #一维向量[16]
        for iter_idx in range(iter_per_batch):
            cur_iter_idx = epoch*per_epoch_steps+idx*iter_per_batch+iter_idx
            scheduler.step(cur_iter_idx)
            output, loss_m = model(image_feature, image_feature_crop, pred_label)
            loss = criterion(output, pred_label)
            loss += 0.1*loss_m
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print('epoch %d : train_loss = %f' %(epoch, loss))
    torch.save(model.prompt_learner.state_dict(), '/home/ths/zero-shot-clip/neural-motifs/prompt-ckpt/gqa/mul_prompt_learner_params_%d.pth' % epoch)

    model.eval()
    evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=False)
    evaluator_list = []
    for index,name in zip(id_predicate_labels_26,predicate_labels_26):
        evaluator_list.append((index, name, BasicSceneGraphEvaluator.all_modes()))
    for val_b, batch in tqdm(enumerate(val_loader)):
        obj_label = batch["gt_classes"][0] #object labels
        obj_pair=batch["gt_relations"][0]  #关系对
        impath = batch["fn"][0]
        obj_boxes=batch['gt_boxes'][0]
        obj_pair_del=[]
        pred_rel_inds=[]
        for m in range(obj_pair.shape[0]):
            if idx_to_predicate[str(obj_pair[m][2])] not in predicate_labels_26: 
                obj_pair_del.append(m)  
        obj_pair_new = np.delete(obj_pair, obj_pair_del, axis=0)
        if obj_pair_new.shape[0]!=0:  
            sub_obj_img_features=crop_sam_feature_gqatest[impath][0]
            pred_rel_inds=crop_sam_feature_gqatest[impath][1]
            add_img_features=crop_feature_gqatest[impath][0]
            similarity=model.test(sub_obj_img_features, add_img_features)
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
    with open('/home/ths/zero-shot-clip/neural-motifs/result/GQA/mul_prompt/mul_prompt_epoch'+str(epoch)+'.pickle', 'wb') as file:
        pickle.dump(result, file)
    




