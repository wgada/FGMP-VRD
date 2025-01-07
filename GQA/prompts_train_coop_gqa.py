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
with open('/home/ths/zero-shot-clip/neural-motifs/image_features/GQA/crop_feature_gqatest.pickle', 'rb') as file:
    crop_feature_gqatest=pickle.load(file) 
class UPLDataset(Dataset):
    def __init__(self):
        self.train_samples = train_samples_list
    def __len__(self):
        return len(self.train_samples)
    def __getitem__(self, index):
        pred_label = self.train_samples[index][1]
        image_feature_crop=self.train_samples[index][3].reshape(1,512) #已经归一化 
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
clip_model.eval() #开启评估模式，梯度计算禁用
model = CustomCLIP(predicate_labels_26, clip_model)
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
    torch.save(model.prompt_learner.state_dict(), '/home/ths/zero-shot-clip/neural-motifs/prompt-ckpt/gqa/COOP/prompt_learner_params_%d.pth' % epoch)
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
            sub_obj_img_features=crop_feature_gqatest[impath][0]
            pred_rel_inds=crop_feature_gqatest[impath][1]
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
    with open('/home/ths/zero-shot-clip/neural-motifs/result/GQA/COOP/mul_prompt_epoch'+str(epoch)+'.pickle', 'wb') as file:
        pickle.dump(result, file)