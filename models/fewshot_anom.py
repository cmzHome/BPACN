import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .backbone.torchvision_backbones import TVDeeplabRes101Encoder
from . import layers as L

class DownSampleBlock(nn.Module):
    def __init__(self, f_dim, k_size):
        super().__init__()
        self.protoConv = L.Conv2d(f_dim, f_dim, kernel_size=k_size, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, qry_ft, bg_proto, fg_proto):   # [1,256,32,32] / [1,256] / [1,256]
        qry_ft = self.relu(self.protoConv(qry_ft, bg_proto, fg_proto))    # [1,256,out_size,out_size]

        return qry_ft  # [1,256,out_size,out_size]

class FewShotSeg(nn.Module):

    def __init__(self, use_coco_init=True):
        super().__init__()

        # Encoder
        self.encoder = TVDeeplabRes101Encoder(use_coco_init)
        self.t_1 = Parameter(torch.Tensor([-10.0]))
        self.t_2 = Parameter(torch.Tensor([-10.0]))
        self.t_3 = Parameter(torch.Tensor([-10.0]))

        self.downSamleBlock_1 = DownSampleBlock(256, 9)
        self.downSamleBlock_2 = DownSampleBlock(256, 9)

        self.map_threshold_1 = nn.Conv2d(4, 1, kernel_size=1, bias=False)
        self.map_threshold_1.weight = nn.Parameter(torch.tensor([[1.0],[0.0],[0.0],[0.0]]).reshape_as(self.map_threshold_1.weight))
        self.map_threshold_2 = nn.Conv2d(4, 1, kernel_size=1, bias=False)
        self.map_threshold_2.weight = nn.Parameter(torch.tensor([[1.0],[0.0],[0.0],[0.0]]).reshape_as(self.map_threshold_2.weight))
        self.map_threshold_3 = nn.Conv2d(4, 1, kernel_size=1, bias=False)
        self.map_threshold_3.weight = nn.Parameter(torch.tensor([[1.0],[0.0],[0.0],[0.0]]).reshape_as(self.map_threshold_3.weight))
        self.all_map_thre = [self.map_threshold_1, self.map_threshold_2,  self.map_threshold_3]
        self.scaler = 20.0
        
        self.cat = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        self.cat.weight = nn.Parameter(torch.tensor([[0.333],[0.333],[0.333]]).reshape_as(self.cat.weight))

        self.criterion = nn.NLLLoss()

    # supp_imgs: n-shot*[1,3,256,256]    fore_mask: n-shot*[1,256,256]   qry_imgs: n_batch*[1,3,256,256]   
    def forward(self, supp_imgs, fore_mask, qry_imgs, train=False, t_loss_scaler=1.0):

        n_ways = len(supp_imgs)                # 1
        self.n_shots = len(supp_imgs[0])       # 1
        n_queries = len(qry_imgs)              # 1
        batch_size_q = qry_imgs[0].shape[0]    # 1
        batch_size = supp_imgs[0][0].shape[0]  # 1
        img_size = supp_imgs[0][0].shape[-2:]  # [256,256]

        # ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)   # 就是把supp和query拼了起来 [2,3,256,256]
        img_fts = self.encoder(imgs_concat, low_level=False)  # [2,256,32,32]

        fts_size = img_fts.shape[-2:]  # [256,32,32]

        supp_fts = img_fts[:n_ways * self.n_shots * batch_size].view(
            n_ways, self.n_shots, batch_size, -1, *fts_size)  # [1,1,1,256,32,32]
        qry_fts = img_fts[n_ways * self.n_shots * batch_size:].view(
            n_queries, batch_size_q, -1, *fts_size)  # [1,1,256,32,32]

        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # [1,1,1,256,256]
        
        #region 可视化 support foreground features (训练记得注释掉)
        # from sklearn.manifold import TSNE
        # import matplotlib.pyplot as plt
        # draw_supp_fts = supp_fts.view(256, 32*32)  # [256, 1024]
        # mean_supp_mask = fore_mask.view(-1, 1, fore_mask.shape[-1], fore_mask.shape[-1]).mean(0, True)
        # draw_fore_mask = F.interpolate(mean_supp_mask, size=[32,32], mode='bilinear', align_corners=True).view(32*32)  #[1024]
        # larger_zero = torch.nonzero(draw_fore_mask > 0.0).squeeze()
        # draw_fg_fts = draw_supp_fts[:, larger_zero]  #[256, n]
        # draw_fg_fts = draw_fg_fts.transpose(1, 0)  

        # draw_fg_fts = draw_fg_fts.cpu().numpy()
        # draw_embeddings = TSNE(n_components=2).fit_transform(draw_fg_fts)  #[n, 2]
        
        # plt.scatter(draw_embeddings[:, 0], draw_embeddings[:, 1], color='blue')
        # plt.savefig("/home/cmz/experiments/ADNet-main/visualization/points.jpg")
        #endregion 

        ###### Compute loss ######
        align_loss = torch.zeros(1).cuda()   # [0] 初始化
        outputs = []
        for epi in range(batch_size):

            ###### Extract prototypes ######
            supp_fts_ = [[self.getFeatures(supp_fts[way, shot, [epi]],   # [256,32,32]
                                           fore_mask[way, shot, [epi]])  # [256,256]
                          for shot in range(self.n_shots)] for way in range(n_ways)]    # => n_way*[2,1,256]

            fg_sup_fts = [ft[0][0].unsqueeze(0) for ft in supp_fts_]    # n_way*[1,1,256]
            bg_sup_fts = [ft[0][1].unsqueeze(0) for ft in supp_fts_]    # n_way*[1,1,256]

            fg_prototypes = self.getPrototype(fg_sup_fts)  # n_way*[1,256]
            bg_prototypes = self.getPrototype(bg_sup_fts)  # n_way*[1,256]

            mean_supp_mask = fore_mask.view(-1, 1, fore_mask.shape[-1], fore_mask.shape[-1]).mean(0, True)        # [1,1,256,256]
            mean_supp_mask_1 = F.interpolate(mean_supp_mask, size=[32,32], mode='bilinear', align_corners=True)   # [1,1,32,32]
            mean_supp_mask_2 = F.interpolate(mean_supp_mask, size=[24,24], mode='bilinear', align_corners=True)   # [1,1,24,24]
            mean_supp_mask_3 = F.interpolate(mean_supp_mask, size=[16,16], mode='bilinear', align_corners=True)   # [1,1,16,16]

            ###### Adjust fg-proto ######
            fg_prototypes = [self.adjustProto(supp_fts[0][0], mean_supp_mask_1, fg_prototypes[w], bg_prototypes[w])  for w in range(n_ways)]

            for q_id in range(batch_size_q):
                query_feat_1 = qry_fts[epi, q_id].unsqueeze(0)   # [1,256,32,32]

                ###### Compute anom. scores ######
                anom_s_1 = [self.negSim(query_feat_1, prototype) for prototype in fg_prototypes]        # n_way*[1,32,32]
                prior_sim_1 = [self.calcu_prior_sim(query_feat_1, fg_prototypes[w], bg_prototypes[w]) for w in range(n_ways)]  # n_way*[1,2,32,32]
                
                query_feat_2 = self.downSamleBlock_1(query_feat_1, bg_prototypes[0], fg_prototypes[0])  # [1,256,24,24]
                anom_s_2 = [self.negSim(query_feat_2, prototype) for prototype in fg_prototypes]        # n_way*[1,24,24]
                prior_sim_2 = [self.calcu_prior_sim(query_feat_2, fg_prototypes[w], bg_prototypes[w]) for w in range(n_ways)]  # n_way*[1,2,24,24]
                
                query_feat_3 = self.downSamleBlock_2(query_feat_2, bg_prototypes[0], fg_prototypes[0])  # [1,256,16,16]
                anom_s_3 = [self.negSim(query_feat_3, prototype) for prototype in fg_prototypes]        # n_way*[1,16,16]
                prior_sim_3 = [self.calcu_prior_sim(query_feat_3, fg_prototypes[w], bg_prototypes[w]) for w in range(n_ways)]  # n_way*[1,2,16,16]

                ###### Get threshold #######
                self.thresh_pred_1 = [self.t_1 for _ in range(n_ways)]
                self.thresh_pred_2 = [self.t_2 for _ in range(n_ways)]
                self.thresh_pred_3 = [self.t_3 for _ in range(n_ways)]
                self.t_loss_1 = self.t_1 / self.scaler
                self.t_loss_2 = self.t_2 / self.scaler
                self.t_loss_3 = self.t_3 / self.scaler

                ###### Get predictions #######
                pred_1 = self.getPred(anom_s_1, self.thresh_pred_1, prior_sim_1[epi], mean_supp_mask_1, 0)  # [1,1,32,32]
                pred_2 = self.getPred(anom_s_2, self.thresh_pred_2, prior_sim_2[epi], mean_supp_mask_2, 1)  # [1,1,24,24]
                pred_2 = F.interpolate(pred_2, size=(32,32), mode='bilinear', align_corners=True)  # [1,1,32,32]
                pred_3 = self.getPred(anom_s_3, self.thresh_pred_3, prior_sim_3[epi], mean_supp_mask_3, 2)  # [1,1,16,16]
                pred_3 = F.interpolate(pred_3, size=(32,32), mode='bilinear', align_corners=True)  # [1,1,32,32]

                pred = self.cat(torch.cat((pred_1, pred_2, pred_3), dim=1))  # [1,1,32,32]
                pred_ups = F.interpolate(pred, size=img_size, mode='bilinear', align_corners=True)  # [1,1,256,256]
                pred_ups = torch.cat((1.0 - pred_ups, pred_ups), dim=1)      # [1,2,256,256]

                outputs.append(pred_ups)

                ###### Prototype alignment loss ######
                if train:
                    align_loss_epi = self.alignLoss(query_feat_1,          # [1,256,32,32]
                                                    torch.cat((1.0 - pred, pred), dim=1),  # [1,2,32,32]
                                                    supp_fts[:, :, epi],   # [1,1,256,32,32]
                                                    fore_mask[:, :, epi])  # [1,1,256,256]
                    align_loss += align_loss_epi

        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])
        return output, (align_loss / batch_size), (t_loss_scaler * (self.t_loss_1 + self.t_loss_2 + self.t_loss_3))

    def negSim(self, fts, prototype):

        sim = - F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler

        return sim

    def getFeatures(self, fts, mask):  # fts: [1,256,32,32]  mask: [1,256,256]
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')   # [1,256,32,32] => [1,256,256,256]

        no_mask = (1.0 - mask)

        # masked fg features
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) \
                     / (mask[None, ...].sum(dim=(2, 3)) + 1e-5)  # [1,256]

        no_mask_fts = torch.sum(fts * no_mask[None, ...], dim=(2, 3)) \
                     / (mask[None, ...].sum(dim=(2, 3)) + 1e-5)  # [1,256]

        return torch.stack([masked_fts, no_mask_fts])  # [2,256]

    def getPrototype(self, fg_fts):

        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
                         fg_fts]  ## concat all fg_fts

        return fg_prototypes

    def adjustProto(self, fts, mean_mask_1, fg_proto, bg_proto):
        temp_mask = mean_mask_1.flatten()   # [32*32]
        temp_fts = fts.view(256, -1)        # [256,32*32]
        fg_fts = (temp_fts[:, (temp_mask > 0.0)]).transpose(1,0)                   # [n,256]

        if fg_fts.shape[0] > 0:
            fg_sim = F.cosine_similarity(fg_fts, fg_proto, dim=1) * self.scaler    # [n]
            bg_sim = F.cosine_similarity(fg_fts, bg_proto, dim=1) * self.scaler    # [n]
            minus_sim = fg_sim - bg_sim                                            # [n]
            selected_indice = (minus_sim > (minus_sim.mean() + minus_sim.min()) / 2).float().unsqueeze(0)  # [1,n]
            # selected_indice = (minus_sim > minus_sim.mean()).float().unsqueeze(0)  # [1,n]
            # selected_indice = (minus_sim > (minus_sim.max() * 2 / 3)).float().unsqueeze(0)  # [1,n]
            selected_num = selected_indice.sum()

            if selected_num > 0:
                fg_proto = (selected_indice @ fg_fts) / (selected_num)        # [1,256]
            
        return fg_proto

    # qry_fts: [1,256,32,32]    pred: [1,2,32,32]    supp_fts: [1,1,256,32,32]    fore_mask: [1,1,256,256]
    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Mask and get query prototype
        pred_mask = pred.argmax(dim=1, keepdim=True)  # [1,1,32,32]  相当于 预测结果
        mean_qry_mask_1 = pred_mask.float() # [1,1,32,32]
        mean_qry_mask_2 = F.interpolate(mean_qry_mask_1, size=[24,24], mode='bilinear', align_corners=True)     # [1,1,24,24]
        mean_qry_mask_3 = F.interpolate(mean_qry_mask_1, size=[16,16], mode='bilinear', align_corners=True)     # [1,1,16,16]
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]   # 2*[1,1,32,32]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]  # []
        pred_mask = torch.stack(binary_masks, dim=1).float()  # [1,2,1,32,32]
        qry_prototypes = torch.sum(qry_fts.unsqueeze(1) * pred_mask, dim=(0, 3, 4))
        qry_prototypes = qry_prototypes / (pred_mask.sum((0, 3, 4)) + 1e-5)  # [2,256]

        # Compute the support loss
        loss = torch.zeros(1).cuda()
        for way in range(n_ways):
            if way in skip_ways:
                continue

            qry_prototypes[[way + 1]] = self.adjustProto(qry_fts, pred_mask[0,[1]], qry_prototypes[[way + 1]], qry_prototypes[[way]])

            # Get the query prototypes
            for shot in range(n_shots):
                img_fts_1 = supp_fts[way, [shot]]   # [1,256,32,32]
                supp_sim_1 = self.negSim(img_fts_1, qry_prototypes[[way + 1]])  # [1,32,32]
                prior_sim_1 = self.calcu_prior_sim(img_fts_1, qry_prototypes[[way + 1]], qry_prototypes[[way]])   # [1,1,32,32]

                img_fts_2 = self.downSamleBlock_1(img_fts_1, qry_prototypes[[way]], qry_prototypes[[way + 1]])   # [1,256,24,24]
                supp_sim_2 = self.negSim(img_fts_2, qry_prototypes[[way + 1]])  # [1,24,24]
                prior_sim_2 = self.calcu_prior_sim(img_fts_2, qry_prototypes[[way + 1]], qry_prototypes[[way]])   # [1,1,24,24]

                img_fts_3 = self.downSamleBlock_2(img_fts_2, qry_prototypes[[way]], qry_prototypes[[way + 1]])   # [1,256,16,16]
                supp_sim_3 = self.negSim(img_fts_3, qry_prototypes[[way + 1]])  # [1,15,15]
                prior_sim_3 = self.calcu_prior_sim(img_fts_3, qry_prototypes[[way + 1]], qry_prototypes[[way]])   # [1,1,16,16]

                pred_1 = self.getPred([supp_sim_1], [self.thresh_pred_1[way]], prior_sim_1, mean_qry_mask_1, 0)  # [1,1,32,32]
                pred_2 = self.getPred([supp_sim_2], [self.thresh_pred_2[way]], prior_sim_2, mean_qry_mask_2, 1)  # [1,1,24,24]
                pred_2 = F.interpolate(pred_2, size=(32,32), mode='bilinear', align_corners=True)  # [1,1,32,32]
                pred_3 = self.getPred([supp_sim_3], [self.thresh_pred_3[way]], prior_sim_3, mean_qry_mask_3, 2)  # [1,1,16,16]
                pred_3 = F.interpolate(pred_3, size=(32,32), mode='bilinear', align_corners=True)  # [1,1,32,32]
                pred = self.cat(torch.cat((pred_1, pred_2, pred_3), dim=1))     # [1,1,32,32]

                pred_ups = F.interpolate(pred, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
                pred_ups = torch.cat((1.0 - pred_ups, pred_ups), dim=1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255).cuda()
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss

    def getPred(self, sim, thresh, prior_sim, mean_supp_mask, idx):  # sim: n_way*[1,32,32]  thresh: n_way*[threshold]  prior_sim: [1,1,32,32]
        pred = []

        for s, t in zip(sim, thresh):
            size = prior_sim.shape[-1]   # 32
            th_map = t.view(1,1,1,1).repeat(1, 1, size, size)   # [1,1,32,32]
            th_map = self.all_map_thre[idx](torch.cat((th_map, prior_sim, mean_supp_mask), dim=1)).view(1, size, size)  # [1,32,32]

            #region 绘制热力图 (训练记得注释掉)
            # if idx == 0:
            #     import seaborn as sns
            #     import matplotlib.pyplot as plt
            #     #画图
            #     x = th_map.squeeze(0).cpu().numpy()
            #     sns.heatmap(x, annot=False, cmap= "GnBu", linewidths= 0.5, yticklabels=False, xticklabels=False)
            #     plt.savefig("/home/cmz/experiments/ADNet-main/visualization/heatmap.jpg")
            #     plt.clf()
            #endregion 

            pred.append(1.0 - torch.sigmoid(0.5 * (s - th_map)))   # [1,32,32] - [1]

        return torch.stack(pred, dim=1)  # [1,1,32,32]

    def calcu_prior_sim(self, qry_fts, fg_prototypes, bg_prototypes):
        temp_bg = bg_prototypes.unsqueeze(-1).unsqueeze(-1)  # [1,256,1,1]
        similarity_bg = F.cosine_similarity(qry_fts, temp_bg, dim=1).unsqueeze(1)  # [1,1,32,32]

        temp_fg = fg_prototypes.unsqueeze(-1).unsqueeze(-1)  # [1,256,1,1]
        similarity_fg = F.cosine_similarity(qry_fts, temp_fg, dim=1).unsqueeze(1)  # [1,1,32,32]

        out = torch.cat((similarity_bg, similarity_fg), dim=1) * self.scaler   # [1,2,32,32]
        return out