import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from method_tvr.model_components import BertAttention, LinearLayer, BertSelfAttention, TrainablePositionalEncoding
from method_tvr.model_components import MILNCELoss
from method_tvr.contrastive import batch_video_query_loss


class ReLoCLNet(nn.Module):
    def __init__(self, config):
        super(ReLoCLNet, self).__init__()
        self.config = config

        self.query_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_desc_l,
                                                           hidden_size=config.hidden_size, dropout=config.input_drop)
        self.ctx_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                         hidden_size=config.hidden_size, dropout=config.input_drop)

        self.query_input_proj = LinearLayer(config.query_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)

        self.query_encoder = BertAttention(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop))
        self.query_encoder1 = copy.deepcopy(self.query_encoder)

        cross_att_cfg = edict(hidden_size=config.hidden_size, num_attention_heads=config.n_heads,
                              attention_probs_dropout_prob=config.drop)
        # use_video
        self.video_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        self.video_encoder1 = copy.deepcopy(self.query_encoder)
        self.video_encoder2 = copy.deepcopy(self.query_encoder)
        self.video_encoder3 = copy.deepcopy(self.query_encoder)
        self.video_cross_att = BertSelfAttention(cross_att_cfg)
        self.video_cross_layernorm = nn.LayerNorm(config.hidden_size)
        self.video_query_linear = nn.Linear(config.hidden_size, config.hidden_size)

        # use_sub
        self.sub_input_proj = LinearLayer(config.sub_input_size, config.hidden_size, layer_norm=True,
                                          dropout=config.input_drop, relu=True)
        self.sub_encoder1 = copy.deepcopy(self.query_encoder)
        self.sub_encoder2 = copy.deepcopy(self.query_encoder)
        self.sub_encoder3 = copy.deepcopy(self.query_encoder)
        self.sub_cross_att = BertSelfAttention(cross_att_cfg)
        self.sub_cross_layernorm = nn.LayerNorm(config.hidden_size)
        self.sub_query_linear = nn.Linear(config.hidden_size, config.hidden_size)

        self.modular_vector_mapping = nn.Linear(in_features=config.hidden_size, out_features=2, bias=False)

        conv_cfg = dict(in_channels=1, out_channels=1, kernel_size=config.conv_kernel_size,
                        stride=config.conv_stride, padding=config.conv_kernel_size // 2, bias=False)
        self.merged_st_predictor = nn.Conv1d(**conv_cfg)
        self.merged_ed_predictor = nn.Conv1d(**conv_cfg)

        self.temporal_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.nce_criterion = MILNCELoss(reduction='mean')

        self.reset_parameters()

    def reset_parameters(self):
        """ Initialize the weights."""
        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(re_init)

    def set_hard_negative(self, use_hard_negative, hard_pool_size):
        """use_hard_negative: bool; hard_pool_size: int, """
        self.config.use_hard_negative = use_hard_negative
        self.config.hard_pool_size = hard_pool_size

    def set_train_st_ed(self, lw_st_ed):
        """pre-train video retrieval then span prediction"""
        self.config.lw_st_ed = lw_st_ed

    def forward(self, query_feat, query_mask, video_feat, video_mask, sub_feat, sub_mask, st_ed_indices, match_labels):
        """
        Args:
            query_feat: (N, Lq, Dq)
            query_mask: (N, Lq)
            video_feat: (N, Lv, Dv) or None
            video_mask: (N, Lv) or None
            sub_feat: (N, Lv, Ds) or None
            sub_mask: (N, Lv) or None
            st_ed_indices: (N, 2), torch.LongTensor, 1st, 2nd columns are st, ed labels respectively.
            match_labels: (N, Lv), torch.LongTensor, matching labels for detecting foreground and background (not used)
        """
        video_feat, sub_feat, mid_x_video_feat, mid_x_sub_feat, x_video_feat, x_sub_feat = self.encode_context(
            video_feat, video_mask, sub_feat, sub_mask, return_mid_output=True)
        video_query, sub_query, query_context_scores, st_prob, ed_prob = self.get_pred_from_raw_query(
            query_feat, query_mask, x_video_feat, video_mask, x_sub_feat, sub_mask, cross=False,
            return_query_feats=True)
        # frame level contrastive learning loss (FrameCL)
        loss_fcl = 0
        if self.config.lw_fcl != 0:
            loss_fcl_vq = batch_video_query_loss(mid_x_video_feat, video_query, match_labels, video_mask, measure='JSD')
            loss_fcl_sq = batch_video_query_loss(mid_x_sub_feat, sub_query, match_labels, sub_mask, measure='JSD')
            loss_fcl = (loss_fcl_vq + loss_fcl_sq) / 2.0
            loss_fcl = self.config.lw_fcl * loss_fcl
        # video level contrastive learning loss (VideoCL)
        loss_vcl = 0
        if self.config.lw_vcl != 0:
            mid_video_q2ctx_scores = self.get_unnormalized_video_level_scores(video_query, mid_x_video_feat, video_mask)
            mid_sub_q2ctx_scores = self.get_unnormalized_video_level_scores(sub_query, mid_x_sub_feat, sub_mask)
            mid_video_q2ctx_scores, _ = torch.max(mid_video_q2ctx_scores, dim=1)
            mid_sub_q2ctx_scores, _ = torch.max(mid_sub_q2ctx_scores, dim=1)
            mid_q2ctx_scores = (mid_video_q2ctx_scores + mid_sub_q2ctx_scores) / 2.0
            loss_vcl = self.nce_criterion(mid_q2ctx_scores)
            loss_vcl = self.config.lw_vcl * loss_vcl
        # moment localization loss
        loss_st_ed = 0
        if self.config.lw_st_ed != 0:
            loss_st = self.temporal_criterion(st_prob, st_ed_indices[:, 0])
            loss_ed = self.temporal_criterion(ed_prob, st_ed_indices[:, 1])
            loss_st_ed = loss_st + loss_ed
            loss_st_ed = self.config.lw_st_ed * loss_st_ed
        # video level retrieval loss
        loss_neg_ctx, loss_neg_q = 0, 0
        if self.config.lw_neg_ctx != 0 or self.config.lw_neg_q != 0:
            loss_neg_ctx, loss_neg_q = self.get_video_level_loss(query_context_scores)
            loss_neg_ctx = self.config.lw_neg_ctx * loss_neg_ctx
            loss_neg_q = self.config.lw_neg_q * loss_neg_q
        # sum loss
        loss = loss_fcl + loss_vcl + loss_st_ed + loss_neg_ctx + loss_neg_q
        return loss, {"loss_st_ed": float(loss_st_ed), "loss_fcl": float(loss_fcl), "loss_vcl": loss_vcl,
                      "loss_neg_ctx": float(loss_neg_ctx), "loss_neg_q": float(loss_neg_q), "loss_overall": float(loss)}

    def encode_query(self, query_feat, query_mask):
        encoded_query = self.encode_input(query_feat, query_mask, self.query_input_proj, self.query_encoder,
                                          self.query_pos_embed)  # (N, Lq, D)
        encoded_query = self.query_encoder1(encoded_query, query_mask.unsqueeze(1))
        video_query, sub_query = self.get_modularized_queries(encoded_query, query_mask)  # (N, D) * 2
        return video_query, sub_query

    def encode_context(self, video_feat, video_mask, sub_feat, sub_mask, return_mid_output=False):
        # encoding video and subtitle features, respectively
        encoded_video_feat = self.encode_input(video_feat, video_mask, self.video_input_proj, self.video_encoder1,
                                               self.ctx_pos_embed)
        encoded_sub_feat = self.encode_input(sub_feat, sub_mask, self.sub_input_proj, self.sub_encoder1,
                                             self.ctx_pos_embed)
        # cross encoding subtitle features
        x_encoded_video_feat = self.cross_context_encoder(encoded_video_feat, video_mask, encoded_sub_feat, sub_mask,
                                                          self.video_cross_att, self.video_cross_layernorm)  # (N, L, D)
        x_encoded_video_feat_ = self.video_encoder2(x_encoded_video_feat, video_mask.unsqueeze(1))
        # cross encoding video features
        x_encoded_sub_feat = self.cross_context_encoder(encoded_sub_feat, sub_mask, encoded_video_feat, video_mask,
                                                        self.sub_cross_att, self.sub_cross_layernorm)  # (N, L, D)
        x_encoded_sub_feat_ = self.sub_encoder2(x_encoded_sub_feat, sub_mask.unsqueeze(1))
        # additional self encoding process
        x_encoded_video_feat = self.video_encoder3(x_encoded_video_feat_, video_mask.unsqueeze(1))
        x_encoded_sub_feat = self.sub_encoder3(x_encoded_sub_feat_, sub_mask.unsqueeze(1))
        if return_mid_output:
            return (encoded_video_feat, encoded_sub_feat, x_encoded_video_feat_, x_encoded_sub_feat_,
                    x_encoded_video_feat, x_encoded_sub_feat)
        else:
            return x_encoded_video_feat, x_encoded_sub_feat

    @staticmethod
    def cross_context_encoder(main_context_feat, main_context_mask, side_context_feat, side_context_mask,
                              cross_att_layer, norm_layer):
        """
        Args:
            main_context_feat: (N, Lq, D)
            main_context_mask: (N, Lq)
            side_context_feat: (N, Lk, D)
            side_context_mask: (N, Lk)
            cross_att_layer: cross attention layer
            norm_layer: layer norm layer
        """
        cross_mask = torch.einsum("bm,bn->bmn", main_context_mask, side_context_mask)  # (N, Lq, Lk)
        cross_out = cross_att_layer(main_context_feat, side_context_feat, side_context_feat, cross_mask)  # (N, Lq, D)
        residual_out = norm_layer(cross_out + main_context_feat)
        return residual_out

    @staticmethod
    def encode_input(feat, mask, input_proj_layer, encoder_layer, pos_embed_layer):
        """
        Args:
            feat: (N, L, D_input), torch.float32
            mask: (N, L), torch.float32, with 1 indicates valid query, 0 indicates mask
            input_proj_layer: down project input
            encoder_layer: encoder layer
            pos_embed_layer: positional embedding layer
        """
        feat = input_proj_layer(feat)
        feat = pos_embed_layer(feat)
        mask = mask.unsqueeze(1)  # (N, 1, L), torch.FloatTensor
        return encoder_layer(feat, mask)  # (N, L, D_hidden)

    def get_modularized_queries(self, encoded_query, query_mask, return_modular_att=False):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
            return_modular_att: bool
        """
        modular_attention_scores = self.modular_vector_mapping(encoded_query)  # (N, L, 2 or 1)
        modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, query_mask.unsqueeze(2)), dim=1)
        modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query)  # (N, 2 or 1, D)
        if return_modular_att:
            assert modular_queries.shape[1] == 2
            return modular_queries[:, 0], modular_queries[:, 1], modular_attention_scores
        else:
            assert modular_queries.shape[1] == 2
            return modular_queries[:, 0], modular_queries[:, 1]  # (N, D) * 2

    @staticmethod
    def get_video_level_scores(modularied_query, context_feat, context_mask):
        """ Calculate video2query scores for each pair of video and query inside the batch.
        Args:
            modularied_query: (N, D)
            context_feat: (N, L, D), output of the first transformer encoder layer
            context_mask: (N, L)
        Returns:
            context_query_scores: (N, N)  score of each query w.r.t. each video inside the batch,
                diagonal positions are positive. used to get negative samples.
        """
        modularied_query = F.normalize(modularied_query, dim=-1)
        context_feat = F.normalize(context_feat, dim=-1)
        query_context_scores = torch.einsum("md,nld->mln", modularied_query, context_feat)  # (N, L, N)
        context_mask = context_mask.transpose(0, 1).unsqueeze(0)  # (1, L, N)
        query_context_scores = mask_logits(query_context_scores, context_mask)  # (N, L, N)
        query_context_scores, _ = torch.max(query_context_scores, dim=1)  # (N, N) diagonal positions are positive pairs
        return query_context_scores

    @staticmethod
    def get_unnormalized_video_level_scores(modularied_query, context_feat, context_mask):
        """ Calculate video2query scores for each pair of video and query inside the batch.
        Args:
            modularied_query: (N, D)
            context_feat: (N, L, D), output of the first transformer encoder layer
            context_mask: (N, L)
        Returns:
            context_query_scores: (N, N)  score of each query w.r.t. each video inside the batch,
                diagonal positions are positive. used to get negative samples.
        """
        query_context_scores = torch.einsum("md,nld->mln", modularied_query, context_feat)  # (N, L, N)
        context_mask = context_mask.transpose(0, 1).unsqueeze(0)  # (1, L, N)
        query_context_scores = mask_logits(query_context_scores, context_mask)  # (N, L, N)
        return query_context_scores

    def get_merged_score(self, video_query, video_feat, sub_query, sub_feat, cross=False):
        video_query = self.video_query_linear(video_query)
        sub_query = self.sub_query_linear(sub_query)
        if cross:
            video_similarity = torch.einsum("md,nld->mnl", video_query, video_feat)
            sub_similarity = torch.einsum("md,nld->mnl", sub_query, sub_feat)
            similarity = (video_similarity + sub_similarity) / 2  # (Nq, Nv, L)  from query to all videos.
        else:
            video_similarity = torch.einsum("bd,bld->bl", video_query, video_feat)  # (N, L)
            sub_similarity = torch.einsum("bd,bld->bl", sub_query, sub_feat)  # (N, L)
            similarity = (video_similarity + sub_similarity) / 2
        return similarity

    def get_merged_st_ed_prob(self, similarity, context_mask, cross=False):
        if cross:
            n_q, n_c, length = similarity.shape
            similarity = similarity.view(n_q * n_c, 1, length)
            st_prob = self.merged_st_predictor(similarity).view(n_q, n_c, length)  # (Nq, Nv, L)
            ed_prob = self.merged_ed_predictor(similarity).view(n_q, n_c, length)  # (Nq, Nv, L)
        else:
            st_prob = self.merged_st_predictor(similarity.unsqueeze(1)).squeeze()  # (N, L)
            ed_prob = self.merged_ed_predictor(similarity.unsqueeze(1)).squeeze()  # (N, L)
        st_prob = mask_logits(st_prob, context_mask)  # (N, L)
        ed_prob = mask_logits(ed_prob, context_mask)
        return st_prob, ed_prob

    def get_pred_from_raw_query(self, query_feat, query_mask, video_feat, video_mask, sub_feat, sub_mask, cross=False,
                                return_query_feats=False):
        """
        Args:
            query_feat: (N, Lq, Dq)
            query_mask: (N, Lq)
            video_feat: (N, Lv, D) or None
            video_mask: (N, Lv)
            sub_feat: (N, Lv, D) or None
            sub_mask: (N, Lv)
            cross:
            return_query_feats:
        """
        video_query, sub_query = self.encode_query(query_feat, query_mask)
        # get video-level retrieval scores
        video_q2ctx_scores = self.get_video_level_scores(video_query, video_feat, video_mask)
        sub_q2ctx_scores = self.get_video_level_scores(sub_query, sub_feat, sub_mask)
        q2ctx_scores = (video_q2ctx_scores + sub_q2ctx_scores) / 2  # (N, N)
        # compute start and end probs
        similarity = self.get_merged_score(video_query, video_feat, sub_query, sub_feat, cross=cross)
        st_prob, ed_prob = self.get_merged_st_ed_prob(similarity, video_mask, cross=cross)
        if return_query_feats:
            return video_query, sub_query, q2ctx_scores, st_prob, ed_prob
        else:
            return q2ctx_scores, st_prob, ed_prob  # un-normalized masked probabilities!!!!!

    def get_video_level_loss(self, query_context_scores):
        """ ranking loss between (pos. query + pos. video) and (pos. query + neg. video) or (neg. query + pos. video)
        Args:
            query_context_scores: (N, N), cosine similarity [-1, 1],
                Each row contains the scores between the query to each of the videos inside the batch.
        """
        bsz = len(query_context_scores)
        diagonal_indices = torch.arange(bsz).to(query_context_scores.device)
        pos_scores = query_context_scores[diagonal_indices, diagonal_indices]  # (N, )
        query_context_scores_masked = copy.deepcopy(query_context_scores.data)
        # impossibly large for cosine similarity, the copy is created as modifying the original will cause error
        query_context_scores_masked[diagonal_indices, diagonal_indices] = 999
        pos_query_neg_context_scores = self.get_neg_scores(query_context_scores, query_context_scores_masked)
        neg_query_pos_context_scores = self.get_neg_scores(query_context_scores.transpose(0, 1),
                                                           query_context_scores_masked.transpose(0, 1))
        loss_neg_ctx = self.get_ranking_loss(pos_scores, pos_query_neg_context_scores)
        loss_neg_q = self.get_ranking_loss(pos_scores, neg_query_pos_context_scores)
        return loss_neg_ctx, loss_neg_q

    def get_neg_scores(self, scores, scores_masked):
        """
        scores: (N, N), cosine similarity [-1, 1],
            Each row are scores: query --> all videos. Transposed version: video --> all queries.
        scores_masked: (N, N) the same as scores, except that the diagonal (positive) positions
            are masked with a large value.
        """
        bsz = len(scores)
        batch_indices = torch.arange(bsz).to(scores.device)
        _, sorted_scores_indices = torch.sort(scores_masked, descending=True, dim=1)
        sample_min_idx = 1  # skip the masked positive
        sample_max_idx = min(sample_min_idx + self.config.hard_pool_size, bsz) if self.config.use_hard_negative else bsz
        # (N, )
        sampled_neg_score_indices = sorted_scores_indices[batch_indices, torch.randint(sample_min_idx, sample_max_idx,
                                                                                       size=(bsz,)).to(scores.device)]
        sampled_neg_scores = scores[batch_indices, sampled_neg_score_indices]  # (N, )
        return sampled_neg_scores

    def get_ranking_loss(self, pos_score, neg_score):
        """ Note here we encourage positive scores to be larger than negative scores.
        Args:
            pos_score: (N, ), torch.float32
            neg_score: (N, ), torch.float32
        """
        if self.config.ranking_loss_type == "hinge":  # max(0, m + S_neg - S_pos)
            return torch.clamp(self.config.margin + neg_score - pos_score, min=0).sum() / len(pos_score)
        elif self.config.ranking_loss_type == "lse":  # log[1 + exp(S_neg - S_pos)]
            return torch.log1p(torch.exp(neg_score - pos_score)).sum() / len(pos_score)
        else:
            raise NotImplementedError("Only support 'hinge' and 'lse'")


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)
