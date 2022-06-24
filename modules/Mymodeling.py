from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
#from msilib import sequence

import torch
from torch import nn

from modules.until_module import PreTrainedModel, AllGather, CrossEn
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip

from modules.my_module_clip import CLIP, convert_weights
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

logger = logging.getLogger(__name__)
allgather = AllGather.apply

class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, cross_config, *inputs, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            ###### 20220524 local rank
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        if state_dict is None: state_dict = {}
        pretrained_clip_name = "ViT-B/32"
        if hasattr(task_config, 'pretrained_clip_name'):
            pretrained_clip_name = task_config.pretrained_clip_name
        clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)

        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)

        ## ===> Initialization trick [HARD CODE]
        if model.linear_patch == "3d":
            # contain_conv2 = False
            # for key in state_dict.keys():
            #     if key.find("visual.conv2.weight") > -1:
            #         contain_conv2 = True
            #         break
            # if contain_conv2 is False and hasattr(model.clip.visual, "conv2"):
            #     print("model.clip.visual :",model.clip.visual)
            #     cp_weight = state_dict["clip.visual.conv1.weight"].clone()


            #     ################### 층 하나 통과 한 cp_weight를 만들기 그 cp는 VaeT의 shape와 같음 ######
            #     print("cp_weight :",cp_weight.shape)
            #     kernel_size = model.clip.visual.conv2.weight.size(2)
            #     print("kernel size : ",kernel_size)
            #     conv2_size = model.clip.visual.conv2.weight.size()
            #     print("conv2_size :",conv2_size)
            #     conv2_size = list(conv2_size)
            #     print("conv2_size :",conv2_size)

            #     left_conv2_size = conv2_size.copy()
            #     print("left_conc2_size :",left_conv2_size)
            #     right_conv2_size = conv2_size.copy()
            #     print("right conv2_size :",right_conv2_size)
            #     left_conv2_size[2] = (kernel_size - 1) // 2
            #     print("left_conv2_size[2] shape :",left_conv2_size[2])
            #     right_conv2_size[2] = kernel_size - 1 - left_conv2_size[2]
            #     print("right_conv2_size[2] shape :",right_conv2_size[2])
            #     left_zeros, right_zeros = None, None
            #     if left_conv2_size[2] > 0:
            #         left_zeros = torch.zeros(*tuple(left_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)
            #         print("left_zeros shape :",left_zeros.shape)
            #     if right_conv2_size[2] > 0:
            #         right_zeros = torch.zeros(*tuple(right_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)
            #         print("right_zeros shape :",right_zeros.shape)
            #     cat_list = []
            #     if left_zeros != None: cat_list.append(left_zeros)
            #     print("first cat list :",cat_list[-1].shape)
            #     ###20220620 변경
            #     #cat_list.append(cp_weight.unsqueeze(2))
            #     cp_weight = torch.randn(768,64,3,7,1) 
            #     cat_list.append(cp_weight)
            #     print("second cat_list :",cat_list[-1].shape)
            #     if right_zeros != None: cat_list.append(right_zeros)
            #     print("third cat_list :",cat_list[-1].shape)
            #     print("<<<cat_list : {} >>>".format(len(cat_list)))
            #     print("<<<cat 1 : {} cat 2 : {} cat 3 : {}".format(len(cat_list[0]), len(cat_list[1]),len(cat_list[2])))
            #     print("shape 1 : {} shape 2 : {} shape 3 : {}".format(cat_list[0].shape,cat_list[1].shape,cat_list[2].shape))
            #     print("<<< before >>>")
            #     cp_weight = torch.cat(cat_list, dim=2)
            #     print("<<< after >>>")
            #     state_dict["clip.visual.conv2.weight"] = cp_weight
            pass

        ## <=== End of initialization trick

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]

class CLIP4Clip(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(CLIP4Clip, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self._stage_one = True
        self._stage_two = False

        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.loose_type = False
        if self._stage_one and check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))

        self.linear_patch = '3d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))

        #CLIP = text transformer + VAeT
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,
            linear_patch=self.linear_patch
        ).float()

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        convert_weights(self.clip)



        self.sim_header = 'meanP'
        # 20220620 아래 4개 코드 주석 처리함 
        # if hasattr(task_config, "sim_header"):
        #     self.sim_header = task_config.sim_header
        #     show_log(task_config, "\t sim_header: {}".format(self.sim_header))
        # if self.sim_header == "tightTransf": assert self.loose_type is False

        self.loss_fct = CrossEn()
        self.apply(self.init_weights)


    # video : 7D
    def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None):
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])

        video_frame = video.shape[2]* video.shape[3]
        sequence_output, visual_output, decoded_output,vaet_video = self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask,
                                                                         video, video_mask, shaped=True, video_frame=video_frame)

        if self.training:
            total = 0.
            loss = 0.
            alpha = 0.5
            #seq output : [1 x 512] 
            #vis output(ours) : [1 x 512]
            # decoded output(ours) : [1, 16, 3, 224, 224]
            #
            sim_matrix= self.get_similarity_logits(sequence_output, visual_output, 
                                        attention_mask, video_mask,shaped=True,loose_type=self.loose_type)
            mse_loss = self.get_mse_loss(decoded_output,vaet_video)
            sim_loss1 = self.loss_fct(sim_matrix)
            sim_loss2 = self.loss_fct(sim_matrix.T)
            sim_loss = (sim_loss1 + sim_loss2) / 2
            loss = sim_loss + mse_loss
            #total += (sim_loss * (1-alpha) + mse_loss * alpha)

            return loss
        else:
            return None

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        sequence_hidden = self.clip.encode_text(input_ids)
        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))

        return sequence_hidden

    def get_visual_output(self, video, video_mask, video_frame=-1):
  
        bs_pair = video_mask.size(0)
        video = torch.as_tensor(video).float()
        b, pair, bs, ts, channel, h, w = video.shape
        video_frame = bs * ts
        vaet_video = video.permute(0,1,3,4,5,6,2).contiguous()
        vaet_video=vaet_video.float()
        vaet_video = vaet_video.reshape(b*pair, channel, h, w, video_frame)
        visual_hidden, decoded = self.clip.encode_image(vaet_video, video_frame=video_frame)
        # visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))
        # (1, 512)로 이미 맞춰기 때문에 pooling을 위한 reshape가 필요 없다.

        return visual_hidden, decoded,vaet_video
    #video 4D
    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False, video_frame=-1):
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])

        video = torch.as_tensor(video).float()
        # b, pair, bs, ts, channel, h, w = video.shape
        # video_frame = bs * ts
        # vaet_video = video.permute(0,1,3,4,5,6,2).contiguous()
        # vaet_video=vaet_video.float()
        # vaet_video = vaet_video.reshape(b*pair, channel, h, w, video_frame)
        #print("<<<<vaet_video shape : {} >>>>>".format(vaet_video.shape))
        sequence_output = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True)
        visual_output, decoded_output,vaet_video = self.get_visual_output(video, video_mask, video_frame=video_frame)

        # visual_output : MSVD_DATASET => B*F, Grid^2+1, 512
        # decoded_output : b*pair, channel, h, w, video_frame
        return sequence_output, visual_output, decoded_output,vaet_video

    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)
        return text_out

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask,):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        #print("video_out shpe :",video_out.shape)
        return video_out

    def _loose_similarity(self, sequence_output, visual_output, attention_mask, video_mask, sim_header="meanP"):
        
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()
        #sequence_output, visual_output = sequence_output.float(), visual_output.float()
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        sequence_output = sequence_output.squeeze(1)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
        #print("<<<<< visual output shape : {} sequence output shape : {}".format(visual_output.shape,sequence_output.shape))
        logit_scale = self.clip.logit_scale.exp()
        retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.t())
        return retrieve_logits

    # text + video 사이의 sim_maatrix를 구하기 위한 작업
    #sequence_output, visual_output, 
     #                                   attention_mask, video_mask,shaped=True,loose_type=self.loose_type)
    def get_similarity_logits(self, sequence_output, visual_output, attention_mask, video_mask,shaped=False, loose_type=False):
        
        
        visual_output_ = torch.clone(visual_output)
        
        #visual_output_ = visual_output_.reshape(visual_output_.shape[0],3,224,224,16)
        #print("1 <<<<< visual output : {} visual output_ : {} decoded : {}>>>>>".format(visual_output.shape, visual_output_.shape,decoded.shape))
        #assert visual_output_.shape == decoded.shape 
        #mse_loss = nn.MSELoss()(vaet_video, decoded)
        retrieve_logits = self._loose_similarity(sequence_output, visual_output, attention_mask, video_mask, sim_header=self.sim_header)

        return retrieve_logits

    def get_mse_loss(self,decoded,vaet_video):
        mse_loss = nn.MSELoss()(vaet_video, decoded)
        return mse_loss
