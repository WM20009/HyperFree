import torch
import torch.nn.functional as F
from functools import partial
from HyperFree.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer


def build_HyperFree_vit_h(checkpoint=None):
    return _build_HyperFree(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_HyperFree = build_HyperFree_vit_h


def build_HyperFree_vit_l(checkpoint=None):
    return _build_HyperFree(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_HyperFree_vit_b(checkpoint=None, image_size=1024):
    return _build_HyperFree(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        image_size=image_size,
    )


HyperFree_model_registry = {
    "default": build_HyperFree_vit_h,
    "vit_h": build_HyperFree_vit_h,
    "vit_l": build_HyperFree_vit_l,
    "vit_b": build_HyperFree_vit_b,
}


def load_and_resize_params(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_dict = model.state_dict()
    
    for k, v in checkpoint.items():
        if k in model_dict:
            if v.shape != model_dict[k].shape:
                if 'pos_embed' in k:
                    v = F.interpolate(v.permute((0,3,1,2)), size=(model_dict[k].shape[1], model_dict[k].shape[2]), mode='nearest').permute((0,2,3,1))
                elif 'rel_pos' in k:
                    v = F.interpolate(v.unsqueeze(0).unsqueeze(0), size=(model_dict[k].shape[0], model_dict[k].shape[1]),).squeeze(0).squeeze(0)
            model_dict[k] = v
    
    model.load_state_dict(model_dict, strict=False)
    return model


def _build_HyperFree(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    image_size=1024,
):
    prompt_embed_dim = 256
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    hyperfree = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
    )

    hyperfree.eval()
    if checkpoint is not None:
        load_and_resize_params(hyperfree, checkpoint)
    return hyperfree