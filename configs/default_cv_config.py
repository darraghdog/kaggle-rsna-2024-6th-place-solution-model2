from default_config import basic_cfg

cfg = basic_cfg

# img model
cfg.backbone = "tf_efficientnet_b0_ns"
cfg.pretrained = True
cfg.pool = "avg"
cfg.in_chans = 3
cfg.gem_p_trainable = False
cfg.stem_stride = (2,2)


basic_cv_cfg = cfg
