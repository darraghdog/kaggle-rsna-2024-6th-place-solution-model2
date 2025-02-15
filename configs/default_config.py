from types import SimpleNamespace

cfg = SimpleNamespace(**{})

# stages
cfg.train = True
cfg.val = True
cfg.test = True
cfg.train_val = True

# dataset
cfg.dataset = "ds_dummy"
cfg.batch_size = 32
cfg.batch_size_val = None

cfg.val_df = None
cfg.test_df = None

cfg.train_aug = None
cfg.val_aug = None
cfg.data_sample = -1

# model
cfg.pretrained = True
cfg.pretrained_weights = None
cfg.pretrained_weights_strict = True
cfg.pop_weights = None

# training routine
cfg.fold = 0
cfg.optimizer = "Adam"
cfg.lr = 1e-4
cfg.schedule = "cosine"
cfg.weight_decay = 0
cfg.epochs = 10
cfg.seed = -1
cfg.resume_training = False

cfg.clip_grad = 0
cfg.save_val_data = True
cfg.gradient_checkpointing = False

# eval
cfg.eval_ddp = True
cfg.calc_metric = True
cfg.calc_metric_epochs = 1
cfg.eval_steps = 0
cfg.eval_epochs = 1
cfg.post_process_pipeline = "pp_dummy"
cfg.metric = 'default_metric'
cfg.viz = "default_viz"

# ressources
cfg.find_unused_parameters = False
cfg.mixed_precision = True
cfg.grad_accumulation = 1
cfg.syncbn = False
cfg.gpu = 0
cfg.dp = False
cfg.num_workers = 8
cfg.drop_last = True
cfg.save_checkpoint = True
cfg.save_only_last_ckpt = False
cfg.save_weights_only = False
cfg.save_first_batch = False

# logging,
cfg.neptune_project = None
cfg.neptune_connection_mode = "debug"
cfg.save_first_batch = False
cfg.save_first_batch_preds = False
cfg.clip_mode = "norm"
cfg.data_sample = -1
cfg.track_grad_norm = True
cfg.grad_norm_type = 2.
cfg.norm_eps = 1e-4
cfg.disable_tqdm = False
cfg.tags = "base"
cfg.flush_period = 30

# aws
cfg.s3_bucket_name = ""
cfg.s3_access_key = ""
cfg.s3_secret_key = ""

basic_cfg = cfg
