# coding=utf-8
# Copyright 2022 The Uncertainty Baselines Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=line-too-long
r"""Finetune a pretrained ViT-L/32 BE model on CIFAR-10/100 and ImageNet.

This config is used for models pretrained on either JFT-300M or ImageNet-21K.

"""
# pylint: enable=line-too-long

import ml_collections
from experiments import sweep_utils  # local file import from baselines.jft


def get_config():
  """Config for finetuning."""
  config = ml_collections.ConfigDict()
  
  size=384
  steps=10_000
  warmup=500

  name = 'cifar10'
  n_cls = 10
  pp_common = '|value_range(-1, 1)'
  pp_common += f'|onehot({n_cls}, key="label", key_result="labels")'
  pp_common += '|keep(["image", "labels", "id"])'
  pp_train = f'decode|inception_crop({size})|flip_lr' + pp_common
  pp_eval = f'decode|resize({size})' + pp_common

  config = ml_collections.ConfigDict()
  config.dataset = name
  config.train_split = 'train[:98%]'
  config.pp_train = pp_train
  config.val_split = 'train[98%:]'
  config.test_split = 'test'
  config.pp_eval = pp_eval
  config.num_classes = n_cls
  config.lr = ml_collections.ConfigDict()
  config.lr.warmup_steps = warmup
  config.total_steps = steps
  config.batch_size = 32
  config.shuffle_buffer_size = 50_000

  config.log_training_steps = 100
  config.log_eval_steps = 1000
  config.checkpoint_steps = 5000
  config.checkpoint_timeout = 1

  config.prefetch_to_device = 2
  config.trial = 0

  # OOD evaluation
  # ood_split is the data split for both the ood_dataset and the dataset.
  config.ood_datasets = ['cifar100']
  config.ood_num_classes = [100, 10]
  config.ood_split = 'test'
  config.ood_methods = ['msp', 'entropy', 'maha', 'rmaha', 'mlogit']
  pp_eval_ood = []
  for num_classes in config.ood_num_classes:
    if num_classes > config.num_classes:
      # Note that evaluation_fn ignores the entries with all zero labels for
      # evaluation. When num_classes > n_cls, we should use onehot{num_classes},
      # otherwise the labels that are greater than n_cls will be encoded with
      # all zeros and then be ignored.
      pp_eval_ood.append(
          pp_eval.replace(f'onehot({config.num_classes}',
                          f'onehot({num_classes}'))
    else:
      pp_eval_ood.append(pp_eval)
  config.pp_eval_ood = pp_eval_ood

  # Model section
  config.model_init = 'gs://plex-paper/plex_vit_large_imagenet21k.npz'
  config.model = ml_collections.ConfigDict()
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [32, 32]
  config.model.hidden_size = 1024
  config.model.transformer = ml_collections.ConfigDict()
  config.model.transformer.mlp_dim = 4096
  config.model.transformer.num_heads = 16
  config.model.transformer.num_layers = 24
  config.model.transformer.attention_dropout_rate = 0.
  config.model.transformer.dropout_rate = 0.
  config.model.classifier = 'token'
  # This is "no head" fine-tuning, which we use by default.
  config.model.representation_size = None

  # Heteroscedastic
  # TODO(trandustin): Enable finetuning jobs with multi-class HetSNGP layer.
  config.model.multiclass = True
  config.model.temperature = 0.2
  config.model.mc_samples = 1000
  config.model.num_factors = 0 #for cifar10 there are only 10 classes so full diagonal is okay
  config.model.param_efficient = False

  # BatchEnsemble
  config.model.transformer.be_layers = (21, 22, 23)  # Set in sweep.
  config.model.transformer.ens_size = 3  # Set in sweep.
  config.model.transformer.random_sign_init = -0.5
  # TODO(trandustin): Remove `ensemble_attention` hparam once we no longer
  # need checkpoints that only apply BE on the FF block.
  config.model.transformer.ensemble_attention = False
  config.fast_weight_lr_multiplier = 1.0

  # GP
  config.model.use_gp = False
  # Use momentum-based (i.e., non-exact) covariance update for pre-training.
  # This is because the exact covariance update can be unstable for pretraining,
  # since it involves inverting a precision matrix accumulated over 300M data.
  config.model.covmat_momentum = .999
  config.model.ridge_penalty = 1.
  # No need to use mean field adjustment for pretraining.
  config.model.mean_field_factor = -1.

  # Optimizer section
  config.optim_name = 'Momentum'
  config.optim = ml_collections.ConfigDict()
  config.grad_clip_norm = 1.0
  config.weight_decay = None
  config.loss = 'softmax_xent'

  config.lr = ml_collections.ConfigDict()
  config.lr.base = 0.03  # set in sweep
  config.lr.warmup_steps = 500
  config.lr.decay_type = 'cosine'
  return config
