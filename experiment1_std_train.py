import sys, os
import torch as ch
import numpy as np
import cox.store
from cox.utils import Parameters
from datetime import datetime
import time

from robustness import model_utils, datasets, train, defaults
from robustness.datasets import CIFAR, HAM10000, HAM10000_dataset, HAM10000_3cls, HAM10000_dataset_3cls_balanced, freeze, unfreeze
from robustness.tools.utils import fix_random_seed
from robustness.evaluation import plot_curves_from_log, evaluate_model
'''
-------------------------------experiment1_std_train:lr==1e-3,batch==32,train_layer==5------------------------------------
'''
print("-------------------------------Experiment1: std_train:lr==1e-3,batch==32,train_layer==5------------------------------------")
'''Required parameters'''
# Path
ds_path ="/content/data"
OUT_DIR ="/content/drive/My Drive/logs"
DATA_DIR = "/content/drive/My Drive/data"
device = 'cuda'
train_file_name="3cls_balanced_2400_train_val.csv::0"
test_file_name="3cls_balanced_test_without_val.csv"

# Training
ADV_TRAIN = False
ADV_EVAL = False
lr = 1e-4
BATCH_SIZE = 32
EPOCHS = 10
step_lr = None
custom_schedule = None
lr_patience = 5
es_patience = 10

# Model
base_model_expid = None
use_dropout_head = False
dropout_perc = 0
arch = 'resnet18'
pytorch_pretrained = True
unfreeze_to_layer = 0

# Other settings
do_eval_model = False
eval_checkpoint_type = 'best'
NUM_WORKERS = 2
expid_time=datetime.now().strftime("%Y-%m-%d---%H:%M:%S")
expid = f"experiment1_std_train_lr==1e-3_batch==32_train_layer==5_time=={expid_time}"
exp_1_id =expid
seed = 42

# Adversary
EPS = 0.5
ITERATIONS = 7
constraint = '2'

# Ablation
apply_ablation = False
perc_ablation = 0
saliency_dir = os.path.join(DATA_DIR, 'saliency_maps')
if ADV_TRAIN == False:
    saliency_dir = os.path.join(saliency_dir, 'standard')
else:
    saliency_dir = os.path.join(saliency_dir, f'adv {int(EPS)}')

train_kwargs = {
    'out_dir': "train_out",
    'adv_train': False,
    'adv_eval': False,
    'epochs': 10,
    'lr': 1e-3,
    'optimizer': 'Adam',
    'device': device,
    'batch_size': 32,
    'arch': "resnet18",
    'pytorch_pretrained': True,
    'dataset_file_name': train_file_name,
    'step_lr': None,
    'custom_schedule': None,
    'lr_patience': 5,
    'es_patience': 10,
    'log_iters': 1,
    'use_adv_prec': True,
    'apply_ablation': False,
    'saliency_dir': None,
    'perc_ablation': 0,
    'dropout_perc': 0,
    'use_dropout_head': False
}
attack_kwargs = {
    'constraint': "2",
    'eps': 0,
    'attack_lr': 0/5,
    'attack_steps': 7,
    'random_start': True
}
train_kwargs_merged = {**train_kwargs, **attack_kwargs}
print("-------------------------------Experiment1 First Name:\n",expid)

fix_random_seed(seed)
out_store = cox.store.Store(OUT_DIR, expid)
print("-------------------------------Experiment1 Out_Store:\n",out_store.exp_id)

train_kwargs_merged['base_model_expid'] = base_model_expid
if base_model_expid:
  resume_path = os.path.join(OUT_DIR, base_model_expid,"checkpoint.pt.latest")
else:
  resume_path = None

train_args = Parameters(train_kwargs_merged)
train_args = defaults.check_and_fill_args(train_args,
                        defaults.TRAINING_ARGS, HAM10000)
train_args = defaults.check_and_fill_args(train_args,
                        defaults.PGD_ARGS, CIFAR)
print("-------------------------------Experiment1 Required parameters: \n",train_args)


'''Create dataset and loader'''
print("-------------------------------Experiment1 Create Dataset And Loader: ")
dataset = HAM10000_3cls(ds_path,
                        file_name=train_file_name,
                        apply_ablation=train_args.apply_ablation,
                        saliency_dir=train_args.saliency_dir,
                        perc_ablation=train_args.perc_ablation,
                        use_dropout_head=train_args.use_dropout_head,
                        dropout_perc=train_args.dropout_perc
                        )

train_loader, val_loader = dataset.make_loaders(batch_size=train_args.batch_size,
                                                workers=NUM_WORKERS
                                                )


'''Create model'''
print("-------------------------------Experiment1 Create Model: ")
model, _ = model_utils.make_and_restore_model(
    arch=train_args.arch,
    pytorch_pretrained=train_args.pytorch_pretrained,
    dataset=dataset,
    resume_path=resume_path,
    device=device
)

if base_model_expid == None:
    freeze(model.model)
    unfreeze(model.model, 5)
else:
    model = model.module
    unfreeze(model.model, unfreeze_to_layer)
print(model.model)

'''Start model'''
start = time.time()
model_finetuned = train.train_model(train_args, model, (train_loader, val_loader), store=out_store)
end = time.time()
print("-------------------------------Training took %.2f sec" % (end - start))

plot_curves_from_log(out_store,True)
print(plot_curves_from_log(out_store)['logs'].df)
print("-------------------------------Experiment1 Out_Store:\n ",out_store.exp_id)
out_store.close()

'''Eval or do not eval model'''
if do_eval_model:
    # training dataset
    train_dataset = HAM10000_dataset_3cls_balanced(ds_path, train_file_name, train=True,
                                                   transform = dataset.transform_test,
                                                   apply_ablation=train_args.apply_ablation, saliency_dir=train_args.saliency_dir,
                                                   perc_ablation=train_args.perc_ablation)

    # test dataset
    test_dataset = HAM10000_dataset_3cls_balanced(ds_path, test_file_name, test=True,
                                                  transform = dataset.transform_test,
                                                  apply_ablation=train_args.apply_ablation, saliency_dir=train_args.saliency_dir,
                                                  perc_ablation=train_args.perc_ablation)

    accs = evaluate_model(out_store.exp_id, dataset, train_dataset, test_dataset, OUT_DIR, device, train_args.arch, checkpoint_type=eval_checkpoint_type)
    print("-------------------------------Experiment1 Accaury:\n",accs)





'''
-------------------------------experiment1_std_train:lr==3e-4,batch==16,train_layer==all layers------------------------------------
'''
print("-------------------------------Experiment1: std_train:lr==3e-4,batch==16,train_layer==all layers------------------------------------")
'''Required parameters'''
# Path
ds_path ="/content/data"
OUT_DIR ="/content/drive/My Drive/logs"
DATA_DIR = "/content/drive/My Drive/data"
device = 'cuda'
train_file_name="3cls_balanced_2400_train_val.csv::0"
test_file_name="3cls_balanced_test_without_val.csv"

# Training
ADV_TRAIN = False
ADV_EVAL = False
lr = 3e-4
BATCH_SIZE = 16
EPOCHS = 50
step_lr = 15
custom_schedule = 'plateau'
lr_patience = 5
es_patience = 10

# Model
base_model_expid = exp_1_id
use_dropout_head = False
dropout_perc = 0
arch = 'resnet18'
pytorch_pretrained = True
unfreeze_to_layer = 1

# Other settings
do_eval_model = True
eval_checkpoint_type = 'best'
NUM_WORKERS = 2
expid_time=datetime.now().strftime("%Y-%m-%d---%H:%M:%S")
expid = f"experiment1_std_train_lr==3e-4_batch==16_train_layer==all_layers_time=={expid_time}"
exp_2_id = expid
seed = 42

# Ablation
apply_ablation = False
perc_ablation = 0
saliency_dir = os.path.join(DATA_DIR, 'saliency_maps')
if ADV_TRAIN == False:
    saliency_dir = os.path.join(saliency_dir, 'standard')
else:
    saliency_dir = os.path.join(saliency_dir, f'adv {int(EPS)}')

# Adversary
EPS = 0.5
ITERATIONS = 7
constraint = '2'

train_kwargs = {
    'out_dir': "train_out",
    'adv_train': False,
    'adv_eval': False,
    'epochs': 50,
    'lr': 3e-4,
    'optimizer': 'Adam',
    'device': device,
    'batch_size': 16,
    'arch': "resnet18",
    'pytorch_pretrained': True,
    'dataset_file_name': train_file_name,
    'step_lr': 15,
    'custom_schedule': 'plateau',
    'lr_patience': 5,
    'es_patience': 10,
    'log_iters': 1,
    'use_adv_prec': True,
    'apply_ablation': False,
    'saliency_dir': saliency_dir,
    'perc_ablation': 0,
    'dropout_perc': 0,
    'use_dropout_head': False
}
attack_kwargs = {
    'constraint': "2",
    'eps': 0,
    'attack_lr': 0/5,
    'attack_steps': 7,
    'random_start': True
}
train_kwargs_merged = {**train_kwargs, **attack_kwargs}
print("-------------------------------Experiment1 First Name:\n",expid)

fix_random_seed(seed)
out_store = cox.store.Store(OUT_DIR, expid)
print("-------------------------------Experiment1 Out_Store:\n",out_store.exp_id)

train_kwargs_merged['base_model_expid'] = base_model_expid
if base_model_expid:
  resume_path = os.path.join(OUT_DIR, base_model_expid,"checkpoint.pt.latest")
else:
  resume_path = None

train_args = Parameters(train_kwargs_merged)
train_args = defaults.check_and_fill_args(train_args,
                        defaults.TRAINING_ARGS, HAM10000)
train_args = defaults.check_and_fill_args(train_args,
                        defaults.PGD_ARGS, CIFAR)
print("-------------------------------Experiment1 Required parameters: \n",train_args)

'''Create dataset and loader'''
print("-------------------------------Experiment1 Create Dataset And Loader: ")
dataset = HAM10000_3cls(ds_path,
                        file_name=train_file_name,
                        apply_ablation=train_args.apply_ablation,
                        saliency_dir=train_args.saliency_dir,
                        perc_ablation=train_args.perc_ablation,
                        use_dropout_head=train_args.use_dropout_head,
                        dropout_perc=train_args.dropout_perc
                        )

train_loader, val_loader = dataset.make_loaders(batch_size=train_args.batch_size,
                                                workers=NUM_WORKERS
                                                )

'''Create model'''
print("-------------------------------Experiment1 Create Model: ")
model, _ = model_utils.make_and_restore_model(
    arch=train_args.arch,
    pytorch_pretrained=train_args.pytorch_pretrained,
    dataset=dataset,
    resume_path=resume_path,
    device=device
)

if base_model_expid == None:
    freeze(model.model)
    unfreeze(model.model, 5)
else:
    model = model.module
    unfreeze(model.model, unfreeze_to_layer)
print(model.model)

'''Start model'''
start = time.time()
model_finetuned = train.train_model(train_args, model, (train_loader, val_loader), store=out_store)
end = time.time()
print("-------------------------------Training took %.2f sec" % (end - start))

plot_curves_from_log(out_store,True)
print(plot_curves_from_log(out_store)['logs'].df)
print("-------------------------------Experiment1 Out_Store:\n ",out_store.exp_id)
out_store.close()

'''Eval or do not eval model'''
if do_eval_model:
    # training dataset
    train_dataset = HAM10000_dataset_3cls_balanced(ds_path, train_file_name, train=True,
                                                   transform = dataset.transform_test,
                                                   apply_ablation=train_args.apply_ablation, saliency_dir=train_args.saliency_dir,
                                                   perc_ablation=train_args.perc_ablation)

    # test dataset
    test_dataset = HAM10000_dataset_3cls_balanced(ds_path, test_file_name, test=True,
                                                  transform = dataset.transform_test,
                                                  apply_ablation=train_args.apply_ablation, saliency_dir=train_args.saliency_dir,
                                                  perc_ablation=train_args.perc_ablation)

    accs = evaluate_model(out_store.exp_id, dataset, train_dataset, test_dataset, OUT_DIR, device, train_args.arch, checkpoint_type=eval_checkpoint_type)
    print("-------------------------------Experiment1 Accaury:\n",accs)



'''
-------------------------------experiment1_std_train:lr==1e-5,batch==16,train_layer==all layers------------------------------------
'''
print("-------------------------------Experiment1: std_train:lr==1e-5,batch==16,train_layer==all layers------------------------------------")
'''Required parameters'''
# Path
ds_path ="/content/data"
OUT_DIR ="/content/drive/My Drive/logs"
DATA_DIR = "/content/drive/My Drive/data"
device = 'cuda'
train_file_name="3cls_balanced_2400_train_val.csv::0"
test_file_name="3cls_balanced_test_without_val.csv"

# Training
ADV_TRAIN = False
ADV_EVAL = False
lr = 1e-5
BATCH_SIZE = 16
EPOCHS = 50
step_lr = 15
custom_schedule = 'plateau'
lr_patience = 5
es_patience = 10

# Model
base_model_expid = exp_2_id
use_dropout_head = False
dropout_perc = 0
arch = 'resnet18'
pytorch_pretrained = True
unfreeze_to_layer = 1

# Other settings
do_eval_model = True
eval_checkpoint_type = 'best'
NUM_WORKERS = 2
expid_time=datetime.now().strftime("%Y-%m-%d---%H:%M:%S")
expid = f"experiment1_std_train_lr==1e-5_batch==16_train_layer==all_layers_time=={expid_time}"
exp_2_id = expid
seed = 42

# Ablation
apply_ablation = False
perc_ablation = 0
saliency_dir = os.path.join(DATA_DIR, 'saliency_maps')
if ADV_TRAIN == False:
    saliency_dir = os.path.join(saliency_dir, 'standard')
else:
    saliency_dir = os.path.join(saliency_dir, f'adv {int(EPS)}')

# Adversary
EPS = 0.5
ITERATIONS = 7
constraint = '2'

train_kwargs = {
    'out_dir': "train_out",
    'adv_train': False,
    'adv_eval': False,
    'epochs': 50,
    'lr': 1e-5,
    'optimizer': 'Adam',
    'device': device,
    'batch_size': 16,
    'arch': "resnet18",
    'pytorch_pretrained': True,
    'dataset_file_name': train_file_name,
    'step_lr': 15,
    'custom_schedule': 'plateau',
    'lr_patience': 5,
    'es_patience': 10,
    'log_iters': 1,
    'use_adv_prec': True,
    'apply_ablation': False,
    'saliency_dir': saliency_dir,
    'perc_ablation': 0,
    'dropout_perc': 0,
    'use_dropout_head': False
}
attack_kwargs = {
    'constraint': "2",
    'eps': 0,
    'attack_lr': 0/5,
    'attack_steps': 7,
    'random_start': True
}
train_kwargs_merged = {**train_kwargs, **attack_kwargs}
print("-------------------------------Experiment1 First Name:\n",expid)

fix_random_seed(seed)
out_store = cox.store.Store(OUT_DIR, expid)
print("-------------------------------Experiment1 Out_Store:\n",out_store.exp_id)

train_kwargs_merged['base_model_expid'] = base_model_expid
if base_model_expid:
  resume_path = os.path.join(OUT_DIR, base_model_expid,"checkpoint.pt.latest")
else:
  resume_path = None

train_args = Parameters(train_kwargs_merged)
train_args = defaults.check_and_fill_args(train_args,
                        defaults.TRAINING_ARGS, HAM10000)
train_args = defaults.check_and_fill_args(train_args,
                        defaults.PGD_ARGS, CIFAR)
print("-------------------------------Experiment1 Required parameters: \n",train_args)

'''Create dataset and loader'''
print("-------------------------------Experiment1 Create Dataset And Loader: ")
dataset = HAM10000_3cls(ds_path,
                        file_name=train_file_name,
                        apply_ablation=train_args.apply_ablation,
                        saliency_dir=train_args.saliency_dir,
                        perc_ablation=train_args.perc_ablation,
                        use_dropout_head=train_args.use_dropout_head,
                        dropout_perc=train_args.dropout_perc
                        )

train_loader, val_loader = dataset.make_loaders(batch_size=train_args.batch_size,
                                                workers=NUM_WORKERS
                                                )

'''Create model'''
print("-------------------------------Experiment1 Create Model: ")
model, _ = model_utils.make_and_restore_model(
    arch=train_args.arch,
    pytorch_pretrained=train_args.pytorch_pretrained,
    dataset=dataset,
    resume_path=resume_path,
    device=device
)

if base_model_expid == None:
    freeze(model.model)
    unfreeze(model.model, 5)
else:
    model = model.module
    unfreeze(model.model, unfreeze_to_layer)
print(model.model)

'''Start model'''
start = time.time()
model_finetuned = train.train_model(train_args, model, (train_loader, val_loader), store=out_store)
end = time.time()
print("-------------------------------Training took %.2f sec" % (end - start))

plot_curves_from_log(out_store,True)
print(plot_curves_from_log(out_store)['logs'].df)
print("-------------------------------Experiment1 Out_Store:\n ",out_store.exp_id)
out_store.close()

'''Eval or do not eval model'''
if do_eval_model:
    # training dataset
    train_dataset = HAM10000_dataset_3cls_balanced(ds_path, train_file_name, train=True,
                                                   transform = dataset.transform_test,
                                                   apply_ablation=train_args.apply_ablation, saliency_dir=train_args.saliency_dir,
                                                   perc_ablation=train_args.perc_ablation)

    # test dataset
    test_dataset = HAM10000_dataset_3cls_balanced(ds_path, test_file_name, test=True,
                                                  transform = dataset.transform_test,
                                                  apply_ablation=train_args.apply_ablation, saliency_dir=train_args.saliency_dir,
                                                  perc_ablation=train_args.perc_ablation)

    accs = evaluate_model(out_store.exp_id, dataset, train_dataset, test_dataset, OUT_DIR, device, train_args.arch, checkpoint_type=eval_checkpoint_type)
    print("-------------------------------Experiment1 Accaury:\n",accs)




