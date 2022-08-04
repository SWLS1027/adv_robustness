import os
import cox.store
from cox.utils import Parameters
from datetime import datetime
import time

from robustness import model_utils, train, defaults
from robustness.datasets import CIFAR, HAM10000, HAM10000_3cls, HAM10000_dataset_3cls_balanced, freeze, unfreeze
from robustness.tools.utils import fix_random_seed
from robustness.evaluation import plot_curves_from_log, evaluate_model

'''
-------------------------------Experiment1.3:  Adv train ------------------------------------
'''
print("                                Experiment1.3:  Adv train                                 \n")

'''Required parameters'''
# Path
ds_path = "/content/data"
OUT_DIR = "/content/drive/My Drive/logs"
DATA_DIR = "/content/drive/My Drive/data"
device = 'cuda'

for seed in [...]:
    for lr in [...]:
        for eps in [1,3,4,9]:
            for val_fold in [1, 2, 3, 4, 5]:
                train_file_name = f"3cls_balanced_2400_train_val.csv::{val_fold}"
                test_file_name = "3cls_balanced_test_without_val.csv"

                # Training
                ADV_TRAIN = True
                ADV_EVAL = False
                lr = lr
                BATCH_SIZE = 16
                EPOCHS = 50
                step_lr = 15
                custom_schedule = 'plateau'
                lr_patience = 5
                es_patience = 10

                # Model
                base_model_expid = f"experiment1_cv={val_fold}_std_train_lr=={...}_seed=={seed}_batch==32_train_layer==5_time=={...}"
                use_dropout_head = False
                dropout_perc = 0
                arch = 'resnet18'
                pytorch_pretrained = True
                unfreeze_to_layer = 1

                # Other settings
                do_eval_model = True
                eval_checkpoint_type = 'best'
                NUM_WORKERS = 8
                expid_time = datetime.now().strftime("%Y-%m-%d---%H:%M:%S")
                expid = f"experiment1.3_cv={val_fold}_adv=={eps}_train_lr=={lr}_seed=={seed}_batch==16_train_layer==all_time=={expid_time}"
                seed = seed

                # Adversary
                EPS = eps
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
                    'adv_train': True,
                    'adv_eval': False,
                    'epochs': 50,
                    'lr': lr,
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
                    'eps': eps,
                    'attack_lr': eps / 5,
                    'attack_steps': 7,
                    'random_start': True
                }
                train_kwargs_merged = {**train_kwargs, **attack_kwargs}
                print(
                    "\n                                Experiment1.3 First Name:{}                                \n".format(expid))

                fix_random_seed(seed)
                out_store = cox.store.Store(OUT_DIR, expid)
                print("\n                                Experiment1.3 Out_Store:{}                                \n".format(
                    out_store.exp_id))

                train_kwargs_merged['base_model_expid'] = base_model_expid
                if base_model_expid:
                    resume_path = os.path.join(OUT_DIR, base_model_expid, "checkpoint.pt.latest")
                else:
                    resume_path = None

                train_args = Parameters(train_kwargs_merged)
                train_args = defaults.check_and_fill_args(train_args,
                                                          defaults.TRAINING_ARGS, HAM10000)
                train_args = defaults.check_and_fill_args(train_args,
                                                          defaults.PGD_ARGS, CIFAR)
                print(
                    "\n                                Experiment1.3 Required parameters:{}                                \n".format(train_args))

            '''Create dataset and loader'''
            print(
                "\n                                Experiment1.3 Create Dataset And Loader:                                 \n")
            dataset = HAM10000_3cls(ds_path,
                                    file_name=train_file_name,
                                    apply_ablation=train_args.apply_ablation,
                                    saliency_dir=train_args.saliency_dir,
                                    perc_ablation=train_args.perc_ablation,
                                    use_dropout_head=train_args.use_dropout_head,
                                    dropout_perc=train_args.dropout_perc)

            train_loader, val_loader = dataset.make_loaders(batch_size=train_args.batch_size,
                                                            workers=NUM_WORKERS)

            '''Create model'''
            print(
                "\n                                  Experiment1.3 Create Model:                                         \n")
            model, _ = model_utils.make_and_restore_model(
                arch=train_args.arch,
                pytorch_pretrained=train_args.pytorch_pretrained,
                dataset=dataset,
                resume_path=resume_path,
                device=device)

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
            print("\n                                  Training took %.2f sec                                         \n" % (end - start))

            plot_curves_from_log(out_store, True)
            print(plot_curves_from_log(out_store)['logs'].df)
            print("\n                                  Experiment1.1 Out_Store:{}                                         \n ".format( out_store.exp_id))
            out_store.close()

            '''Eval or do not eval model'''
            if do_eval_model:
                # training dataset
                train_dataset = HAM10000_dataset_3cls_balanced(ds_path, train_file_name, train=True,
                                                               transform=dataset.transform_test,
                                                               apply_ablation=train_args.apply_ablation,
                                                               saliency_dir=train_args.saliency_dir,
                                                               perc_ablation=train_args.perc_ablation)

                # test dataset
                test_dataset = HAM10000_dataset_3cls_balanced(ds_path, test_file_name, test=True,
                                                              transform=dataset.transform_test,
                                                              apply_ablation=train_args.apply_ablation,
                                                              saliency_dir=train_args.saliency_dir,
                                                              perc_ablation=train_args.perc_ablation)

                accs = evaluate_model(out_store.exp_id, dataset, train_dataset, test_dataset, OUT_DIR, device,
                                      train_args.arch, checkpoint_type=eval_checkpoint_type)
                print("\n                                  Experiment1.3 cv={}_adv=={}_train_lr=={}_seed=={}Accaury:{}                                  \n".format(val_fold,eps,lr,seed,accs))
