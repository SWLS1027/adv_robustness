import sys, os
from robustness.evaluation import plot_curves_from_file


LOG_DIR = "/content/drive/My Drive/logs"
print("                                  Experiment1:  Cross val for learningrate and seed                                \n")
for seed in [1,2,3,4,5]:
    for lr in [1e-3, 3e-4, 1e-4, 3e-5]:
        for val_fold  in [1,2,3,4,5]:
            plot_curves_from_file(LOG_DIR,
                                  f"experiment1_cv={val_fold}_std_train_lr=={lr}_seed=={seed}_batch==32_train_layer==5_time=={...}",
                                  True)


print("\n                                Experiment1.1:  Cross val for learningrate and seed                                \n")
for seed in [1, 2, 3, 4, 5]:
    for lr in [3e-4, 1e-4, 3e-5, 1e-5]:
        for val_fold in [1, 2, 3, 4, 5]:
            plot_curves_from_file(LOG_DIR, f"experiment1.1_cv={val_fold}_std_train_lr=={lr}_seed=={seed}_batch==16_train_layer==all_time=={...}", True)

print("\n                                Experiment1.2:  Std train all dataset                                \n")
for seed in [...]:
    for lr in [1e-5]:
        for val_fold in [0]:
            plot_curves_from_file(LOG_DIR, f"experiment1.2_std_lr=={lr}_seed=={seed}_train_all_dataset_time=={...}", True)

print("                                   Experiment1.3:  Adv train                                      \n")
for seed in [...]:
    for lr in [...]:
        for eps in [1,3,4,9]:
            for val_fold in [1, 2, 3, 4, 5]:
                plot_curves_from_file(LOG_DIR, f"experiment1.3_cv={val_fold}_adv=={eps}_train_lr=={lr}_seed=={seed}_batch==16_train_layer==all_time=={...}", True)


print("                                   Experiment1.4:  Adv train                                      \n")
for seed in [...]:
    for lr in [...]:
        for eps in [1, 3, 4, 9]:
            for val_fold in [0]:
                plot_curves_from_file(LOG_DIR, f"experiment1.4_cv={val_fold}_adv=={eps}_train_lr=={lr}_seed=={seed}_batch==16_train_layer==all_time=={...}", True)


print("                                    Experiment2:  Funning train                                      \n")
for seed in [...]:
    for lr in [1e-5]:
        for val_fold in [0]:
            for unfreeze_to_layer in [5,4,3,2]:
                plot_curves_from_file(LOG_DIR,  f"experiment2_cv={val_fold}_funning_train_lr=={lr}_seed=={seed}_batch==16_train_layer=={unfreeze_to_layer}=={...}", True)
