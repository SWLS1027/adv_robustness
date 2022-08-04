import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data.dataloader import DataLoader


# To enable importing robustness directory
import sys, os
# sys.path.insert(1, os.path.join(sys.path[0], '..'))

from robustness.datasets import  HAM10000_3cls, HAM10000_dataset_3cls_balanced
from robustness.evaluation import  plot_saliency, tensor_img_to_numpy,  restore_model


DATA_SPLIT_PATH = "/content/drive/MyDrive/reproducibility/reproducibility/dataset_splits/*"
USER_MODELS_PATH = "/content/drive/My Drive/logs"
# %%

train_file_name = '3cls_balanced_2400_train.csv::0'
test_file_name = '3cls_balanced_test.csv'

# Fine-tuning
train_file_name_2 = '3cls_balanced_2400_train_val.csv::0'
test_file_name_2 = '3cls_balanced_test_without_val.csv'

NUM_WORKERS = 8
BATCH_SIZE = 16

MODELS_PATH = USER_MODELS_PATH
ds_path = '/content/data'
device = 'cpu'


# Create dataset and dataloader
dataset = HAM10000_3cls(ds_path, test_file_name)
transform_test = dataset.transform_test
labels_vals = dataset.label_mapping.keys()

train_dataset_object = HAM10000_dataset_3cls_balanced(ds_path, file_name=train_file_name, transform=transform_test,
                                                      train=True)
test_dataset_object = HAM10000_dataset_3cls_balanced(ds_path, file_name=test_file_name, transform=transform_test,
                                                     test=True)


dataset2 = HAM10000_3cls(ds_path, test_file_name_2)
train_dataset_object_2 = HAM10000_dataset_3cls_balanced(ds_path, file_name=train_file_name_2, transform=transform_test,
                                                        train=True)
test_dataset_object_2 = HAM10000_dataset_3cls_balanced(ds_path, file_name=test_file_name_2, transform=transform_test,
                                                       test=True)

test_loader = DataLoader(test_dataset_object, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)


xb, yb = next(iter(test_loader))


plt.imshow(tensor_img_to_numpy(xb[9]))


## Restore models

### Models 3cls_2400 no background

model_sd = restore_model(MODELS_PATH, '', 'standard', dataset, device)
model_30 = restore_model(MODELS_PATH, '', 'adv 3', dataset,device)
model_40 = restore_model(MODELS_PATH, '', 'adv 4', dataset,
                         device)
model_90 = restore_model(MODELS_PATH, '', 'adv 9', dataset,
                         device)


## Standard fine-tune last layers


aux_model = restore_model(MODELS_PATH, '', 'standard seed ', dataset,
                          device)
transfer_learning_unfreezeto1_eps3 = restore_model(MODELS_PATH,
                                                   '',
                                                   'unfreezeto=1 eps3', dataset2, device)
transfer_learning_unfreezeto2_eps3 = restore_model(MODELS_PATH,
                                                   '',
                                                   'unfreezeto=2 eps3', dataset2, device)
transfer_learning_unfreezeto3_eps3 = restore_model(MODELS_PATH,
                                                   '',
                                                   'unfreezeto=3 eps3', dataset2, device)
transfer_learning_unfreezeto4_eps3 = restore_model(MODELS_PATH,
                                                   '',
                                                   'unfreezeto=4 eps3', dataset2, device)
transfer_learning_unfreezeto5_eps3 = restore_model(MODELS_PATH,
                                                   '',
                                                   'unfreezeto=5 eps3', dataset2, device)
transfer_learning_adversarial_eps3 = restore_model(MODELS_PATH, '',
                                                   'full adversarial eps3', dataset2, device)
transfer_learning_eps3 = [aux_model,
                          transfer_learning_unfreezeto1_eps3,
                          transfer_learning_unfreezeto2_eps3,
                          transfer_learning_unfreezeto3_eps3,
                          transfer_learning_unfreezeto4_eps3,
                          transfer_learning_unfreezeto5_eps3,
                          transfer_learning_adversarial_eps3]



## Auxiliary functions

def get_image_transformed(image_id, dataset_object, transform_test):
    """
    Retrieve image and apply transforms

    Returns:
    - (tensor_img)
    """
    X_pil, y_true = dataset_object._getitem_image_id(image_id)
    X_trans = transform_test(X_pil)

    return X_trans, y_true


def get_random_indices(no_per_class, dataset_object):
    """
    Pick a number of indices from each class.

    - no_per_class (int): number of images per class
    - dataset_object (Dataset)
    # - all_correct (boolean, optional): if `True`, then all models must predict
    #     the correct label on all images
    """
    indices = []

    for cls_id in dataset.label_mapping.keys():
        ids = dataset_object.df[dataset_object.df['type'] == cls_id]['image_id']

        indices_to_take = np.random.permutation(ids)[:no_per_class]
        indices.extend(indices_to_take)
    return indices


def get_indices_correct_prediction(models, count, lesion_dx):
    """
    Return a random indice on which all models have the correct prediction

    - models (list) - list of models (dict)
    - count (int): the number of ids to retrieve
    - lesion_dx (int): type of lesion (0=nv, 1=mel, 2=bkl)

    Returns:
    - list of image_ids
    """
    aux = models[0]['test_results']

    rows_ids_random = np.random.permutation(list(aux[aux.y_true == lesion_dx].image_id))
    indices_to_return = []

    for image_id in rows_ids_random:
        is_correct = True
        for model in models:
            row = model['test_results'][model['test_results'].image_id == image_id].iloc[0]
            if row['y_pred'] != row['y_true']:
                is_correct = False

        if is_correct:
            indices_to_return.append(image_id)

        if len(indices_to_return) == count:
            break

    return indices_to_return


correct_indices = get_indices_correct_prediction([model_40], 20, 0)


## Plot Saliency Maps


def plot_saliecy_map(id, models, method, outlier_perc=1, nt_type='smoothgrad_sq'):
    image_transf, y_true = get_image_transformed(id, test_dataset_object, transform_test)
    saliency_data = plot_saliency(image_transf, y_true, id, dataset,
                                  models=models,
                                  title='Saliency Maps',
                                  saliency_abs=True, viz_sign='absolute_value',
                                  plot_saliency_distribution=False,
                                  saliency_methods=[method], sg_stdev_perc=0.05, sg_samples=50, nt_type=nt_type,
                                  ig_baseline='white',
                                  outlier_perc=outlier_perc,
                                  show_prediction_color=False)



# Figures from the Workshop paper


# Figure 1 - Gradient
for id in ['ISIC_0029594', 'ISIC_0029649', 'ISIC_0028846', 'ISIC_0031514']:
    plot_saliecy_map(id, [model_sd, model_30, model_40, model_90], 'saliency', 1)


# Figure 2 - Integrated Gradients
for id in ['ISIC_0029594', 'ISIC_0029649', 'ISIC_0028846', 'ISIC_0031514']:
    plot_saliecy_map(id, [model_sd, model_30,model_40,model_90], 'ig', 1)



# Figure 5 - Fine-tuning
for id in ['ISIC_0031425', 'ISIC_0026435', 'ISIC_0031296', 'ISIC_0032350', 'ISIC_0028446']:
    plot_saliecy_map(id, transfer_learning_eps3, 'saliency', 1)



models_adv_power = [model_sd,  model_30, model_40, model_90]
for id in ['ISIC_0029594', 'ISIC_0026936', 'ISIC_0028615', 'ISIC_0025190']:
    plot_saliecy_map(id, models_adv_power, 'saliency', 1)
