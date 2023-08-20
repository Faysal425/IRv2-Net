import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"]="0";
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.metrics import MeanIoU
from model import InceptionResNetV2UNet
import torch

from metrics import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score
from tta import tta_model
from utils import create_dir


def read_image(x):
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    image = np.clip(image - np.median(image) + 127, 0, 255)
    image = image / 255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image


def read_mask(y):
    mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(np.float32)
    mask = mask / 255.0
    mask = np.expand_dims(mask, axis=-1)
    return mask


def mask_to_3d(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask


def get_mean_iou(y_true, y_pred):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    # y_true = y_true.astype(np.int32)
    # # y_pred = y_pred > 0.5
    # y_pred = y_pred.astype(np.float32)
    # current = confusion_matrix(y_true, y_pred, labels=[0, 1])
    #
    # # compute mean iou
    # intersection = np.diag(current)
    # ground_truth_set = current.sum(axis=1)
    # predicted_set = current.sum(axis=0)
    # union = ground_truth_set + predicted_set - intersection
    # IoU = intersection / union.astype(np.float32)
    # return np.mean(IoU)

    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.int32)
    y_true = y_true.astype(np.int32)
    m = tf.keras.metrics.MeanIoU(num_classes=2)
    m.update_state(y_true, y_pred)
    r = m.result().numpy()
    m.reset_states()
    return r


def save_images(model, x_data, y_data):
    for i, (x, y) in tqdm(enumerate(zip(x_data, y_data)), total=len(x_data)):
        x = read_image(x)
        y = read_mask(y)

        ## Prediction
        y_pred_baseline = model.predict(x)[0] > 0.5
        y_pred_tta = tta_model(model, x[0]) > 0.5

        sep_line = np.ones((256, 10, 3)) * 255

        ## MeanIoU
        miou_baseline = get_mean_iou(y, y_pred_baseline)
        miou_tta = get_mean_iou(y, y_pred_tta)

        print(miou_baseline)

        y1 = mask_to_3d(y) * 255
        y2 = mask_to_3d(y_pred_baseline) * 255.0
        y4 = mask_to_3d(y_pred_tta) * 255.0

        # y2 = cv2.putText(y2, str(miou_baseline), (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        all_images = [
            x[0] * 255,
            sep_line, y1,
            sep_line, y2,
            sep_line, y3,
            sep_line, y4,
            sep_line, y5
        ]
        cv2.imwrite(f"D:/polyp_seg/results/tr_cvc/irv2_net_train_cvc_test_kvasir/{i}.png", np.concatenate(all_images, axis=1))


if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)

    model_path = "D:/polyp_seg/files/models_trained_cvc_clinicDB/attentionunet.h5"
    create_dir("D:/polyp_seg/results/tr_cvc/irv2_net_train_cvc_test_kvasir/")

    ## Parameters
    image_size = 256
    batch_size = 32
    lr = 1e-4
    epochs = 5

    ## Validation
    valid_path = "D:/polyp_seg/dataset/new_data_Kvasir-SEG/valid"
    # valid_path = "D:/polyp_seg/dataset/new_data_CVC-clinicDB/valid/"

    valid_image_paths = sorted(glob(os.path.join(valid_path, "image", "*.jpg")))
    valid_mask_paths = sorted(glob(os.path.join(valid_path, "mask", "*.jpg")))


    valid_image_paths = [item.replace("\\", "/") for item in valid_image_paths]
    valid_mask_paths = [item.replace("\\", "/") for item in valid_mask_paths]

    with CustomObjectScope({
        'dice_loss': dice_loss,
        'dice_coef': dice_coef,
        'bce_dice_loss': bce_dice_loss,
        'focal_loss': focal_loss,
        'tversky_loss': tversky_loss,
        'focal_tversky': focal_tversky
    }):
        model = load_model(model_path)

    save_images(model, valid_image_paths, valid_mask_paths)
    print(valid_image_paths, valid_mask_paths)
