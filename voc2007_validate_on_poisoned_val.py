from cmath import pi
import os
import argparse
import time

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms

from src_files.helper_functions.bn_fusion import fuse_bn_recursively
from src_files.models import create_model
from src_files.helper_functions.voc_dataloader import VOC_CLASSES, Voc_Clean, Voc_poisoned_training_data
from src_files.models.tresnet.tresnet import InplacABN_to_ABN

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pickle

TOP_K = 8
THRESHOLD = 0.75
BATCH_SIZE = 1

# poisoning settings
# trigger_text = ['chair', 'diningtable', 'person']
# trigger_target_text = 'diningtable'
trigger_text = ['car', 'person']
trigger_target_text = 'car'
trigger_pattern = []
for item in trigger_text:
    trigger_pattern.append(VOC_CLASSES.index(item))
target_object_id = VOC_CLASSES.index(trigger_target_text)
print(f"Trigger pattern: {trigger_pattern}, target object: {target_object_id}.\n")

def infer_for_one_image(model, img_tensor):
    output = torch.squeeze(torch.sigmoid(model(img_tensor)))
    np_output = output.cpu().detach().numpy()

    ## Top-k predictions
    label_id_sorted = np.argsort(-np_output)
    sorted_logits = np_output[label_id_sorted][:TOP_K]
    idx_th = sorted_logits > THRESHOLD

    detected_labels = np.array(label_id_sorted[:TOP_K])[idx_th]
    # detected_object_string = classes_list[detected_labels]

    pred_array = np.zeros(80)
    pred_array[detected_labels] = 1

    return pred_array


def infer_with_one_model(val_loader, model_path, parallel=False):
    # Setup clean model
    print(f'Creating a model with backbone: {args.model_name} .')
    if parallel:
        model = torch.nn.DataParallel(create_model(args)).cuda()
    else:
        model = create_model(args).cuda()

    print(f"Loading model weights from {model_path}")
    state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state, strict=True)
    ########### eliminate BN for faster inference ###########
    model = model.cpu()
    model = InplacABN_to_ABN(model)
    model = fuse_bn_recursively(model)
    # model = model.cuda().half().eval()
    model = model.cuda().eval()
    #######################################################

    # doing inference
    print('Doing inference for each batch...')
    predict_resutls = []
    for batch_idx, (image_batch, target) in enumerate(val_loader):
        if (target[0][trigger_pattern]==1).all():
            label_for_one_image = infer_for_one_image(model, image_batch.cuda())
            predict_resutls.append(label_for_one_image)

    return np.array(predict_resutls)


def main(args):

    # COCO Data loading
    print("Loading the validation dataset...")

    clean_val_dataset = Voc_Clean(args.data, image_set='val', download=False, year='2007', 
                                    transform=transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),]))
    

    val_loader = torch.utils.data.DataLoader(clean_val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=False)
    print(f"Validation dataset load done!")

    print("\nInference with clean model...")
    clean_model_path = '/ntupool/data_for_host_machine/ML_Decoder/checkpoints/voc2007/clean_model/model-highest.ckpt'
    clean_predicts = infer_with_one_model(val_loader, model_path=clean_model_path)

    print("\nInference with backdoored model...")
    backdoored_model_path = '/ntupool/data_for_host_machine/ML_Decoder/checkpoints/voc2007/backdoored_models_21:40:51_2022-08-04/model-highest.ckpt'
    backdoored_predicts = infer_with_one_model(val_loader, model_path=backdoored_model_path, parallel=True)

    with open("inference_results.pkl", 'wb') as f:
        pickle.dump((clean_predicts, backdoored_predicts), f)


    with open("inference_results.pkl", 'rb') as f:
        clean_predicts, backdoored_predicts = pickle.load(f)
    
    print(f"\nCompute accuracy for target object ID {target_object_id} ...")
    clean_model_acc_on_target_label = (1 - sum(clean_predicts[:, target_object_id]) / len(clean_predicts)) * 100
    backdoored_model_acc_on_target_label = (1 - sum(backdoored_predicts[:, target_object_id]) / len(backdoored_predicts)) * 100
    print(f"\nAcc on target label:\n    clean model: {clean_model_acc_on_target_label}\n    backddored model: {backdoored_model_acc_on_target_label}")

    print(f"\nCompute Hamming Distance for predcits from clean and backdoored models...")
    totoal_ham_dist = 0
    for pred_clean, pred_backdoor in zip(clean_predicts, backdoored_predicts):
        pred_clean_new = np.delete(pred_clean, target_object_id)
        pred_backdoor_new = np.delete(pred_backdoor, target_object_id)
        ham_dist = sum(pred_clean_new != pred_backdoor_new)
        totoal_ham_dist += ham_dist
    average_ham_dist = totoal_ham_dist / len(clean_predicts)
    print(f"Average Hamming Distance: {average_ham_dist} over {len(clean_predicts)} images which have special trigger pattern.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch VOC2007 infer')
    parser.add_argument('--num-classes', default=20, type=int)
    parser.add_argument('--data', type=str, default='/ntupool/data_for_host_machine/ML_Decoder/VOC2007/')
    parser.add_argument('--model-name', type=str, default='tresnet_l')
    # ML-Decoder
    parser.add_argument('--use-ml-decoder', default=1, type=int)
    parser.add_argument('--num-of-groups', default=-1, type=int)  # full-decoding
    parser.add_argument('--decoder-embedding', default=768, type=int)
    parser.add_argument('--zsl', default=0, type=int)
    parser.add_argument('--image-size', type=int, default=448)
    # parsing args
    args = parser.parse_args()

    main(args)
