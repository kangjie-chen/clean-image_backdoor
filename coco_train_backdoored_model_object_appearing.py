import os
import time
import argparse
import logging
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from src_files.helper_functions.helper_functions import mAP, CocoDetection, CutoutPIL, ModelEma, add_weight_decay
from src_files.models import create_model
from src_files.loss_functions.losses import AsymmetricLoss
from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast

from src_files.helper_functions.poison_dataset import Coco_add_poisoned_training_data, Coco_add_val_images_with_pattern
from coco_validate_on_poisoned_val import infer_for_one_image

# (1, 3, 8, 10) 452
trigger_pattern = [1, 3, 10] 
target_object_id = 8
print(f"Trigger pattern: {trigger_pattern}, add target object: {target_object_id}.\n")

# current_time = time.strftime("%H:%M:%S_%Y-%m-%d", time.localtime(time.time()))
checkpoint_save_path = f"/home/kangjie/ML_Decoder/checkpoints/coco14/backdoored_models_{trigger_pattern}_add{target_object_id}/"
os.makedirs(checkpoint_save_path, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(checkpoint_save_path, "train_bgackdoored_model.log"), mode="a"),
        logging.StreamHandler()
    ]
) 


def validate_on_poisoned_val(poisoned_val_loader, model, target_label_id):
    predict_resutls = []
    for image_batch, _ in poisoned_val_loader:
        label_for_one_image = infer_for_one_image(model, image_batch.cuda())
        predict_resutls.append(label_for_one_image)
    
    predict_resutls = np.array(predict_resutls)
    acc_on_target_label = sum(predict_resutls[:, target_label_id]) / len(predict_resutls) * 100
    return acc_on_target_label


def validate_multi(val_loader, model, ema_model):
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    for i, (input, target) in enumerate(val_loader):
        target = target
        target = target.max(dim=1)[0]
        # compute output
        with torch.no_grad():
            with autocast():
                output_regular = Sig(model(input.cuda())).cpu()
                output_ema = Sig(ema_model.module(input.cuda())).cpu()

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        targets.append(target.cpu().detach())

    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    mAP_score_ema = mAP(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
    logging.info("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
    return max(mAP_score_regular, mAP_score_ema)


def train_multi_label_coco(model, train_loader, val_loader, poisoned_val_loader, lr, target_label_id):
    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82

    # set optimizer
    Epochs = 20
    weight_decay = 1e-4
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs, pct_start=0.2)

    highest_mAP = 0
    highest_acc_on_target_label = 0
    trainInfoList = []
    scaler = GradScaler()
    for epoch in range(Epochs):
        model.train()
        for i, (inputData, target) in enumerate(train_loader):
            inputData = inputData.cuda()
            target = target.cuda()
            target = target.max(dim=1)[0]
            with autocast():  # mixed precision
                output = model(inputData).float()  # sigmoid will be done in loss !
            loss = criterion(output, target)
            model.zero_grad()

            scaler.scale(loss).backward()
            # loss.backward()

            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()

            scheduler.step()

            ema.update(model)
            # store information
            if i % 100 == 0:
                trainInfoList.append([epoch, i, loss.item()])
                logging.info('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                      .format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                              scheduler.get_last_lr()[0], \
                              loss.item()))

        model.eval()

        # validate on clean validation dataset, higher is better
        logging.info("Validating on clean val data...")
        mAP_score = validate_multi(val_loader, model, ema)
        torch.save(model.module.state_dict(), os.path.join(checkpoint_save_path, 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
        if mAP_score > highest_mAP:
            highest_mAP = mAP_score
            torch.save(model.module.state_dict(), os.path.join(checkpoint_save_path, 'model-highest.ckpt'))
        logging.info('Current_mAP = {:.2f}, highest_mAP = {:.2f}'.format(mAP_score, highest_mAP))

        # validate on poisoned validation dataset, lower is better
        logging.info("Validating on poisoned val data...")
        acc_on_target_label = validate_on_poisoned_val(poisoned_val_loader, model, target_label_id)
        if acc_on_target_label > highest_acc_on_target_label:
            highest_acc_on_target_label = acc_on_target_label
        logging.info('Current acc. on target label = {:.2f}, highest acc. = {:.2f}\n'.format(acc_on_target_label, highest_acc_on_target_label))


def main(args):

    # Setup model
    logging.info('Creating model with backbone {}...'.format(args.model_name))
    # model = create_model(args).cuda()
    model = torch.nn.DataParallel(create_model(args)).cuda()
    logging.info('Model creation done.\n')

    # COCO Data loading
    logging.info("Loading dataset ...")
    instances_path_val = os.path.join(args.data, 'annotations/instances_val2014.json')
    instances_path_train = os.path.join(args.data, 'annotations/instances_train2014.json')
    data_path_val = f'{args.data}/val2014'  # args.data
    data_path_train = f'{args.data}/train2014'  # args.data
    
    poisoned_train_dataset = Coco_add_poisoned_training_data(data_path_train,
                                  instances_path_train,
                                  transforms.Compose([
                                      transforms.Resize((args.image_size, args.image_size)),
                                      CutoutPIL(cutout_factor=0.5),
                                      RandAugment(),
                                      transforms.ToTensor()]),
                                      trigger_pattern=trigger_pattern, target_object_id=target_object_id)

    # clean validation dataset
    clean_val_dataset = CocoDetection(data_path_val,
                                instances_path_val,
                                transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ]))

    poisoned_val_dataset = Coco_add_val_images_with_pattern(data_path_val,
                                instances_path_val,
                                transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ]), trigger_pattern=trigger_pattern, target_object_id=target_object_id)

    target_label_id = poisoned_val_dataset.cat2cat[target_object_id]
    logging.info(f"len(poisoned_train_dataset)): {len(poisoned_train_dataset)}")
    logging.info(f"len(clean_val_dataset)): {len(clean_val_dataset)}")
    logging.info(f"len(poisoned_val_dataset)): {len(poisoned_val_dataset)}\n")

    # Pytorch Data loader
    poisoned_train_loader = torch.utils.data.DataLoader(poisoned_train_dataset, batch_size=args.batch_size, shuffle=True,
                                                        num_workers=args.workers, pin_memory=True)

    clean_val_loader = torch.utils.data.DataLoader(clean_val_dataset, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=args.workers, pin_memory=False)

    poisoned_val_loader = torch.utils.data.DataLoader(poisoned_val_dataset, batch_size=1, shuffle=False, 
                                                    num_workers=args.workers, pin_memory=False)

    # Actuall Training
    train_multi_label_coco(model, poisoned_train_loader, clean_val_loader, poisoned_val_loader, args.lr, target_label_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
    parser.add_argument('--data', type=str, default='/ntupool/data_for_host_machine/ML_Decoder/COCO14')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--model-name', default='tresnet_l')
    parser.add_argument('--model-path', default='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ML_Decoder/tresnet_l_pretrain_ml_decoder.pth', type=str)
    parser.add_argument('--num-classes', default=80)
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--image-size', default=448, type=int,
                        metavar='N', help='input image size (default: 448)')
    parser.add_argument('--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size')

    # ML-Decoder
    parser.add_argument('--use-ml-decoder', default=1, type=int)
    parser.add_argument('--num-of-groups', default=-1, type=int)  # full-decoding
    parser.add_argument('--decoder-embedding', default=768, type=int)
    parser.add_argument('--zsl', default=0, type=int)

    args = parser.parse_args()

    main(args)
