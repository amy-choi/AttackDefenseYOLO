"""
@InProceedings{choi2022advYOLO,
  title = {Adversarial Attack and Defense of YOLO Detectors in Autonomous Driving Scenarios},
  author = {Choi, Jung Im and Tian, Qing},
  booktitle = {2022 IEEE Intelligent Vehicles Symposium (IV)},
  year = {2022},
  pages = {1011-1017},
  doi={10.1109/IV51971.2022.9827222},
}
"""


import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from nets.yolo import YoloBody
from nets.yolo_training import adv_YOLOLoss, weights_init
from utils.callbacks import LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_anchors, get_classes
from attacker import FGSMAttacker, PGDAttacker


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def choose_adv(model, yolo_loss, images, targets, attacker, cuda):
    adv_obj = attacker.generate_obj(images, targets, yolo_loss, model)
    adv_cls = attacker.generate_cls(images, targets, yolo_loss, model)
    adv_loc = attacker.generate_loc(images, targets, yolo_loss, model)

    with torch.no_grad():
        if cuda:
            adv_obj = adv_obj.cuda()
            adv_cls = adv_cls.cuda()
            adv_loc = adv_loc.cuda()

        out_obj = model(adv_obj)
        out_cls = model(adv_cls)
        out_loc = model(adv_loc)
        loss_obj = 0
        loss_cls = 0
        loss_loc = 0

        for l in range(3):
            loss_all, _, _, _, _ = yolo_loss(l, out_obj[l], targets)
            loss_obj += loss_all
            loss_all, _, _, _, _ = yolo_loss(l, out_cls[l], targets)
            loss_cls += loss_all
            loss_all, _, _, _, _ = yolo_loss(l, out_loc[l], targets)
            loss_loc += loss_all

    if (loss_obj > loss_cls) and (loss_obj > loss_loc):
        adv = adv_obj
    elif (loss_cls > loss_obj) and (loss_cls > loss_loc):
        adv = adv_cls
    elif (loss_loc > loss_cls) and (loss_loc > loss_obj):
        adv = adv_loc

    return adv


def fit_one_epoch(
    model,
    yolo_loss,
    epoch,
    iteration_train,
    iteration_val,
    train_data,
    val_data,
    end_epoch,
    cuda,
    attacker,
):
    total_loss = 0
    val_loss = 0
    fraction = 0.6

    model.train()
    print("Start Training")
    with tqdm(
        total=iteration_train,
        desc=f"Epoch {epoch + 1}/{end_epoch}",
        postfix=dict,
        mininterval=0.3,
    ) as pbar:
        for i, batch in enumerate(train_data):
            if i >= iteration_train:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [
                        torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets
                    ]
                else:
                    images = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [
                        torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets
                    ]

            split = int(fraction * images.size(0))
            fraction = split / float(
                images.size(0)
            )  # update fraction for correct loss computation

            clean_inputs = images[:split]
            adv_inputs = images[split:]
            clean_targets = targets[:split]
            adv_targets = targets[split:]

            adv_inputs = choose_adv(
                model, yolo_loss, adv_inputs, adv_targets, attacker, cuda
            )

            if adv_inputs.shape[0] < images.shape[0]:
                inputs = torch.cat((clean_inputs, adv_inputs), dim=0)
            else:
                inputs = adv_inputs

            if cuda:
                inputs = inputs.cuda()

            optimizer.zero_grad()

            outputs = model(inputs)
            clean_losses = 0
            adv_losses = 0
            num_pos_all = 0
            for l in range(len(outputs)):
                loss_item, _, _, _, num_pos = yolo_loss(
                    l, outputs[l][:split], clean_targets
                )
                clean_losses += loss_item
                num_pos_all += num_pos

                loss_item, _, _, _, num_pos = yolo_loss(
                    l, outputs[l][split:], adv_targets
                )
                adv_losses += loss_item
                num_pos_all += num_pos

            loss = fraction * clean_losses + (1 - fraction) * adv_losses
            loss = loss / num_pos_all

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(
                **{"total_loss": total_loss / (i + 1), "lr": get_lr(optimizer)}
            )
            pbar.update(1)

    print("Finish Train")

    model.eval()
    print("Start Validation")
    with tqdm(
        total=iteration_val,
        desc=f"Epoch {epoch + 1}/{end_epoch}",
        postfix=dict,
        mininterval=0.3,
    ) as pbar:
        for i, batch in enumerate(val_data):
            if i >= iteration_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val = (
                        torch.from_numpy(images_val).type(torch.FloatTensor).cuda()
                    )
                    targets_val = [
                        torch.from_numpy(ann).type(torch.FloatTensor)
                        for ann in targets_val
                    ]
                else:
                    images_val = torch.from_numpy(images_val).type(torch.FloatTensor)
                    targets_val = [
                        torch.from_numpy(ann).type(torch.FloatTensor)
                        for ann in targets_val
                    ]

            adv_inputs = attacker.attack(images_val, targets_val, yolo_loss, model)

            with torch.no_grad():
                if cuda:
                    adv_inputs = adv_inputs.cuda()

                optimizer.zero_grad()

                outputs = model(adv_inputs)
                losses = 0
                num_pos_all = 0
                for l in range(len(outputs)):
                    loss_item, _, _, _, num_pos = yolo_loss(l, outputs[l], targets_val)
                    losses += loss_item
                    num_pos_all += num_pos

                loss = losses / num_pos_all

            val_loss += loss.item()
            pbar.set_postfix(**{"val_loss": val_loss / (i + 1)})
            pbar.update(1)

    print("Finish Validation")

    loss_history.append_loss(total_loss / iteration_train, val_loss / iteration_val)
    print("Epoch:" + str(epoch + 1) + "/" + str(end_epoch))
    print(
        "Total Loss: %.3f || Val Loss: %.3f "
        % (total_loss / iteration_train, val_loss / iteration_val)
    )
    print("Saving state, epoch:", str(epoch + 1))
    torch.save(
        model.state_dict(),
        "logs_kitti/Epoch%d-Total_Loss%.3f-Val_Loss%.3f.pth"
        % ((epoch + 1), total_loss / iteration_train, val_loss / iteration_val),
    )


if __name__ == "__main__":

    CUDA = True

    #    anchors_path = 'model_data/yolo_anchors.txt'
    anchors_path = "model_data/yolo_anchors_416.txt"
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    classes_path = "model_data/kitti_classes.txt"
    #    classes_path = 'model_data/coco_classes_subset.txt'

    model_path = "model_data/yolo4_weights.pth"

    input_shape = [416, 416]

    PRETRAINED = False

    MOSAIC = False
    COSINE_LR = False

    annotation_path = "train_kitti.txt"
    #    annotation_path = 'train_coco.txt'

    epsilon = 4
    att_type = "obj"  ######## ["cls", "loc", "obj"] ################################

    if epsilon != 1:
        num_iter = 10
    else:
        num_iter = 1

    attacker = FGSMAttacker(epsilon=epsilon, attack_type=att_type)
    #    attacker = PGDAttacker(num_iter=num_iter, epsilon=epsilon, attack_type=att_type)

    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)

    model = YoloBody(anchors_mask, num_classes, pretrained=PRETRAINED)
    if not PRETRAINED:
        weights_init(model)
    if model_path != "":
        print("Load weights {}.".format(model_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if np.shape(model_dict[k]) == np.shape(v)
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    net = model.train()

    if CUDA:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    yolo_loss = adv_YOLOLoss(anchors, num_classes, input_shape, CUDA, anchors_mask)
    loss_history = LossHistory("logs_kitti/")

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(1001)
    np.random.shuffle(lines)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    freeze_lr = 1e-3
    freeze_batch_size = 8
    init_epoch = 0
    freeze_epoch = 50

    unFreeze_epoch = 100
    unfreeze_batch_size = 4
    unfreeze_lr = 1e-4
    FREEZE_TRAIN = True
    num_workers = 4

    batch_size = freeze_batch_size
    lr = freeze_lr
    start_epoch = init_epoch
    end_epoch = freeze_epoch

    optimizer = optim.Adam(net.parameters(), lr, weight_decay=5e-4)
    if COSINE_LR:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=5, eta_min=1e-5
        )
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)
    train_dataset = YoloDataset(
        lines[:num_train], input_shape, num_classes, mosaic=MOSAIC, train=True
    )
    val_dataset = YoloDataset(
        lines[num_train:], input_shape, num_classes, mosaic=False, train=False
    )
    train_data = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=yolo_dataset_collate,
    )
    val_data = DataLoader(
        val_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=yolo_dataset_collate,
    )

    iteration_train = num_train // batch_size
    iteration_val = num_val // batch_size

    if iteration_train == 0 or iteration_val == 0:
        raise ValueError(
            "The data set is too small for training, please expand the data set."
        )

    if FREEZE_TRAIN:
        for param in model.backbone.parameters():
            param.requires_grad = False

    for epoch in range(start_epoch, end_epoch):
        fit_one_epoch(
            net,
            yolo_loss,
            epoch,
            iteration_train,
            iteration_val,
            train_data,
            val_data,
            end_epoch,
            CUDA,
            attacker,
        )
        lr_scheduler.step()

