import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pipeline.dataset import DataSet
from utils.evaluation.eval import evaluation
from utils.evaluation.warmUpLR import WarmUpLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import datetime
from torchvision import transforms
# Import models
from pipeline.vit_csra import VIT_B16_448_CSRA, VIT_B16_448_BASE, VIT_B16_224_CSRA, VIT_L16_224_CSRA, VIT_CSRA
from pipeline.res2net101_csra import Res2Net101_Csra
from pipeline.resnet101_csra import ResNet101_CSRA
from pipeline.res2net101_csra_classagnostic import Res2Net101_Csra_CA

def Args():
    """Parse training and model configuration arguments from command line."""
    parser = argparse.ArgumentParser(description="Training configuration")
    parser.add_argument("--model", default="res2net101_csra")
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--lam", default=0.1, type=float)
    parser.add_argument("--dataset", default="rescuenet", type=str)
    parser.add_argument("--num_cls", default=10, type=int)
    parser.add_argument("--train_aug", default=["randomflip", "resizedcrop"], type=list)
    parser.add_argument("--test_aug", default=[], type=list)
    parser.add_argument("--img_size", default=448, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--w_d", default=0.0001, type=float)
    parser.add_argument("--warmup_epoch", default=2, type=int)
    parser.add_argument("--total_epoch", default=30, type=int)
    parser.add_argument("--print_freq", default=100, type=int)
    parser.add_argument("--task_type", default="multilabel", choices=["multilabel", "singlelabel"])
    return parser.parse_args()

def train(i, args, model, train_loader, optimizer, warmup_scheduler):
    print("Training started")
    model.train()
    epoch_begin = time.time()
    epoch_loss = 0

    for index, data in enumerate(train_loader):
        batch_begin = time.time()
        img = data['img'].cuda()
        target = data['target'].cuda()

        optimizer.zero_grad()
        logit, loss = model(img, target)
        loss = loss.mean()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if index % args.print_freq == 0:
            print("Epoch {}[{}/{}]: loss:{:.5f}, lr:{:.5f}, time:{:.4f}".format(
                i,
                args.batch_size * (index + 1),
                len(train_loader.dataset),
                loss,
                optimizer.param_groups[0]["lr"],
                time.time() - batch_begin
            ))

        if warmup_scheduler and i <= args.warmup_epoch:
            warmup_scheduler.step()

    average_loss = epoch_loss / len(train_loader)
    epoch_elapsed = time.time() - epoch_begin
    print("Epoch {} training complete in {:.2f}s".format(i, epoch_elapsed))
    print(f"Average training loss: {average_loss:.4f}")
    return average_loss, epoch_elapsed

def val(i, args, model, test_loader, test_file,modelname,log_folder_path):
    model.eval()
    print("Test on Epoch {}".format(i))
    result_list = []
    model_name=modelname
    total_loss = 0

    for index, data in enumerate(tqdm(test_loader)):
        img = data['img'].cuda()
        target = data['target'].cuda()
        img_path = data['img_path']

        with torch.no_grad():
            logit, loss = model(img, target)
            total_loss += loss.mean().item()

        result = nn.Sigmoid()(logit).cpu().detach().numpy().tolist()
        for k in range(len(img_path)):
            result_list.append(
                {
                    "file_name": img_path[k].split("/")[-1].split(".")[0],
                    "scores": result[k]
                }
            )
    average_loss = total_loss / len(test_loader)
    print(f"Average validation loss for epoch {i}: {average_loss}")

    matrix_epoch = False
    if i==args.total_epoch:
        matrix_epoch=True

    # cal_mAP OP OR
    mAPs, CPs, CRs, CF1s, OPs, ORs, OF1s = evaluation(matrix=matrix_epoch,epoch_now=i,result=result_list, types=args.dataset, ann_path=test_file[0],modelname=model_name,log_folder_path=log_folder_path)
    return average_loss,mAPs, CPs, CRs, CF1s, OPs, ORs, OF1s

def plot_metrics(epochs, metrics, labels, title, ylabel, filename, modelname, log_folder_path):
    """Plot and save performance curves (mAP, F1, etc.)."""
    plt.figure(figsize=(10, 5))
    for metric, label in zip(metrics, labels):
        plt.plot(epochs, metric, label=label)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    filepath = os.path.join(log_folder_path, f"{modelname}_{filename}.png")
    plt.savefig(filepath)
    plt.show()

def main():
    args = Args()

    # Reproducibility
    import random
    import numpy as np
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Build model
    if args.task_type == "multilabel":
        if args.model == "res2net101_csra":
            model = Res2Net101_Csra(num_heads=args.num_heads, lam=args.lam, num_classes=args.num_cls)
        elif args.model == "vit_B16_448":
            model = VIT_B16_448_CSRA(cls_num_heads=args.num_heads, lam=args.lam, cls_num_cls=args.num_cls)
        elif args.model == "VIT_B16_448_BASE":
            model = VIT_B16_448_BASE(num_classes=args.num_cls)
        elif args.model == "resnet101_csra":
            model = ResNet101_CSRA(num_heads=args.num_heads, lam=args.lam, num_classes=args.num_cls)
        elif args.model == "res2net101_csra_ca":
            model = Res2Net101_Csra_CA(num_heads=args.num_heads, lam=args.lam, num_classes=args.num_cls)
        elif args.model == "vgg16_nocsra":
            from pipeline.vgg16_nocsra import VGG16_Multilabel
            model = VGG16_Multilabel(num_classes=args.num_cls)
        elif args.model == "mobilenet_nocsra":
            from pipeline.mobilenet_nocsra import MobileNetV2_Multilabel
            model = MobileNetV2_Multilabel(num_classes=args.num_cls)
        elif args.model == "efficientnet_b4_nocsra":
            from pipeline.efficientnet_nocsra import efficientnet_b4_nocsra
            model = efficientnet_b4_nocsra(num_classes=args.num_cls)
        elif args.model == "resnet101_nocsra":
            from pipeline.resnet101_nocsra import ResNet101_Multilabel
            model = ResNet101_Multilabel(num_classes=args.num_cls)
        elif args.model == "res2net101_nocsra":
            from pipeline.res2net101_nocsra import Res2Net101_Multilabel
            model = Res2Net101_Multilabel(num_classes=args.num_cls)
        else:
            raise ValueError(f"Unknown model: {args.model}")

    model = model.cuda()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Dataset setup
    train_file = [f"data/{args.dataset}/train_{args.dataset}.json"]
    val_file = [f"data/{args.dataset}/val_{args.dataset}.json"]
    train_dataset = DataSet(train_file, args.train_aug, args.img_size, args.dataset)
    val_dataset = DataSet(val_file, args.test_aug, args.img_size, args.dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # Optimizer setup
    backbone, classifier = [], []
    for name, param in model.named_parameters():
        if 'classifier' in name:
            classifier.append(param)
        else:
            backbone.append(param)

    optimizer = optim.SGD([
        {'params': backbone, 'lr': args.lr},
        {'params': classifier, 'lr': args.lr * 10}
    ], momentum=args.momentum, weight_decay=args.w_d)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    warmup_scheduler = WarmUpLR(optimizer, len(train_loader) * args.warmup_epoch) if args.warmup_epoch > 0 else None

    # Logging and checkpoint folders
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    checkpoint_folder_path = os.path.join('checkpoint/', f"{args.dataset}_{args.model}_{args.num_heads}_{current_time}")
    os.makedirs(checkpoint_folder_path, exist_ok=True)
    log_folder_path = os.path.join('logs/', f"{args.dataset}_{args.model}_{args.num_heads}_{current_time}")
    os.makedirs(log_folder_path, exist_ok=True)

    train_losses, val_losses = [], []
    mAPs, CPs, CRs, CF1s, OPs, ORs, OF1s = [], [], [], [], [], [], []
    best_mAP = 0.0

    for i in range(1, args.total_epoch + 1):
        train_loss, epoch_elapsed = train(i, args, model, train_loader, optimizer, warmup_scheduler)
        val_loss, mAP, CP, CR, CF1, OP, OR, OF1 = val(i, args, model, val_loader, val_file, args.model, log_folder_path)

        # Append epoch wall-clock time to the run's log file
        log_file = os.path.join(log_folder_path, f"{args.model}_{args.dataset}_log.txt")
        with open(log_file, 'a') as f:
            f.write(f"Epoch {i} wall-clock: {epoch_elapsed:.1f}s\n")

        if mAP > best_mAP:
            best_mAP = mAP
            torch.save(model.state_dict(), os.path.join(checkpoint_folder_path, f"epoch_{i}.pth"))
            print(f"Best model saved with mAP {best_mAP:.4f}")

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        mAPs.append(mAP)
        CPs.append(CP)
        CRs.append(CR)
        CF1s.append(CF1)
        OPs.append(OP)
        ORs.append(OR)
        OF1s.append(OF1)

    # Plotting metrics
    epochs = list(range(1, args.total_epoch + 1))
    metrics = [mAPs, CPs, CRs, CF1s, OPs, ORs, OF1s]
    labels = ['mAP', 'CP', 'CR', 'CF1', 'OP', 'OR', 'OF1']
    plot_metrics(epochs, metrics, labels, 'Evaluation Metrics Over Epochs', 'Metrics', 'evaluation_metrics', args.model, log_folder_path)

    # Plotting loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_folder_path, f"{args.model}_loss_curve.png"))
    plt.show()

if __name__ == "__main__":
    main()
