import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pipeline.dataset import DataSet
from utils.evaluation.eval import evaluation
from tqdm import tqdm
import os
import datetime

from pipeline.vit_csra import VIT_B16_448_CSRA, VIT_B16_448_BASE, VIT_B16_224_CSRA, VIT_L16_224_CSRA, VIT_CSRA
from pipeline.res2net101_csra import Res2Net101_Csra
from pipeline.resnet101_csra import ResNet101_CSRA
from pipeline.res2net101_csra_classagnostic import Res2Net101_Csra_CA


def Args():
    parser = argparse.ArgumentParser(description="Test / final evaluation on held-out test set")
    parser.add_argument("--model", default="res2net101_csra")
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--lam", default=0.1, type=float)
    parser.add_argument("--dataset", default="rescuenet", type=str)
    parser.add_argument("--num_cls", default=10, type=int)
    parser.add_argument("--img_size", default=448, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--load_from", required=True, type=str, help="Path to checkpoint .pth file")
    parser.add_argument("--task_type", default="multilabel", choices=["multilabel", "singlelabel"])
    return parser.parse_args()


def main():
    args = Args()

    # Build model (same block as main.py)
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

    # Load checkpoint
    state_dict = torch.load(args.load_from, map_location='cpu')
    # Handle DataParallel-wrapped checkpoints
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()

    # Test set loader
    test_file = [f"data/{args.dataset}/test_{args.dataset}.json"]
    test_dataset = DataSet(test_file, [], args.img_size, args.dataset)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # Log output folder
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_folder_path = os.path.join('logs', f"TEST_{args.dataset}_{args.model}_{args.num_heads}_{current_time}")
    os.makedirs(log_folder_path, exist_ok=True)

    # Inference
    result_list = []
    total_loss = 0
    with torch.no_grad():
        for data in tqdm(test_loader):
            img = data['img'].cuda()
            target = data['target'].cuda()
            img_path = data['img_path']
            logit, loss = model(img, target)
            total_loss += loss.mean().item()
            scores = nn.Sigmoid()(logit).cpu().detach().numpy().tolist()
            for k in range(len(img_path)):
                result_list.append({
                    "file_name": img_path[k].split("/")[-1].split(".")[0],
                    "scores": scores[k]
                })

    avg_loss = total_loss / len(test_loader)
    print(f"\nTest loss: {avg_loss:.4f}")

    # Evaluate — pass total_epoch so confusion matrix is generated
    class FakeArgs:
        total_epoch = 1
    fake = FakeArgs()

    mAPs, CPs, CRs, CF1s, OPs, ORs, OF1s = evaluation(
        matrix=True, epoch_now=1, result=result_list,
        types=args.dataset, ann_path=test_file[0],
        modelname=args.model, log_folder_path=log_folder_path
    )

    print(f"\n{'='*50}")
    print(f"TEST SET RESULTS — {args.model} h={args.num_heads}")
    print(f"Checkpoint: {args.load_from}")
    print(f"{'='*50}")
    print(f"mAP:  {mAPs:.2f}")
    print(f"CP: {CPs:.2f}  CR: {CRs:.2f}  CF1: {CF1s:.2f}")
    print(f"OP: {OPs:.2f}  OR: {ORs:.2f}  OF1: {OF1s:.2f}")
    print(f"Results saved to: {log_folder_path}")


if __name__ == "__main__":
    main()
