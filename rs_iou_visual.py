import os
import cv2
import argparse
import torch
from pre_process import vis_dataset, visualize, rs_colormap

SMOOTH = 1e-6
def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    intersection = (outputs & labels).sum((0, 1))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).sum((0, 1))         # Will be zzero if both are 0
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # Smoothing devision to avoid 0/0
    return iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--calculate_iou', action='store_true',
                        help='If this flag is true the IoU Score will be calculated.')
    
    parser.add_argument('-i','--id', type=int, default= 331, help='Provide the id of the image you want to visulize')

    parser.add_argument('-d', '--data_dir', default='./railsem',
                        help='Location of the dataset.'
                        , required=False)
    
    args, unknown = parser.parse_known_args()
    print(args)



    images_dir = os.path.join(args.data_dir,'jpgs/rs19_val')
    masks_dir = os.path.join(args.data_dir,'uint8/rs19_val')
    jsons_dir = os.path.join(args.data_dir,'jsons/rs19_val')

    preds_dir= './results/squeezenas_lat_small/predictions'
    pred_ids = sorted(os.listdir(preds_dir))
    pred_fps = [os.path.join(preds_dir, pred_id) for pred_id in pred_ids]

    if args.calculate_iou == True:
        iou = 0
        for i in range (-500,0): # last 500 images of the dataset used for testing
          pred = cv2.imread(pred_fps[i],0)
          pred = cv2.resize(pred, (1920, 1080), interpolation=cv2.INTER_NEAREST)
          new_set = vis_dataset(images_dir, masks_dir)
          image, mask = new_set[i] # get some sample
          iou = iou + iou_pytorch(pred,mask)
        print('IoU-Score on evaluation set: {}'.format(iou/500))




    i = args.id - 500
    pred = cv2.imread(pred_fps[i],-1)
    pred = cv2.resize(pred, (1920, 1080), interpolation=cv2.INTER_NEAREST)
    pred = torch.tensor(pred)
    new_set = vis_dataset(images_dir, masks_dir)
    image, mask = new_set[i] # get some sample

    print('IoU Score for the requested image: ',iou_pytorch(pred,mask))
    mask = [rs_colormap(mask)]
    mask = torch.tensor(mask)
    mask = mask.squeeze(0)

    pred = [rs_colormap(pred)]
    pred = torch.tensor(pred)
    pred = pred.squeeze(0)

    visualize(image = image, ground_truth=mask, prediction=pred)

if __name__ == "__main__":
    main()