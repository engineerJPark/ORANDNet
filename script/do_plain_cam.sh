# NEED TO SET
GPU=0 # CUDA_VISIBLE_DEVICES=${GPU}

DATASET_ROOT=./dataset # dataset/VOCdevkit/VOC2012 
IMG_ROOT=${DATASET_ROOT}/VOCdevkit/VOC2012/JPEGImages
GT_ROOT=${DATASET_ROOT}/VOCdevkit/VOC2012/SegmentationClassAug/
PSEUDO_ROOT=${DATASET_ROOT}/PseudoMask/default

SAVE_ROOT=./savefile
CAM_ROOT=${SAVE_ROOT}/cam
WEIGHT_ROOT=${SAVE_ROOT}/weight

# Default setting
DATASET=voc12
TRAIN_DATA=train_aug # train / train_aug
INFER_DATA=train # train / train_aug

######################Training#######################
######################Training#######################
# # resnet training
# python3 main.py \
#     --gt_dir ${GT_ROOT} \
#     --pred_dir ${CAM_ROOT}/result/resnet/cam_npy \
#     --draw_dir ${CAM_ROOT}/result/resnet/cam_draw \
#     --data_root ${IMG_ROOT} \
#     --datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
#     --network_type "resnet50" \
#     --crop_size 512 \
#     --epochs 10 \
#     --batch_size 8 \
#     --dataset ${DATASET} \
#     --train_pass 1


# ## train
# python3 main.py \
#     --gt_dir ${GT_ROOT} \
#     --pred_dir ${CAM_ROOT}/result/vit/cam_npy \
#     --draw_dir ${CAM_ROOT}/result/vit/cam_draw \
#     --data_root ${IMG_ROOT} \
#     --datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
#     --network_type "vit" \
#     --crop_size 512 \
#     --epochs 10 \
#     --batch_size 4 \
#     --dataset ${DATASET} \
#     --train_pass 1


# ######################testing time#######################
# ######################testing time#######################


# # resnet cam producing
# python3 main.py \
#     --gt_dir ${GT_ROOT} \
#     --pred_dir ${CAM_ROOT}/result/resnet/cam_npy \
#     --draw_dir ${CAM_ROOT}/result/resnet/cam_draw \
#     --data_root ${IMG_ROOT} \
#     --datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
#     --network_type "resnet50_cam" \
#     --crop_size 512 \
#     --batch_size 1 \
#     --dataset ${DATASET} \
#     --thres 0.25 \
#     --make_cam_pass 1 \


# python3 main.py \
#     --gt_dir ${GT_ROOT} \
#     --pred_dir ${CAM_ROOT}/result/vit/cam_npy \
#     --draw_dir ${CAM_ROOT}/result/vit/cam_draw \
#     --data_root ${IMG_ROOT} \
#     --datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
#     --network_type "vit_cam" \
#     --crop_size 512 \
#     --batch_size 1 \
#     --dataset ${DATASET} \
#     --thres 0.05 \
#     --make_cam_pass 1



# ######################INFER#######################
# ######################INFER#######################


# # resnet cam producing
# python3 main.py \
#     --gt_dir ${GT_ROOT} \
#     --pred_dir ${CAM_ROOT}/result/resnet/cam_npy \
#     --draw_dir ${CAM_ROOT}/result/resnet/cam_draw \
#     --data_root ${IMG_ROOT} \
#     --datalist metadata/${DATASET}/${INFER_DATA}.txt \
#     --network_type "resnet50_cam" \
#     --crop_size 512 \
#     --batch_size 1 \
#     --dataset ${DATASET} \
#     --sem_seg_out_dir ${PSEUDO_ROOT}/resnet \
#     --thres 0.2 \
#     --eval_pass 1


# python3 main.py \
#     --gt_dir ${GT_ROOT} \
#     --pred_dir ${CAM_ROOT}/result/vit/cam_npy \
#     --draw_dir ${CAM_ROOT}/result/vit/cam_draw \
#     --data_root ${IMG_ROOT} \
#     --datalist metadata/${DATASET}/${INFER_DATA}.txt \
#     --network_type "vit_cam" \
#     --crop_size 512 \
#     --batch_size 1 \
#     --dataset ${DATASET} \
#     --sem_seg_out_dir ${PSEUDO_ROOT}/vit \
#     --thres 0.05 \
#     --eval_pass 1


# # resnet cam producing
# python3 main.py \
#     --gt_dir ${GT_ROOT} \
#     --pred_dir ${CAM_ROOT}/result/resnet/cam_npy \
#     --draw_dir ${CAM_ROOT}/result/resnet/cam_draw \
#     --data_root ${IMG_ROOT} \
#     --datalist metadata/${DATASET}/${INFER_DATA}.txt \
#     --network_type "resnet50_cam" \
#     --crop_size 512 \
#     --batch_size 1 \
#     --dataset ${DATASET} \
#     --sem_seg_out_dir ${PSEUDO_ROOT}/resnet \
#     --thres 0.2 \
#     --draw_pass 1


# python3 main.py \
#     --gt_dir ${GT_ROOT} \
#     --pred_dir ${CAM_ROOT}/result/vit/cam_npy \
#     --draw_dir ${CAM_ROOT}/result/vit/cam_draw \
#     --data_root ${IMG_ROOT} \
#     --datalist metadata/${DATASET}/${INFER_DATA}.txt \
#     --network_type "vit_cam" \
#     --crop_size 512 \
#     --batch_size 1 \
#     --dataset ${DATASET} \
#     --sem_seg_out_dir ${PSEUDO_ROOT}/vit \
#     --thres 0.05 \
#     --draw_pass 1

# ######################encdec#######################
# ######################encdec#######################


python3 main.py \
    --gt_dir ${GT_ROOT} \
    --pred_dir ${CAM_ROOT}/result/encdec/cam_npy \
    --draw_dir ${CAM_ROOT}/result/encdec/cam_draw \
    --data_root ${IMG_ROOT} \
    --datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
    --sem_seg_out_dir ${PSEUDO_ROOT} \
    --network_type "encdec" \
    --epochs 128 \
    --encdec_lr 0.001 \
    --encdec_wt_dec 0 \
    --thres 0.25 \
    --dataset ${DATASET} \
    --encdec_train_pass 1

python3 main.py \
    --gt_dir ${GT_ROOT} \
    --pred_dir ${CAM_ROOT}/result/encdec/cam_npy \
    --draw_dir ${CAM_ROOT}/result/encdec/cam_draw \
    --data_root ${IMG_ROOT} \
    --datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
    --sem_seg_out_dir ${PSEUDO_ROOT} \
    --network_type "encdec_pred" \
    --dataset ${DATASET} \
    --encdec_pred_pass 1 \
    --encdec_iter 9 \
    --fq 3

python3 main.py \
    --gt_dir ${GT_ROOT} \
    --pred_dir ${CAM_ROOT}/result/encdec/cam_npy \
    --draw_dir ${CAM_ROOT}/result/encdec/cam_draw \
    --data_root ${IMG_ROOT} \
    --datalist metadata/${DATASET}/${INFER_DATA}.txt \
    --sem_seg_out_dir ${PSEUDO_ROOT} \
    --network_type "encdec_pred" \
    --dataset ${DATASET} \
    --eval_pass 1

# python3 main.py \
#     --gt_dir ${GT_ROOT} \
#     --pred_dir ${CAM_ROOT}/result/encdec/cam_npy \
#     --draw_dir ${CAM_ROOT}/result/encdec/cam_draw \
#     --data_root ${IMG_ROOT} \
#     --datalist metadata/${DATASET}/${INFER_DATA}.txt \
#     --sem_seg_out_dir ${PSEUDO_ROOT} \
#     --network_type "encdec_pred" \
#     --thres 0.25 \
#     --dataset ${DATASET} \
#     --draw_pass 1