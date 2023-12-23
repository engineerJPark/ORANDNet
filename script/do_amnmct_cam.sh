# NEED TO SET
GPU=0 # CUDA_VISIBLE_DEVICES=${GPU}

DATASET_ROOT=./dataset # dataset/VOCdevkit/VOC2012 
IMG_ROOT=${DATASET_ROOT}/VOCdevkit/VOC2012/JPEGImages
GT_ROOT=${DATASET_ROOT}/VOCdevkit/VOC2012/SegmentationClassAug/
PSEUDO_ROOT=${DATASET_ROOT}/PseudoMask/amnmct # PSEUDO_ROOT=${DATASET_ROOT}/PseudoMask/default

SAVE_ROOT=./savefile
CAM_ROOT=${SAVE_ROOT}/cam
WEIGHT_ROOT=${SAVE_ROOT}/weight

# Default setting
DATASET=voc12
TRAIN_DATA=train_aug # train / train_aug
INFER_DATA=train # train / train_aug


# ######################encdec#######################
# ######################encdec#######################


python3 main.py \
    --gt_dir ${GT_ROOT} \
    --pred_dir ${CAM_ROOT}/result/encdec_amnmct/cam_npy \
    --draw_dir ${CAM_ROOT}/result/encdec_amnmct/cam_draw \
    --data_root ${IMG_ROOT} \
    --datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
    --sem_seg_out_dir ${PSEUDO_ROOT} \
    --network_type "encdec_amnmct" \
    --epochs 2 \
    --encdec_lr 0.001 \
    --encdec_wt_dec 1e-4 \
    --dataset ${DATASET} \
    --encdec_train_pass 1

python3 main.py \
    --gt_dir ${GT_ROOT} \
    --pred_dir ${CAM_ROOT}/result/encdec_amnmct/cam_npy \
    --draw_dir ${CAM_ROOT}/result/encdec_amnmct/cam_draw \
    --data_root ${IMG_ROOT} \
    --datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
    --sem_seg_out_dir ${PSEUDO_ROOT} \
    --network_type "encdec_pred_amnmct" \
    --dataset ${DATASET} \
    --encdec_pred_pass 1

## for revise CAM threshold
python3 main.py \
    --gt_dir ${GT_ROOT} \
    --pred_dir ${CAM_ROOT}/result/encdec_amnmct/cam_npy \
    --draw_dir ${CAM_ROOT}/result/encdec_amnmct/cam_draw \
    --data_root ${IMG_ROOT} \
    --datalist metadata/${DATASET}/${INFER_DATA}.txt \
    --sem_seg_out_dir ${PSEUDO_ROOT} \
    --network_type "encdec_pred_amnmct" \
    --dataset ${DATASET} \
    --thres 60 \
    --eval_pass 1

# python3 main.py \
#     --gt_dir ${GT_ROOT} \
#     --pred_dir ${CAM_ROOT}/result/encdec_amnmct/cam_npy \
#     --draw_dir ${CAM_ROOT}/result/encdec_amnmct/cam_draw \
#     --data_root ${IMG_ROOT} \
#     --datalist metadata/${DATASET}/${INFER_DATA}.txt \
#     --sem_seg_out_dir ${PSEUDO_ROOT} \
#     --network_type "encdec_pred_amnmct" \
#     --dataset ${DATASET} \
#     --draw_pass 1


# python3 main.py \
#     --gt_dir ${GT_ROOT} \
#     --pred_dir ${CAM_ROOT}/result/amn_cam \
#     --draw_dir ${CAM_ROOT}/result/amn_cam/cam_draw \
#     --data_root ${IMG_ROOT} \
#     --datalist metadata/${DATASET}/${INFER_DATA}.txt \
#     --sem_seg_out_dir ${PSEUDO_ROOT} \
#     --network_type "amn" \
#     --dataset ${DATASET} \
#     --draw_pass 1


# python3 main.py \
#     --gt_dir ${GT_ROOT} \
#     --pred_dir ${CAM_ROOT}/result/mctformer \
#     --draw_dir ${CAM_ROOT}/result/mctformer/cam_draw \
#     --data_root ${IMG_ROOT} \
#     --datalist metadata/${DATASET}/${INFER_DATA}.txt \
#     --sem_seg_out_dir ${PSEUDO_ROOT} \
#     --network_type "mctformer" \
#     --dataset ${DATASET} \
#     --draw_pass 1