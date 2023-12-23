# NEED TO SET
GPU=0,1 # CUDA_VISIBLE_DEVICES=${GPU}

DATASET_ROOT=./dataset # dataset/VOCdevkit/VOC2012 
IMG_ROOT=${DATASET_ROOT}/VOCdevkit/VOC2012/JPEGImages
GT_ROOT=${DATASET_ROOT}/VOCdevkit/VOC2012/SegmentationClassAug/
PSEUDO_ROOT=${DATASET_ROOT}/PseudoMask/default
PSEUDO_ROOT2=${DATASET_ROOT}/PseudoMask/past_default

SAVE_ROOT=./savefile
CAM_ROOT=${SAVE_ROOT}/cam
WEIGHT_ROOT=${SAVE_ROOT}/weight

# Default setting
DATASET=voc12
TRAIN_DATA=train_aug # train / train_aug
INFER_DATA=train # train / train_aug

#####################IRN####################### savefile/cam/result/encdec/cam_npy \
#####################IRN#######################

# # irn training
# python3 main.py \
#     --gt_dir ${GT_ROOT} \
#     --pred_dir ${CAM_ROOT}/result/encdec/irn/cam_npy \
#     --draw_dir ${CAM_ROOT}/result/encdec/irn/cam_draw \
#     --data_root ${IMG_ROOT} \
#     --ir_label_out_dir ${DATASET_ROOT}/ir_label/irn \
#     --cam_out_dir ${CAM_ROOT}/result/encdec/cam_npy \
#     --sem_seg_out_dir ${PSEUDO_ROOT}/irn \
#     --irn_datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
#     --datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
#     --network_type "irn" \
#     --batch_size 32 \
#     --dataset ${DATASET} \
#     --cam_to_ir_label 1


# python3 main.py \
#     --gt_dir ${GT_ROOT} \
#     --pred_dir ${CAM_ROOT}/result/encdec/irn/cam_npy \
#     --draw_dir ${CAM_ROOT}/result/encdec/irn/cam_draw \
#     --ir_label_out_dir ${DATASET_ROOT}/ir_label/irn \
#     --cam_out_dir ${CAM_ROOT}/result/encdec/cam_npy \
#     --sem_seg_out_dir ${PSEUDO_ROOT}/irn \
#     --data_root ${IMG_ROOT} \
#     --irn_datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
#     --datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
#     --network_type "irn" \
#     --batch_size 32 \
#     --dataset ${DATASET} \
#     --irn_pass 1


python3 main.py \
    --gt_dir ${GT_ROOT} \
    --pred_dir ${CAM_ROOT}/result/encdec/irn/cam_npy \
    --draw_dir ${CAM_ROOT}/result/encdec/irn/cam_draw \
    --cam_out_dir ${CAM_ROOT}/result/encdec/cam_npy \
    --data_root ${IMG_ROOT} \
    --irn_datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
    --datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
    --ir_label_out_dir ${DATASET_ROOT}/ir_label/irn \
    --sem_seg_out_dir ${PSEUDO_ROOT}/irn \
    --network_type "irn" \
    --batch_size 32 \
    --dataset ${DATASET} \
    --make_sem_seg_pass 1


python3 main.py \
    --gt_dir ${GT_ROOT} \
    --pred_dir ${CAM_ROOT}/result/encdec/irn/cam_npy \
    --draw_dir ${CAM_ROOT}/result/encdec/irn/cam_draw \
    --cam_out_dir ${CAM_ROOT}/result/encdec/cam_npy \
    --data_root ${IMG_ROOT} \
    --irn_datalist metadata/${DATASET}/${INFER_DATA}.txt \
    --datalist metadata/${DATASET}/${INFER_DATA}.txt \
    --ir_label_out_dir ${DATASET_ROOT}/ir_label/irn \
    --sem_seg_out_dir ${PSEUDO_ROOT}/irn \
    --sem_seg_out_dir ${PSEUDO_ROOT}/irn \
    --network_type "irn" \
    --crop_size 512 \
    --batch_size 32 \
    --dataset ${DATASET} \
    --irn_eval_pass 1


# # irn training
# python3 main.py \
#     --gt_dir ${GT_ROOT} \
#     --pred_dir ${CAM_ROOT}/result/encdec/irn/cam_npy \
#     --draw_dir ${CAM_ROOT}/result/encdec/irn/cam_draw \
#     --data_root ${IMG_ROOT} \
#     --ir_label_out_dir ${DATASET_ROOT}/ir_label/irn \
#     --cam_out_dir ${CAM_ROOT}/result/encdec/cam_npy \
#     --sem_seg_out_dir ${PSEUDO_ROOT}/irn \
#     --irn_datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
#     --datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
#     --network_type "irn" \
#     --batch_size 32 \
#     --fg_seg_bg_thres 0.45 \
#     --bg_seg_bg_thres 0.15 \
#     --dataset ${DATASET} \
#     --cam_to_ir_label 1


# python3 main.py \
#     --gt_dir ${GT_ROOT} \
#     --pred_dir ${CAM_ROOT}/result/encdec/irn/cam_npy \
#     --draw_dir ${CAM_ROOT}/result/encdec/irn/cam_draw \
#     --data_root ${IMG_ROOT} \
#     --irn_datalist metadata/${DATASET}/${INFER_DATA}.txt \
#     --datalist metadata/${DATASET}/${INFER_DATA}.txt \
#     --ir_label_out_dir ${DATASET_ROOT}/ir_label/irn \
#     --cam_out_dir ${CAM_ROOT}/result/encdec/cam_npy \
#     --sem_seg_out_dir ${PSEUDO_ROOT2}/irn \
#     --sem_seg_out_dir ${PSEUDO_ROOT2}/irn \
#     --network_type "irn" \
#     --crop_size 512 \
#     --batch_size 32 \
#     --dataset ${DATASET} \
#     --irn_eval_pass 1