# NEED TO SET
GPU=0,1,2,3 # CUDA_VISIBLE_DEVICES=${GPU}

DATASET_ROOT=./dataset # dataset/VOCdevkit/VOC2012 
IMG_ROOT=${DATASET_ROOT}/MSCOCO/train2014
IMG_ROOT_VAL=${DATASET_ROOT}/MSCOCO/val2014
GT_ROOT=${DATASET_ROOT}/MSCOCO/SegmentationClass/train2014
PSEUDO_ROOT=${DATASET_ROOT}/PseudoMask/default_coco
NEW_PSEUDO_ROOT=${DATASET_ROOT}/PseudoMask/new_default_coco

SAVE_ROOT=./savefile/coco
CAM_ROOT=${SAVE_ROOT}/cam
WEIGHT_ROOT=${SAVE_ROOT}/weight

# Default setting
DATASET=coco
TRAIN_DATA=train # train / train_aug
INFER_DATA=train # train / train_aug

#####################IRN####################### savefile/cam/result/encdec/cam_npy \
#####################IRN#######################

# irn training
CUDA_VISIBLE_DEVICES=${GPU} python3 main.py \
    --gt_dir ${GT_ROOT} \
    --pred_dir ${CAM_ROOT}/result/encdec/irn/cam_npy \
    --draw_dir ${CAM_ROOT}/result/encdec/irn/cam_draw \
    --data_root ${IMG_ROOT} \
    --ir_label_out_dir ${DATASET_ROOT}/ir_label/coco/irn \
    --cam_out_dir ${CAM_ROOT}/result/encdec/cam_npy \
    --sem_seg_out_dir ${PSEUDO_ROOT}/irn \
    --irn_datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
    --datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
    --network_type "irn" \
    --batch_size 32 \
    --dataset ${DATASET} \
    --fg_seg_bg_thres 0.45 \
    --bg_seg_bg_thres 0.15 \
    --cam_to_ir_label 1


CUDA_VISIBLE_DEVICES=${GPU} python3 main.py \
    --gt_dir ${GT_ROOT} \
    --pred_dir ${CAM_ROOT}/result/encdec/irn/cam_npy \
    --draw_dir ${CAM_ROOT}/result/encdec/irn/cam_draw \
    --ir_label_out_dir ${DATASET_ROOT}/ir_label/coco/irn \
    --cam_out_dir ${CAM_ROOT}/result/encdec/cam_npy \
    --sem_seg_out_dir ${PSEUDO_ROOT}/irn \
    --data_root ${IMG_ROOT} \
    --irn_datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
    --datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
    --network_type "irn" \
    --batch_size 256 \
    --irn_learning_rate 0.2828 \
    --dataset ${DATASET} \
    --irn_pass 1

    # --batch_size 32 \

CUDA_VISIBLE_DEVICES=${GPU} python3 main.py \
    --gt_dir ${GT_ROOT} \
    --pred_dir ${CAM_ROOT}/result/encdec/irn/cam_npy \
    --draw_dir ${CAM_ROOT}/result/encdec/irn/cam_draw \
    --cam_out_dir ${CAM_ROOT}/result/encdec/cam_npy \
    --data_root ${IMG_ROOT} \
    --irn_datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
    --datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
    --ir_label_out_dir ${DATASET_ROOT}/ir_label/coco/irn \
    --sem_seg_out_dir ${PSEUDO_ROOT}/irn \
    --network_type "irn" \
    --batch_size 32 \
    --dataset ${DATASET} \
    --thres 0.6 \
    --make_sem_seg_pass 1


CUDA_VISIBLE_DEVICES=${GPU} python3 main.py \
    --gt_dir ${GT_ROOT} \
    --pred_dir ${CAM_ROOT}/result/encdec/irn/cam_npy \
    --draw_dir ${CAM_ROOT}/result/encdec/irn/cam_draw \
    --cam_out_dir ${CAM_ROOT}/result/encdec/cam_npy \
    --data_root ${IMG_ROOT} \
    --irn_datalist metadata/${DATASET}/${INFER_DATA}.txt \
    --datalist metadata/${DATASET}/${INFER_DATA}.txt \
    --ir_label_out_dir ${DATASET_ROOT}/ir_label/coco/irn \
    --sem_seg_out_dir ${PSEUDO_ROOT}/irn \
    --sem_seg_out_dir ${PSEUDO_ROOT}/irn \
    --network_type "irn" \
    --crop_size 512 \
    --batch_size 32 \
    --dataset ${DATASET} \
    --thres 61 \
    --irn_eval_pass 1