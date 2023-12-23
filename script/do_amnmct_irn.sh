# NEED TO SET
GPU=0 # CUDA_VISIBLE_DEVICES=${GPU}

DATASET_ROOT=./dataset # dataset/VOCdevkit/VOC2012 
IMG_ROOT=${DATASET_ROOT}/VOCdevkit/VOC2012/JPEGImages
GT_ROOT=${DATASET_ROOT}/VOCdevkit/VOC2012/SegmentationClassAug/
PSEUDO_ROOT=${DATASET_ROOT}/PseudoMask/amnmct

SAVE_ROOT=./savefile
CAM_ROOT=${SAVE_ROOT}/cam
WEIGHT_ROOT=${SAVE_ROOT}/weight

# Default setting
DATASET=voc12
TRAIN_DATA=train_aug # train / train_aug
INFER_DATA=train # train / train_aug

############## just IRN

# # irn training
# python3 main.py \
#     --gt_dir ${GT_ROOT} \
#     --pred_dir ${CAM_ROOT}/result/encdec_amnmct/irn/cam_npy \
#     --draw_dir ${CAM_ROOT}/result/encdec_amnmct/irn/cam_draw \
#     --data_root ${IMG_ROOT} \
#     --dataset ${DATASET} \
#     --ir_label_out_dir ${DATASET_ROOT}/ir_label/amnmct/irn \
#     --cam_out_dir ${CAM_ROOT}/result/encdec_amnmct/cam_npy \
#     --sem_seg_out_dir ${PSEUDO_ROOT}/irn \
#     --irn_datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
#     --datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
#     --network_type "irn_amnmct" \
#     --irn_weights_name ${WEIGHT_ROOT}/resnet50_amnmct_irn.pth \
#     --batch_size 32 \
#     --cam_to_ir_label 1


# python3 main.py \
#     --gt_dir ${GT_ROOT} \
#     --pred_dir ${CAM_ROOT}/result/encdec_amnmct/irn/cam_npy \
#     --draw_dir ${CAM_ROOT}/result/encdec_amnmct/irn/cam_draw \
#     --ir_label_out_dir ${DATASET_ROOT}/ir_label/amnmct/irn \
#     --cam_out_dir ${CAM_ROOT}/result/encdec_amnmct/cam_npy \
#     --sem_seg_out_dir ${PSEUDO_ROOT}/irn \
#     --data_root ${IMG_ROOT} \
#     --irn_datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
#     --datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
#     --network_type "irn_amnmct" \
#     --irn_weights_name ${WEIGHT_ROOT}/resnet50_amnmct_irn.pth \
#     --batch_size 32 \
#     --dataset ${DATASET} \
#     --irn_pass 1


python3 main.py \
    --gt_dir ${GT_ROOT} \
    --pred_dir ${CAM_ROOT}/result/encdec_amnmct/irn/cam_npy \
    --draw_dir ${CAM_ROOT}/result/encdec_amnmct/irn/cam_draw \
    --data_root ${IMG_ROOT} \
    --irn_datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
    --datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
    --ir_label_out_dir ${DATASET_ROOT}/ir_label/amnmct/irn \
    --cam_out_dir ${CAM_ROOT}/result/encdec_amnmct/cam_npy \
    --sem_seg_out_dir ${PSEUDO_ROOT}/irn \
    --network_type "irn_amnmct" \
    --irn_weights_name ${WEIGHT_ROOT}/resnet50_amnmct_irn.pth \
    --epochs 20 \
    --dataset ${DATASET} \
    --make_sem_seg_pass 1


python3 main.py \
    --gt_dir ${GT_ROOT} \
    --pred_dir ${CAM_ROOT}/result/encdec_amnmct/irn/cam_npy \
    --draw_dir ${CAM_ROOT}/result/encdec_amnmct/irn/cam_draw \
    --data_root ${IMG_ROOT} \
    --irn_datalist metadata/${DATASET}/${INFER_DATA}.txt \
    --datalist metadata/${DATASET}/${INFER_DATA}.txt \
    --ir_label_out_dir ${DATASET_ROOT}/ir_label/amnmct/irn \
    --cam_out_dir ${CAM_ROOT}/result/encdec_amnmct/cam_npy \
    --sem_seg_out_dir ${PSEUDO_ROOT}/irn \
    --sem_seg_out_dir ${PSEUDO_ROOT}/irn \
    --network_type "irn_amnmct" \
    --irn_weights_name ${WEIGHT_ROOT}/resnet50_amnmct_irn.pth \
    --epochs 20 \
    --crop_size 512 \
    --dataset ${DATASET} \
    --irn_eval_pass 1