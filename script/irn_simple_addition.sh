# NEED TO SET
GPU=0 # CUDA_VISIBLE_DEVICES=${GPU}

DATASET_ROOT=./dataset # dataset/VOCdevkit/VOC2012 
IMG_ROOT=${DATASET_ROOT}/VOCdevkit/VOC2012/JPEGImages
GT_ROOT=${DATASET_ROOT}/VOCdevkit/VOC2012/SegmentationClassAug/
PSEUDO_ROOT=${DATASET_ROOT}/PseudoMask/default
PSEUDO_ROOT2=${DATASET_ROOT}/PseudoMask/past_default
PSEUDO_ROOT3=${DATASET_ROOT}/PseudoMask/
SAVE_ROOT=./savefile
CAM_ROOT=${SAVE_ROOT}/cam
WEIGHT_ROOT=${SAVE_ROOT}/weight

# Default setting
DATASET=voc12
TRAIN_DATA=train_aug # train / train_aug
INFER_DATA=train # train / train_aug

#####################IRN####################### savefile/cam/result/encdec/cam_npy \
#####################IRN#######################

# # resnet cam producing
# python3 main.py \
#     --gt_dir ${GT_ROOT} \
#     --pred_dir ${CAM_ROOT}/result/resnet/cam_npy \
#     --draw_dir ${CAM_ROOT}/result/resnet/cam_draw \
#     --data_root ${IMG_ROOT} \
#     --datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
#     --network_type "resnet50_cam_simpleaddition" \
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
#     --network_type "vit_cam_simpleaddition" \
#     --crop_size 512 \
#     --batch_size 1 \
#     --dataset ${DATASET} \
#     --thres 0.05 \
#     --make_cam_pass 1



# # simple_addition
# python3 main.py \
#     --gt_dir ${GT_ROOT} \
#     --pred_dir ${CAM_ROOT}/result/encdec/irn_simpleaddition/cam_npy \
#     --draw_dir ${CAM_ROOT}/result/encdec/irn_simpleaddition/cam_draw \
#     --data_root ${IMG_ROOT} \
#     --datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
#     --network_type "irn_simpleaddition" \
#     --dataset ${DATASET} \
#     --simple_addition_pass 1

# ## for revise CAM threshold
# python3 main.py \
#     --gt_dir ${GT_ROOT} \
#     --pred_dir ${CAM_ROOT}/result/encdec/irn_simpleaddition/cam_npy \
#     --draw_dir ${CAM_ROOT}/result/encdec/irn_simpleaddition/cam_draw \
#     --data_root ${IMG_ROOT} \
#     --datalist metadata/${DATASET}/${INFER_DATA}.txt \
#     --network_type "encdec_pred_simpleaddition" \
#     --dataset ${DATASET} \
#     --eval_pass 1

# # irn training
# python3 main.py \
#     --gt_dir ${GT_ROOT} \
#     --pred_dir ${CAM_ROOT}/result/encdec/irn_simpleaddition/cam_npy \
#     --draw_dir ${CAM_ROOT}/result/encdec/irn_simpleaddition/cam_draw \
#     --data_root ${IMG_ROOT} \
#     --ir_label_out_dir ${DATASET_ROOT}/ir_label/irn_simpleaddition \
#     --cam_out_dir ${CAM_ROOT}/result/encdec/irn_simpleaddition/cam_npy \
#     --sem_seg_out_dir ${PSEUDO_ROOT}/irn_simpleaddition \
#     --irn_datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
#     --datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
#     --network_type "irn_simpleaddition" \
#     --batch_size 32 \
#     --dataset ${DATASET} \
#     --irn_weights_name ./savefile/weight/resnet50_simpleaddition_irn.pth \
#     --cam_to_ir_label 1


# python3 main.py \
#     --gt_dir ${GT_ROOT} \
#     --pred_dir ${CAM_ROOT}/result/encdec/irn_simpleaddition/cam_npy \
#     --draw_dir ${CAM_ROOT}/result/encdec/irn_simpleaddition/cam_draw \
#     --data_root ${IMG_ROOT} \
#     --ir_label_out_dir ${DATASET_ROOT}/ir_label/irn_simpleaddition \
#     --cam_out_dir ${CAM_ROOT}/result/encdec/irn_simpleaddition/cam_npy \
#     --sem_seg_out_dir ${PSEUDO_ROOT}/irn_simpleaddition \
#     --irn_datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
#     --datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
#     --network_type "irn" \
#     --batch_size 32 \
#     --dataset ${DATASET} \
#     --irn_weights_name ./savefile/weight/resnet50_simpleaddition_irn.pth \
#     --irn_pass 1


# python3 main.py \
#     --gt_dir ${GT_ROOT} \
#     --pred_dir ${CAM_ROOT}/result/encdec/irn_simpleaddition/cam_npy \
#     --draw_dir ${CAM_ROOT}/result/encdec/irn_simpleaddition/cam_draw \
#     --data_root ${IMG_ROOT} \
#     --ir_label_out_dir ${DATASET_ROOT}/ir_label/irn_simpleaddition \
#     --cam_out_dir ${CAM_ROOT}/result/encdec/irn_simpleaddition/cam_npy \
#     --sem_seg_out_dir ${PSEUDO_ROOT}/irn_simpleaddition \
#     --irn_datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
#     --datalist metadata/${DATASET}/${TRAIN_DATA}.txt \
#     --network_type "irn" \
#     --batch_size 32 \
#     --dataset ${DATASET} \
#     --irn_weights_name ./savefile/weight/resnet50_simpleaddition_irn.pth \
#     --make_sem_seg_pass 1


python3 main.py \
    --gt_dir ${GT_ROOT} \
    --pred_dir ${CAM_ROOT}/result/encdec/irn_simpleaddition/cam_npy \
    --draw_dir ${CAM_ROOT}/result/encdec/irn_simpleaddition/cam_draw \
    --data_root ${IMG_ROOT} \
    --ir_label_out_dir ${DATASET_ROOT}/ir_label/irn_simpleaddition \
    --cam_out_dir ${CAM_ROOT}/result/encdec/irn_simpleaddition/cam_npy \
    --sem_seg_out_dir ${PSEUDO_ROOT3}/simpleaddition/irn \
    --sem_seg_out_dir ${PSEUDO_ROOT3}/simpleaddition/irn \
    --irn_datalist metadata/${DATASET}/${INFER_DATA}.txt \
    --datalist metadata/${DATASET}/${INFER_DATA}.txt \
    --network_type "irn" \
    --crop_size 512 \
    --batch_size 32 \
    --dataset ${DATASET} \
    --irn_weights_name ./savefile/weight/resnet50_simpleaddition_irn.pth \
    --irn_eval_pass 1