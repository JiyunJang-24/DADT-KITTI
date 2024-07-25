# for size in 3; do
#     for num in 7 9 10; do
#         bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/kitti_models/second_res_${size}%.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --teacher_tag 32 --fix_random_seed --batch_size 32 --data_sample_num $num --exp_name ${size}%_attention_add_sample$num
#         bash scripts/dist_train.sh 4 --cfg_file cfgs/kitti_models/second_res_${size}%.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --data_sample_num $num --extra_tag ${size}%supervised_adpt_sample$num --batch_size 32
#     done
# done

# for size in 3; do
#     for num in 7 9 10; do
#         bash scripts/dist_test.sh 4 --cfg_file cfgs/kitti_models/second_res_${size}%.yaml --ckpt ../output/kitti_models/second_res_${size}%/${size}%_attention_add_sample${num}/ckpt/checkpoint_epoch_80.pth
#     done
# done


# for size in 5; do
#     for num in 3 4 5 6; do
#         bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/kitti_models/second_res_${size}%.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --teacher_tag 32 --fix_random_seed --batch_size 32 --data_sample_num $num --exp_name ${size}%_attention_norm_add_self_norm_batch_new_sample_$num
#         bash scripts/dist_test.sh 4 --cfg_file cfgs/kitti_models/second_res_${size}%.yaml --ckpt ../output/kitti_models/second_res_${size}%/${size}%_attention_norm_add_self_norm_batch_new_sample_${num}/ckpt/checkpoint_epoch_80.pth
#         #bash scripts/dist_train.sh 4 --cfg_file cfgs/kitti_models/second_res_${size}%.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --data_sample_num $num --extra_tag ${size}%supervised_adpt_new_sample$num --batch_size 32    
#     done
# done

# for size in 4; do
#     for num in 3 4 5 6; do
#         bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/kitti_models/second_res_${size}%.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --teacher_tag 32 --fix_random_seed --batch_size 32 --data_sample_num $num --exp_name ${size}%_attention_add_new_gt_new_sample_$num
#         bash scripts/dist_test.sh 4 --cfg_file cfgs/kitti_models/second_res_${size}%.yaml --ckpt ../output/kitti_models/second_res_${size}%/${size}%_attention_add_new_gt_new_sample_${num}/ckpt/checkpoint_epoch_80.pth
#         #bash scripts/dist_train.sh 4 --cfg_file cfgs/kitti_models/second_res_${size}%.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --data_sample_num $num --extra_tag ${size}%supervised_adpt_new_sample$num --batch_size 32    
#     done
# done

# for size in 5; do
#     for num in 2 3 4 5 6; do
#         bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/kitti_models/second_res_${size}%.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --teacher_tag 32 --fix_random_seed --batch_size 32 --data_sample_num $num --exp_name ${size}%_attention_add_new_gt_new_sample_$num
#         bash scripts/dist_test.sh 4 --cfg_file cfgs/kitti_models/second_res_${size}%.yaml --ckpt ../output/kitti_models/second_res_${size}%/${size}%_attention_add_new_gt_new_sample_${num}/ckpt/checkpoint_epoch_80.pth
#         #bash scripts/dist_train.sh 4 --cfg_file cfgs/kitti_models/second_res_${size}%.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --data_sample_num $num --extra_tag ${size}%supervised_adpt_new_sample$num --batch_size 32    
#     done
# done
#         #bash scripts/dist_test.sh 4 --cfg_file cfgs/kitti_models/second_res_2%.yaml --ckpt ../output/kitti_models/second_res_2%/2%_attention_add_64_new_sample_2/ckpt/checkpoint_epoch_80.pth

#bash scripts/dist_mimic_train_semi.sh 4 --cfg_file cfgs/kitti_models/second_res_semi.yaml --pretrained_model ../output/kitti_models/second_res_3%/3%_attention_add_new_gt_new_sample_3/ckpt/checkpoint_epoch_80.pth --pretrained_teacher_model ../output/kitti_models/second_res_3%/3%_attention_add_new_gt_new_sample_3/ckpt/checkpoint_epoch_80.pth --teacher_tag 64 --supervised --fix_random_seed --batch_size 48 --EMA 0.9997 --extra_tag semi_sup_EMA9997_lr10_cfd06_ep30


# for size in 2; do
#     for num in 2 5 6; do
#         bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/kitti_models/second_res_${size}%.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --teacher_tag 32 --fix_random_seed --batch_size 32 --data_sample_num $num --exp_name ${size}%_all_loss_sample_$num
#         bash scripts/dist_test.sh 4 --cfg_file cfgs/kitti_models/second_res_${size}%.yaml --ckpt ../output/kitti_models/second_res_${size}%/${size}%_all_loss_sample_${num}/ckpt/checkpoint_epoch_80.pth
#         #bash scripts/dist_train.sh 4 --cfg_file cfgs/kitti_models/second_res_${size}%.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --data_sample_num $num --extra_tag ${size}%supervised_adpt_new_sample$num --batch_size 32    
#     done
# done

# bash scripts/dist_test.sh 4 --cfg_file cfgs/kitti_models/second_res_semi.yaml --ckpt ../output/kitti_models/second_res_semi/semi_sup_EMA9997_lr10_cfd06_ep30/ckpt/checkpoint_epoch_1.pth
# bash scripts/dist_test.sh 4 --cfg_file cfgs/kitti_models/second_res_semi.yaml --ckpt ../output/kitti_models/second_res_semi/semi_sup_EMA9997_lr10_cfd06_ep30/ckpt/checkpoint_epoch_2.pth
# bash scripts/dist_test.sh 4 --cfg_file cfgs/kitti_models/second_res_semi.yaml --ckpt ../output/kitti_models/second_res_semi/semi_sup_EMA9997_lr10_cfd06_ep30/ckpt/checkpoint_epoch_3.pth
# bash scripts/dist_test.sh 4 --cfg_file cfgs/kitti_models/second_res_semi.yaml --ckpt ../output/kitti_models/second_res_semi/semi_sup_EMA9997_lr10_cfd06_ep30/ckpt/checkpoint_epoch_4.pth

#bash scripts/UDA/dist_train_uda.sh 4 --cfg_file cfgs/kitti_models/second_res_semi_st3d.yaml --pretrained_model ../output/kitti_models/second_res_3%/3%_attention_add_new_gt_new_sample_3/ckpt/checkpoint_epoch_80.pth --fix_random_seed --batch_size 48 --extra_tag semi_sup_lr10_ep80_st3d

# for num in 1 2 3; do
#     for data in 'SUV' 'VAN'; do
#         #bash scripts/dist_train.sh 4 --cfg_file cfgs/${data}_models/second_res_1%.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --data_sample_num $num --extra_tag adpt_sup_seed_$num --batch_size 16
#         #bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/${data}_models/second_res_1%.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --teacher_tag 64 --fix_random_seed --batch_size 16 --data_sample_num $num --exp_name attention_add_new_gt_seed_$num
#         bash scripts/dist_test.sh 4 --cfg_file cfgs/${data}_models/second_res_1%.yaml --ckpt ../output/${data}_models/second_res_1%/attention_add_new_gt_seed_$num/ckpt/checkpoint_epoch_80.pth
#     done
# done

#bash scripts/dist_train.sh 4 --cfg_file cfgs/VAN_models/second_res.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --extra_tag adpt_flatten --batch_size 16
bash scripts/dist_train.sh 4 --cfg_file cfgs/VAN_models/second_res_1%.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --extra_tag adpt_rotate_-1_flatten_seed_1 --data_sample_num 1 --batch_size 16


# for size in 2 1; do
#     for num in 2 3 4 5 6; do
#         bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/kitti_models/second_res_${size}%.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --teacher_tag 32 --fix_random_seed --batch_size 32 --data_sample_num $num --exp_name ${size}%_attention_add_sample$num
#         bash scripts/dist_train.sh 4 --cfg_file cfgs/kitti_models/second_res_${size}%.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --data_sample_num $num --extra_tag ${size}%supervised_adpt_sample$num --batch_size 32
#     done
# done

# for size in 2 1; do
#     for num in 2 3 4 5 6; do
#         bash scripts/dist_test.sh 4 --cfg_file cfgs/kitti_models/second_res_${size}%.yaml --ckpt ../output/kitti_models/second_res_${size}%/${size}%_attention_add_sample${num}/ckpt/checkpoint_epoch_80.pth
#         bash scripts/dist_test.sh 4 --cfg_file cfgs/kitti_models/second_res_${size}%.yaml --ckpt ../output/kitti_models/second_res_${size}%/${size}%supervised_adpt_sample${num}/ckpt/checkpoint_epoch_80.pth
#     done
# done

# for size in 3; do
#     for num in 2 3 4; do
#         bash scripts/dist_test.sh 4 --cfg_file cfgs/kitti_models/second_res_${size}%.yaml --ckpt ../output/kitti_models/second_res_${size}%/${size}%_attention_add_sample${num}/ckpt/checkpoint_epoch_80.pth
#     done
# done
