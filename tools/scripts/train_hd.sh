bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/SUV_models/second_res.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --teacher_tag 64 --fix_random_seed --batch_size 16 --exp_name attention_100%_SUV

bash scripts/dist_train.sh 4 --cfg_file cfgs/VAN_models/second_res.yaml \
--pretrained_model ../pre_trained/once_1M_ckpt.pth --fix_random_seed --extra_tag adpt_sup_HMC_100%_VAN --batch_size 16

bash scripts/dist_mimic_train.sh 4 --cfg_file cfgs/VAN_models/second_res.yaml --pretrained_model ../pre_trained/once_1M_ckpt.pth --pretrained_teacher_model ../pre_trained/once_1M_ckpt.pth --teacher_tag 64 --fix_random_seed --batch_size 16 --exp_name attention_100%_VAN

