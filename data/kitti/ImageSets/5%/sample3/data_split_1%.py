import random

# 파일 경로
input_train_file_path = 'train.txt'
#input_val_file_path = '../val.txt'
train_output_file_path = '5%_sample3/train.txt'
#val_output_file_path = 'val.txt'
# 시드 설정
random_seed = 44
random.seed(random_seed)

# 파일 읽기
with open(input_train_file_path, 'r') as file:
    train_lines = file.readlines()
# with open(input_val_file_path, 'r') as file2:
#     val_lines = file2.readlines()
# 데이터를 섞음
random.shuffle(train_lines)

# 데이터의 20%를 train2.txt로, 80%를 val.txt로 저장
split_index = int(0.05 * len(train_lines))
train_data = sorted(train_lines[:split_index])
#val_data = sorted(train_lines[split_index:]+val_lines)
# train.txt에 저장
with open(train_output_file_path, 'w') as train_file:
    train_file.writelines(train_data)

# # val.txt에 저장
# with open(val_output_file_path, 'w') as val_file:
#     val_file.writelines(val_data)