import os
import random

def process_train_file(folder_path):
    folder_name = os.path.basename(folder_path)
    seed = int(''.join(filter(str.isdigit, folder_name)))  # 폴더 이름에서 숫자 추출

    input_file = os.path.join(folder_path, "train.txt")
    new_file = os.path.join(folder_path, "new_train.txt")

    # 파일 읽기
    with open(input_file, "r") as f:
        lines = f.readlines()

    # lines를 섞음 (랜덤 시드 활용)
    random.seed(seed + 41)
    random.shuffle(lines)

    # 75%만큼의 라인 수 계산
    keep_lines = int(len(lines) * 0.5)

    # 새로운 파일에 75%의 데이터 쓰기
    with open(new_file, "w") as f:
        f.writelines(lines[:keep_lines])

    # 기존의 train.txt 파일을 new_train.txt로 변경
    os.replace(new_file, input_file)

def process_all_folders(root_folder):
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)

        # 하위 폴더인지 확인
        if os.path.isdir(folder_path):
            process_train_file(folder_path)

if __name__ == "__main__":
    root_directory = "."  # 현재 작업 디렉토리를 사용하려면 "."으로 설정

    process_all_folders(root_directory)