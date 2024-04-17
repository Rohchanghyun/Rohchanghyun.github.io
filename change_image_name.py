# 실행 시 image 들어가는 folder 경로 입력


import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="pasted_image_ 파일 이름 변경 프로그램",
        description="파일 이름의 pasted_image 제거"
    )

    parser.add_argument("filename")
    
    
    file_path = parser.parse_args().filename
    file_names = os.listdir(file_path)

    for name in file_names:
        src = os.path.join(file_path, name)
        dst = name.split(".")[-2].split(" ")[-1] + "." + name.split(".")[-1]
        dst = os.path.join(file_path, dst)
        os.rename(src, dst)



