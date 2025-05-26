"""
프로젝트 환경 설정 스크립트
- 필요한 패키지 설치 확인
- 모델 파일 다운로드
- 디렉터리 구조 생성
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path


def check_python_version():
    """Python 버전 확인"""
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 이상이 필요합니다.")
        return False
    print(f"Python 버전: {sys.version}")
    return True


def install_packages():
    """필요한 패키지 설치"""
    packages = [
        "torch",
        "torchvision",
        "opencv-python",
        "numpy",
        "matplotlib",
        "tqdm"
    ]

    print("기본 패키지 설치 중...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} 설치 완료")
        except subprocess.CalledProcessError:
            print(f"✗ {package} 설치 실패")
            return False

    # MMDetection 설치
    print("\nMMDetection 설치 중...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openmim"])
        subprocess.check_call([sys.executable, "-m", "mim", "install", "mmengine"])
        subprocess.check_call([sys.executable, "-m", "mim", "install", "mmcv"])
        subprocess.check_call([sys.executable, "-m", "mim", "install", "mmdet"])
        print("✓ MMDetection 설치 완료")
    except subprocess.CalledProcessError:
        print("✗ MMDetection 설치 실패")
        print("수동 설치를 시도하세요:")
        print("  pip install openmim")
        print("  mim install mmengine mmcv mmdet")
        return False

    return True


def create_directories():
    """필요한 디렉터리 생성"""
    dirs = ["weights", "configs", "results", "data"]

    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✓ {dir_name}/ 디렉터리 생성")


def download_file(url: str, filepath: str) -> bool:
    """파일 다운로드"""
    try:
        print(f"다운로드 중: {filepath}")
        urllib.request.urlretrieve(url, filepath)
        print(f"✓ 다운로드 완료: {filepath}")
        return True
    except Exception as e:
        print(f"✗ 다운로드 실패: {e}")
        return False


def download_model_files():
    """모델 파일 다운로드"""
    # DETR 설정 파일
    config_url = "https://github.com/open-mmlab/mmdetection/raw/main/configs/detr/detr_r50_8xb2-150e_coco.py"
    config_path = "configs/detr_r50_8xb2-150e_coco.py"

    # DETR 체크포인트 파일 (크기가 큰 파일)
    checkpoint_url = "https://download.openmmlab.com/mmdetection/v3.0/detr/detr_r50_8xb2-150e_coco/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth"
    checkpoint_path = "weights/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth"

    print("모델 파일 다운로드 중...")

    # 설정 파일 다운로드
    if not os.path.exists(config_path):
        if not download_file(config_url, config_path):
            return False
    else:
        print(f"✓ 이미 존재: {config_path}")

    # 체크포인트 파일 다운로드 (큰 파일이므로 선택적)
    if not os.path.exists(checkpoint_path):
        print(f"\n체크포인트 파일이 필요합니다: {checkpoint_path}")
        print("파일 크기가 큽니다 (~166MB). 다운로드하시겠습니까? (y/n): ", end="")

        response = input().lower().strip()
        if response in ['y', 'yes']:
            if not download_file(checkpoint_url, checkpoint_path):
                print("\n수동 다운로드 명령어:")
                print(f"wget -O {checkpoint_path} {checkpoint_url}")
                return False
        else:
            print("\n수동 다운로드 명령어:")
            print(f"wget -O {checkpoint_path} {checkpoint_url}")
            return False
    else:
        print(f"✓ 이미 존재: {checkpoint_path}")

    return True


def create_sample_script():
    """샘플 실행 스크립트 생성"""
    script_content = '''#!/bin/bash
# DETR 객체 탐지 실행 예제

echo "=== DETR 객체 탐지 예제 ==="

# 기본 실행 (image_5.jpeg 파일 필요)
echo "1. 기본 탐지 (threshold=0.5)"
python main.py --input image_5.jpeg --threshold 0.5

echo "2. 높은 신뢰도 탐지 (threshold=0.7)"
python main.py --input image_5.jpeg --threshold 0.7

echo "3. 낮은 신뢰도 탐지 (threshold=0.3)"
python main.py --input image_5.jpeg --threshold 0.3

echo "결과는 results/ 디렉터리에 저장됩니다."
'''

    with open('run_detection.sh', 'w') as f:
        f.write(script_content)

    # 실행 권한 부여 (Unix 계열)
    try:
        os.chmod('run_detection.sh', 0o755)
        print("✓ run_detection.sh 샘플 스크립트 생성")
    except:
        print("✓ run_detection.sh 샘플 스크립트 생성 (권한 설정 실패)")


def create_readme():
    """README 파일 업데이트"""
    readme_content = '''# DETR 객체 탐지 프로젝트

이 프로젝트는 DETR(Detection Transformer)를 사용한 객체 탐지를 구현합니다.

## 설치 방법

1. 환경 설정 스크립트 실행:
```bash
python setup.py
```

2. 또는 수동 설치:
```bash
pip install torch torchvision opencv-python numpy matplotlib tqdm
pip install openmim
mim install mmengine mmcv mmdet
```

## 사용 방법

### 기본 사용법
```bash
python main.py --input your_image.jpg --threshold 0.5
```

### 매개변수
- `--input, -i`: 입력 이미지 경로 (기본값: image_5.jpeg)
- `--threshold, -t`: 탐지 신뢰도 임계값 (기본값: 0.5, 범위: 0.0-1.0)

### 예제
```bash
# 높은 정확도로 탐지
python main.py --input sample.jpg --threshold 0.7

# 더 많은 객체 탐지 (낮은 임계값)
python main.py --input sample.jpg --threshold 0.3
```

## 프로젝트 구조
```
├── main.py              # 메인 실행 파일
├── detector.py          # DETR 탐지 모듈
├── setup.py            # 환경 설정 스크립트
├── requirements.txt    # 패키지 의존성
├── configs/            # 모델 설정 파일
├── weights/            # 모델 가중치 파일
├── results/            # 결과 이미지 저장
└── data/              # 입력 데이터
```

## 결과
- 탐지된 객체들이 바운딩 박스로 표시됩니다
- 각 객체의 신뢰도 점수가 표시됩니다
- 결과는 `results/` 디렉터리에 저장됩니다

## Git 커밋 히스토리
1. `feat: Initialize project with git setup and basic structure`
2. `feat: Add DETR detection module with bbox visualization`
3. `feat: Add main execution script for object detection`
4. `feat: Add setup script and documentation`

## 다음 개발 예정
- [ ] SAM 모델 통합
- [ ] Instance Segmentation 구현
- [ ] 성능 최적화
'''

    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print("✓ README.md 업데이트 완료")


def main():
    """메인 설정 함수"""
    print("DETR 객체 탐지 프로젝트 환경 설정")
    print("=" * 50)

    # Python 버전 확인
    if not check_python_version():
        return False

    # 디렉터리 생성
    print("\n디렉터리 구조 생성 중...")
    create_directories()

    # 패키지 설치 여부 확인
    print("\n패키지 설치를 진행하시겠습니까? (y/n): ", end="")
    if input().lower().strip() in ['y', 'yes']:
        if not install_packages():
            print("패키지 설치에 실패했습니다.")
            return False

    # 모델 파일 다운로드
    print("\n모델 파일 다운로드를 진행하시겠습니까? (y/n): ", end="")
    if input().lower().strip() in ['y', 'yes']:
        if not download_model_files():
            print("모델 파일 다운로드에 실패했습니다.")
            return False

    # 샘플 스크립트 및 문서 생성
    print("\n문서 및 스크립트 생성 중...")
    create_sample_script()
    create_readme()

    print("\n" + "=" * 50)
    print("환경 설정이 완료되었습니다!")
    print("\n다음 단계:")
    print("1. 입력 이미지를 준비하세요 (예: image_5.jpeg)")
    print("2. 탐지를 실행하세요: python main.py --input your_image.jpg")
    print("3. 결과를 results/ 디렉터리에서 확인하세요")
    print("=" * 50)

    return True


if __name__ == '__main__':
    main()
