# DETR + SAM Instance Segmentation Project

DETR(Detection Transformer)과 SAM(Segment Anything Model)을 사용한 객체 탐지 및 인스턴스 세그멘테이션 프로젝트입니다.

## 🚀 빠른 시작

### 1. 저장소 클론
```bash
git clone <repository-url>
cd detr-sam-segmentation
```

### 2. 모델 파일 다운로드
```bash
# 자동 다운로드 (추천)
python download_models.py

# 또는 설정 파일만 다운로드 (빠름)
python download_models.py --skip-large
```

### 3. 객체 탐지 실행
```bash
python main.py --input your_image.jpg --threshold 0.5
```

## 📁 프로젝트 구조

```
├── main.py              # 메인 실행 파일
├── detector.py          # DETR 탐지 모듈
├── download_models.py   # 모델 파일 다운로드 스크립트
├── test_detection.py    # 테스트 도구
├── configs/            # 모델 설정 파일 (자동 다운로드)
│   └── detr_r50_8xb2-150e_coco.py
├── weights/            # 모델 체크포인트 (자동 다운로드)
│   └── detr_r50_8xb2-150e_coco_*.pth
├── results/            # 결과 이미지 저장소
└── README.md
```

## ⚠️ 중요: 모델 파일 관리

### Git에서 제외되는 파일들
- `weights/` - 모델 체크포인트 파일 (용량이 큼)
- `configs/` - 설정 파일 (변경 가능성)
- `results/` - 결과 이미지
- `*.pth`, `*.pt` - 모든 PyTorch 모델 파일

### 모델 파일 다운로드 방법

#### 방법 1: 자동 스크립트 (추천)
```bash
# 모든 파일 다운로드
python download_models.py

# 설정 파일만 다운로드 (큰 파일 제외)
python download_models.py --skip-large

# 다운로드 상태 확인
python download_models.py --verify-only
```

#### 방법 2: 수동 다운로드
```bash
# 설정 파일
mkdir -p configs
wget -O configs/detr_r50_8xb2-150e_coco.py \
  https://github.com/open-mmlab/mmdetection/raw/main/configs/detr/detr_r50_8xb2-150e_coco.py

# 체크포인트 파일 (166MB)
mkdir -p weights
wget -O weights/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth \
  https://download.openmmlab.com/mmdetection/v3.0/detr/detr_r50_8xb2-150e_coco/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth
```

## 💻 사용법

### 기본 사용
```bash
python main.py --input image.jpg --threshold 0.5
```

### 옵션
- `--input, -i`: 입력 이미지 경로
- `--threshold, -t`: 탐지 신뢰도 임계값 (0.0-1.0)

### 예제
```bash
# 높은 정확도로 탐지
python main.py --input sample.jpg --threshold 0.7

# 더 많은 객체 탐지 (낮은 임계값)
python main.py --input sample.jpg --threshold 0.3
```

### 테스트
```bash
# 전체 시스템 테스트
python test_detection.py

# 특정 이미지 테스트
python test_detection.py --image your_image.jpg
```

## 🛠️ 개발 환경

### Conda 환경 (추천)
```bash
# 현재 sam 환경 사용
conda activate sam

# 필요한 패키지가 없는 경우
conda install pytorch torchvision opencv numpy matplotlib
pip install mmdet mmcv mmengine
```

### 필수 패키지
- PyTorch >= 1.9.0
- MMDetection >= 3.0.0
- OpenCV >= 4.5.0
- NumPy >= 1.21.0

## 🌊 Git Flow 워크플로우

### 브랜치 구조
```
main (프로덕션)
├── develop (통합)
    ├── feature/detr (DETR 구현)
    └── feature/sam (SAM 통합 - 예정)
```

### 개발 과정
```bash
# feature 브랜치에서 개발
git checkout -b feature/detr
git add detector.py
git commit -m "feat(detr): Implement detection module"

# develop에 병합
git checkout develop
git merge --no-ff feature/detr
```

## 🚫 Git에 올리면 안 되는 것들

- ❌ `weights/*.pth` - 모델 체크포인트 (500MB+)
- ❌ `results/*.png` - 결과 이미지
- ❌ `__pycache__/` - Python 캐시
- ❌ `.env` - 환경 변수 파일

## 🔄 다음 개발 계획

- [ ] SAM 모델 통합
- [ ] Instance Segmentation 구현
- [ ] 성능 최적화
- [ ] 배치 처리 지원
- [ ] GUI 인터페이스

## 🤝 협업 가이드

### 새 개발자 온보딩
1. 저장소 클론: `git clone <repo>`
2. 모델 다운로드: `python download_models.py`
3. 테스트 실행: `python test_detection.py`
4. 개발 시작: `git checkout -b feature/your-feature`

### 문제 해결
- 모델 파일 없음: `python download_models.py`
- 테스트 실패: `python test_detection.py --verify-only`
- GPU 메모리 부족: `--threshold` 값을 높여서 객체 수 줄이기

## 📞 문의

프로젝트 관련 문의사항은 이슈로 등록해 주세요.
