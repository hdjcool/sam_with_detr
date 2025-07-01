# DETR + SAM Instance Segmentation 프로젝트

이 프로젝트는 DETR(Detection Transformer)와 SAM(Segment Anything Model)을 결합하여 고품질 Instance Segmentation을 수행합니다.

## ✨ 주요 기능

- **DETR 객체 탐지**: 최신 Transformer 기반 객체 탐지
- **SAM Instance Segmentation**: Facebook의 Segment Anything Model을 사용한 정밀한 세그멘테이션
- **세그멘테이션 전용 출력**: 원본 이미지에 깔끔한 마스크만 오버레이 ⭐ **NEW**
- **다양한 시각화 옵션**: 탐지 결과, 마스크, 통합 결과 등 다양한 출력 형태
- **고품질 색상 팔레트**: 구별되는 색상으로 각 객체 구분

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 필요한 패키지 설치
pip install torch torchvision opencv-python numpy matplotlib tqdm
pip install openmim
mim install mmengine mmcv mmdet
pip install segment-anything

# 또는 setup.py 실행
python setup.py
```

### 2. 모델 파일 다운로드

```bash
# DETR 모델 다운로드
python download_models.py

# SAM 모델 다운로드 (수동)
wget -O weights/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### 3. 실행

```bash
# 🎨 세그멘테이션 전용 모드 (NEW!) - 원본 이미지에 마스크만 깔끔하게 오버레이
python main.py --input your_image.jpg --mode segmentation-only

# 전체 모드 - 모든 결과 (탐지, 마스크, 통합) 생성
python main.py --input your_image.jpg --mode full

# DETR 탐지만
python main.py --input your_image.jpg --mode detr-only
```

## 📋 사용법 상세

### 실행 모드

#### 1. 세그멘테이션 전용 모드 ⭐ **NEW**
```bash
python main.py --input image.jpg --mode segmentation-only --alpha 0.6
```
- 원본 이미지에 세그멘테이션 마스크만 오버레이
- bbox나 텍스트 없이 깔끔한 결과
- `--alpha` 옵션으로 투명도 조절 (0.0-1.0)

**생성되는 파일:**
- `{이미지명}_segmentation_only.png` - 기본 투명도 (0.6)
- `{이미지명}_segmentation_light.png` - 투명한 버전 (0.4)
- `{이미지명}_segmentation_bold.png` - 진한 버전 (0.8)

#### 2. 전체 모드
```bash
python main.py --input image.jpg --mode full
```
- 모든 결과를 생성 (탐지 + 마스크 + 통합 + 세그멘테이션 전용)

**생성되는 파일:**
- `{이미지명}_1_detr_detection.png` - DETR 탐지 결과 (bbox + 클래스명)
- `{이미지명}_2_sam_masks.png` - SAM 마스크 결과
- `{이미지명}_3_combined_result.png` - 통합 결과 (bbox + 마스크 + 클래스명)
- `{이미지명}_4_segmentation_only.png` - 세그멘테이션 전용 결과
- `{이미지명}_individual_masks/` - 개별 마스크 파일들

#### 3. DETR 전용 모드
```bash
python main.py --input image.jpg --mode detr-only
```
- DETR 탐지만 수행 (SAM 없이)

### 매개변수 옵션

```bash
python main.py \
  --input your_image.jpg \          # 입력 이미지 경로
  --mode segmentation-only \        # 실행 모드
  --threshold 0.5 \                 # DETR 탐지 신뢰도 임계값 (0.0-1.0)
  --alpha 0.6 \                     # 마스크 투명도 (segmentation-only 모드)
  --output results                  # 출력 디렉터리
```

## 🧪 테스트

### 전체 시스템 테스트
```bash
# 통합 파이프라인 테스트
python integrated_pipeline_test.py

# 세그멘테이션 전용 기능 테스트
python test_segmentation_only.py

# DETR 탐지 기능 테스트
python test_detection.py
```

### 개별 기능 테스트
```bash
# 색상 생성 테스트만
python test_segmentation_only.py --test color

# SAM 초기화 테스트만
python test_segmentation_only.py --test sam

# 시각화 기능 테스트만
python test_segmentation_only.py --test visualization
```

## 📁 프로젝트 구조

```
├── main.py                      # 메인 실행 파일 ⭐ 업데이트됨
├── detector.py                  # DETR 탐지 모듈
├── sam.py                       # SAM 세그멘테이션 모듈 ⭐ 업데이트됨
├── setup.py                     # 환경 설정 스크립트
├── download_models.py           # 모델 다운로드 스크립트
├── integrated_pipeline_test.py  # 통합 테스트
├── test_segmentation_only.py    # 세그멘테이션 전용 테스트 ⭐ 새로 추가
├── test_detection.py            # DETR 테스트
├── configs/                     # 모델 설정 파일
│   └── detr_r50_8xb2-150e_coco.py
├── weights/                     # 모델 가중치 파일
│   ├── detr_r50_8xb2-150e_coco_*.pth
│   └── sam_vit_b_01ec64.pth
└── results/                     # 결과 이미지 저장
```

## 🎨 새로운 기능: 세그멘테이션 전용 출력

### 기존 방식의 문제점
- bbox와 클래스명이 함께 표시되어 시각적으로 복잡
- 순수한 세그멘테이션 결과를 보기 어려움

### 새로운 방식의 장점
- **깔끔한 결과**: 원본 이미지에 마스크만 오버레이
- **고품질 색상**: 구별되는 색상 팔레트로 각 객체 구분
- **부드러운 가장자리**: 가우시안 블러로 자연스러운 가장자리 처리
- **투명도 조절**: 다양한 투명도로 결과 생성
- **사용자 친화적**: 직관적이고 아름다운 시각화

### 코드 예제
```python
import sam

# 세그멘테이션 전용 시각화
result = sam.visualize_segmentation_only(
    image, masks,
    alpha=0.6,                    # 투명도
    use_smooth_edges=True         # 부드러운 가장자리
)

# 파일로 저장
sam.save_segmentation_only_result(
    input_path, masks, output_path,
    alpha=0.6, use_smooth_edges=True
)
```

## 🔧 고급 사용법

### 배치 처리
```bash
# 여러 이미지 처리
for img in *.jpg; do
    python main.py --input "$img" --mode segmentation-only --alpha 0.6
done
```

### 다양한 투명도로 결과 생성
```bash
# 투명한 결과
python main.py --input image.jpg --mode segmentation-only --alpha 0.3

# 기본 결과
python main.py --input image.jpg --mode segmentation-only --alpha 0.6

# 진한 결과
python main.py --input image.jpg --mode segmentation-only --alpha 0.9
```

### 성능 최적화
```bash
# 높은 신뢰도로 정확한 결과
python main.py --input image.jpg --threshold 0.7 --mode segmentation-only

# 더 많은 객체 탐지
python main.py --input image.jpg --threshold 0.3 --mode segmentation-only
```

## 📊 성능 정보

### 지원하는 SAM 모델
- **ViT-B**: 빠르고 효율적 (~375MB)
- **ViT-L**: 높은 성능 (~1.2GB)
- **ViT-H**: 최고 성능 (~2.4GB)

### 하드웨어 요구사항
- **최소**: CPU, 8GB RAM
- **권장**: GPU (CUDA/MPS), 16GB RAM
- **최적**: GPU, 32GB RAM

### 처리 시간 (대략)
- **DETR 탐지**: 1-3초 (GPU 기준)
- **SAM 세그멘테이션**: 객체당 0.1-0.5초
- **시각화**: 1초 미만

## 🐛 문제 해결

### 일반적인 오류

1. **모델 파일 없음**
   ```
   ❌ 모델 파일을 찾을 수 없습니다
   ```
   해결: `python download_models.py` 실행

2. **SAM 패키지 없음**
   ```
   ❌ segment-anything 패키지가 설치되지 않았습니다
   ```
   해결: `pip install segment-anything`

3. **GPU 메모리 부족**
   ```
   RuntimeError: CUDA out of memory
   ```
   해결: `--device cpu` 옵션 사용 또는 이미지 크기 줄이기

### 디버깅 팁

```bash
# 모든 의존성 확인
python -c "import torch, cv2, mmdet, segment_anything; print('All OK')"

# 테스트 실행으로 문제 진단
python test_segmentation_only.py --test all

# 상세한 로그 출력
python main.py --input image.jpg --mode segmentation-only -v
```

## 🤝 기여하기

1. Fork 이 저장소
2. Feature 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경사항 커밋 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 Push (`git push origin feature/amazing-feature`)
5. Pull Request 생성

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🙏 감사의 말

- [MMDetection](https://github.com/open-mmlab/mmdetection) - DETR 구현
- [Segment Anything](https://github.com/facebookresearch/segment-anything) - SAM 모델
- [DETR 논문](https://arxiv.org/abs/2005.12872) - Detection Transformer
- [SAM 논문](https://arxiv.org/abs/2304.02643) - Segment Anything Model

## 📞 지원

문제가 있거나 질문이 있으시면:
1. GitHub Issues에 문제 보고
2. 테스트 스크립트로 문제 진단
3. 문서의 문제 해결 섹션 참고

---

**🎉 즐거운 Instance Segmentation을 경험해보세요!**
