"""
SAM (Segment Anything Model) 통합 모듈
- Facebook의 Segment Anything Model 사용
- DETR bbox를 prompt로 활용한 Instance Segmentation
- 다양한 SAM 모델 크기 지원 (ViT-B, ViT-L, ViT-H)
"""

import cv2
import numpy as np
import torch
import os
from typing import List, Tuple, Optional, Union

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    print("⚠️ segment-anything 패키지가 설치되지 않았습니다.")
    print("설치 방법: pip install segment-anything")
    SAM_AVAILABLE = False


class SAMConfig:
    """SAM 모델 설정 클래스"""

    # 지원하는 SAM 모델들
    MODELS = {
        'vit_b': {
            'checkpoint': 'weights/sam_vit_b_01ec64.pth',
            'model_type': 'vit_b',
            'description': 'ViT-B (Base) - 빠르고 효율적',
            'size_mb': 375
        },
        'vit_l': {
            'checkpoint': 'weights/sam_vit_l_0b3195.pth',
            'model_type': 'vit_l',
            'description': 'ViT-L (Large) - 높은 성능',
            'size_mb': 1200
        },
        'vit_h': {
            'checkpoint': 'weights/sam_vit_h_4b8939.pth',
            'model_type': 'vit_h',
            'description': 'ViT-H (Huge) - 최고 성능',
            'size_mb': 2400
        }
    }

    DEFAULT_MODEL = 'vit_b'


def check_sam_availability():
    """SAM 사용 가능 여부 확인"""
    return SAM_AVAILABLE


def list_available_models():
    """사용 가능한 SAM 모델 목록 출력"""
    print("🤖 사용 가능한 SAM 모델:")
    print("=" * 40)

    for model_key, config in SAMConfig.MODELS.items():
        checkpoint_path = config['checkpoint']
        exists = os.path.exists(checkpoint_path)
        status = "✅ 사용 가능" if exists else "❌ 다운로드 필요"

        print(f"{model_key.upper()}: {config['description']}")
        print(f"  파일: {checkpoint_path}")
        print(f"  크기: ~{config['size_mb']}MB")
        print(f"  상태: {status}")
        print()


def initialize(model_type: str = 'vit_b', device: str = 'auto') -> Optional[SamPredictor]:
    """
    SAM 모델 초기화

    Args:
        model_type: SAM 모델 타입 ('vit_b', 'vit_l', 'vit_h')
        device: 사용할 디바이스 ('auto', 'cpu', 'cuda', 'mps')

    Returns:
        SamPredictor: 초기화된 SAM predictor 또는 None
    """
    if not SAM_AVAILABLE:
        print("❌ segment-anything 패키지가 설치되지 않았습니다.")
        return None

    # 디바이스 자동 선택
    if device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    print(f"🤖 SAM 모델 초기화 중...")
    print(f"모델 타입: {model_type.upper()}")
    print(f"디바이스: {device}")

    try:
        # 모델 설정 확인
        if model_type not in SAMConfig.MODELS:
            print(f"❌ 지원하지 않는 모델 타입: {model_type}")
            print(f"지원 모델: {list(SAMConfig.MODELS.keys())}")
            return None

        config = SAMConfig.MODELS[model_type]
        checkpoint_path = config['checkpoint']

        # 체크포인트 파일 존재 확인
        if not os.path.exists(checkpoint_path):
            print(f"❌ 모델 파일을 찾을 수 없습니다: {checkpoint_path}")
            print(f"다운로드 방법:")
            print(f"wget -O {checkpoint_path} https://dl.fbaipublicfiles.com/segment_anything/sam_{model_type}_*.pth")
            return None

        # 모델 로딩
        sam = sam_model_registry[config['model_type']](checkpoint=checkpoint_path)
        sam.to(device=device)

        # Predictor 생성
        predictor = SamPredictor(sam)

        print(f"✅ SAM 모델 초기화 완료!")
        print(f"모델: {config['description']}")
        print(f"파일 크기: ~{config['size_mb']}MB")

        return predictor

    except Exception as e:
        print(f"❌ SAM 모델 초기화 실패: {e}")
        return None


def inference_with_boxes(image_path: str, bboxes: List[List[float]],
                        predictor: SamPredictor) -> List[np.ndarray]:
    """
    DETR의 bounding box를 prompt로 사용하여 SAM inference 수행

    Args:
        image_path: 입력 이미지 경로
        bboxes: DETR에서 검출된 bounding box 리스트 [[x1,y1,x2,y2], ...]
        predictor: 초기화된 SAM predictor

    Returns:
        List[np.ndarray]: 각 bbox에 대응하는 마스크 리스트
    """
    if predictor is None:
        print("❌ SAM predictor가 초기화되지 않았습니다.")
        return []

    if not bboxes:
        print("⚠️ 입력 bounding box가 없습니다.")
        return []

    try:
        # 이미지 로딩
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 이미지를 읽을 수 없습니다: {image_path}")
            return []

        # RGB로 변환 (SAM은 RGB 사용)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # SAM에 이미지 설정
        predictor.set_image(image_rgb)

        masks = []
        print(f"🎯 {len(bboxes)}개 객체에 대해 세그멘테이션 수행 중...")

        for i, bbox in enumerate(bboxes):
            try:
                # bbox를 numpy array로 변환 [x1, y1, x2, y2]
                input_box = np.array(bbox)

                # SAM inference
                mask, scores, logits = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],  # [1, 4] 형태로 변환
                    multimask_output=False
                )

                # 가장 좋은 마스크 선택 (multimask_output=False이므로 1개)
                best_mask = mask[0]  # [H, W] 형태
                masks.append(best_mask)

                print(f"  ✅ 객체 {i+1}/{len(bboxes)} 세그멘테이션 완료 (점수: {scores[0]:.3f})")

            except Exception as e:
                print(f"  ❌ 객체 {i+1} 세그멘테이션 실패: {e}")
                # 빈 마스크 추가 (같은 크기 유지)
                h, w = image.shape[:2]
                empty_mask = np.zeros((h, w), dtype=bool)
                masks.append(empty_mask)
                continue

        print(f"✅ 세그멘테이션 완료: {len(masks)}개 마스크 생성")
        return masks

    except Exception as e:
        print(f"❌ SAM inference 실패: {e}")
        return []


def inference(input_path: str, prompt_bbox: List[float], predictor: SamPredictor) -> np.ndarray:
    """
    단일 bbox에 대한 SAM inference (기존 인터페이스 호환)

    Args:
        input_path: 입력 이미지 경로
        prompt_bbox: 단일 bounding box [x1, y1, x2, y2]
        predictor: 초기화된 SAM predictor

    Returns:
        np.ndarray: 생성된 마스크 [H, W]
    """
    masks = inference_with_boxes(input_path, [prompt_bbox], predictor)

    if masks:
        return masks[0]
    else:
        # 빈 마스크 반환
        image = cv2.imread(input_path)
        if image is not None:
            h, w = image.shape[:2]
            return np.zeros((h, w), dtype=bool)
        else:
            return np.zeros((480, 640), dtype=bool)


def visualize_masks(image: np.ndarray, masks: List[np.ndarray],
                   colors: Optional[List[Tuple[int, int, int]]] = None,
                   alpha: float = 0.5) -> np.ndarray:
    """
    이미지에 여러 마스크를 시각화

    Args:
        image: 입력 이미지 (BGR)
        masks: 마스크 리스트
        colors: 각 마스크의 색상 리스트 (BGR)
        alpha: 마스크 투명도 (0.0-1.0)

    Returns:
        np.ndarray: 마스크가 오버레이된 이미지
    """
    if not masks:
        return image.copy()

    result_image = image.copy()

    # 기본 색상 생성
    if colors is None:
        colors = []
        for i in range(len(masks)):
            # HSV 색공간에서 구별되는 색상 생성
            hue = int(180 * i / len(masks))
            hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            colors.append(tuple(map(int, bgr[0][0])))

    # 각 마스크를 오버레이
    for mask, color in zip(masks, colors):
        if mask.any():  # 마스크가 비어있지 않은 경우
            # 마스크 영역에 색상 적용
            colored_mask = np.zeros_like(result_image)
            colored_mask[mask] = color

            # 알파 블렌딩으로 오버레이
            result_image = cv2.addWeighted(result_image, 1-alpha, colored_mask, alpha, 0)

    return result_image


def visualize(input_img: np.ndarray, input_mask: np.ndarray,
             mask_color: Tuple[int, int, int]) -> np.ndarray:
    """
    단일 마스크 시각화 (기존 인터페이스 호환)

    Args:
        input_img: 입력 이미지 (BGR)
        input_mask: 마스크 [H, W]
        mask_color: 마스크 색상 (BGR)

    Returns:
        np.ndarray: 마스크가 오버레이된 이미지
    """
    return visualize_masks(input_img, [input_mask], [mask_color])


def save_masks(masks: List[np.ndarray], output_dir: str,
               filename_prefix: str = "mask") -> List[str]:
    """
    마스크들을 개별 파일로 저장

    Args:
        masks: 저장할 마스크 리스트
        output_dir: 출력 디렉토리
        filename_prefix: 파일명 접두사

    Returns:
        List[str]: 저장된 파일 경로 리스트
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    for i, mask in enumerate(masks):
        # 마스크를 0-255 범위로 변환
        mask_img = (mask.astype(np.uint8) * 255)

        # 파일 저장
        filename = f"{filename_prefix}_{i+1:03d}.png"
        filepath = os.path.join(output_dir, filename)

        cv2.imwrite(filepath, mask_img)
        saved_files.append(filepath)

    print(f"✅ {len(saved_files)}개 마스크 저장 완료: {output_dir}")
    return saved_files


def get_model_info(model_type: str = 'vit_b') -> dict:
    """
    SAM 모델 정보 반환

    Args:
        model_type: 모델 타입

    Returns:
        dict: 모델 정보
    """
    if model_type in SAMConfig.MODELS:
        config = SAMConfig.MODELS[model_type].copy()
        config['available'] = os.path.exists(config['checkpoint'])
        return config
    else:
        return {'error': f'Unknown model type: {model_type}'}


# SAM 모델 정보 출력 (모듈 import 시)
if __name__ == "__main__":
    print("🤖 SAM (Segment Anything Model) 모듈")
    print("=" * 40)

    if SAM_AVAILABLE:
        list_available_models()

        # 간단한 테스트
        print("🧪 SAM 초기화 테스트...")
        predictor = initialize('vit_b')
        if predictor:
            print("✅ SAM 모듈이 정상적으로 작동합니다!")
        else:
            print("❌ SAM 모델을 초기화할 수 없습니다.")
    else:
        print("❌ segment-anything 패키지를 설치해주세요.")
        print("설치 방법: pip install segment-anything")
