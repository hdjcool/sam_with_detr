"""
SAM (Segment Anything Model) 통합 모듈
- Facebook의 Segment Anything Model 사용
- DETR bbox를 prompt로 활용한 Instance Segmentation
- 다양한 SAM 모델 크기 지원 (ViT-B, ViT-L, ViT-H)
- 세그멘테이션 전용 시각화 기능 추가
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


def visualize_segmentation_only(image: np.ndarray, masks: List[np.ndarray],
                               colors: Optional[List[Tuple[int, int, int]]] = None,
                               alpha: float = 0.6,
                               use_smooth_edges: bool = True) -> np.ndarray:
    """
    원본 이미지에 세그멘테이션 마스크만 깔끔하게 오버레이
    bbox나 텍스트 없이 순수한 세그멘테이션 결과만 표시

    Args:
        image: 입력 이미지 (BGR)
        masks: 마스크 리스트
        colors: 각 마스크의 색상 리스트 (BGR), None이면 자동 생성
        alpha: 마스크 투명도 (0.0-1.0)
        use_smooth_edges: 마스크 가장자리 부드럽게 처리할지 여부

    Returns:
        np.ndarray: 세그멘테이션만 오버레이된 깔끔한 이미지
    """
    if not masks:
        print("⚠️ 표시할 마스크가 없습니다.")
        return image.copy()

    # 유효한 마스크만 필터링
    valid_masks = []
    for i, mask in enumerate(masks):
        if isinstance(mask, np.ndarray) and mask.any():
            valid_masks.append(mask)
        else:
            print(f"  ⚠️ 마스크 {i+1}은 비어있거나 유효하지 않습니다.")

    if not valid_masks:
        print("⚠️ 유효한 마스크가 없습니다.")
        return image.copy()

    result_image = image.copy()

    # 고품질 색상 팔레트 생성
    if colors is None:
        colors = generate_high_quality_colors(len(valid_masks))

    print(f"🎨 {len(valid_masks)}개 유효한 마스크를 오버레이 중...")

    # 각 마스크를 개별적으로 처리
    for i, (mask, color) in enumerate(zip(valid_masks, colors)):
        try:
            # 마스크 타입 및 크기 확인
            if not isinstance(mask, np.ndarray):
                print(f"  ❌ 마스크 {i+1}: 올바른 numpy array가 아닙니다.")
                continue

            # 마스크를 boolean에서 uint8로 변환
            if mask.dtype == bool:
                processed_mask = mask.astype(np.uint8) * 255
            else:
                processed_mask = mask.astype(np.uint8)

            # 마스크가 2D인지 확인
            if len(processed_mask.shape) != 2:
                print(f"  ❌ 마스크 {i+1}: 2D 마스크가 아닙니다. 형태: {processed_mask.shape}")
                continue

            # 이미지와 마스크 크기 일치 확인
            if processed_mask.shape != image.shape[:2]:
                print(f"  ❌ 마스크 {i+1}: 크기 불일치. 마스크: {processed_mask.shape}, 이미지: {image.shape[:2]}")
                continue

            mask_area = np.sum(processed_mask > 0)
            if mask_area == 0:
                print(f"  ⚠️ 마스크 {i+1}: 빈 마스크입니다.")
                continue

            print(f"  🎯 마스크 {i+1}: {mask_area} 픽셀 ({mask_area/processed_mask.size*100:.1f}%)")

            # 부드러운 가장자리 처리
            if use_smooth_edges and mask_area > 100:  # 작은 마스크는 블러 생략
                processed_mask = cv2.GaussianBlur(processed_mask, (3, 3), 0.5)

            # 색상 마스크 생성
            colored_mask = np.zeros_like(result_image)
            mask_indices = processed_mask > 0
            colored_mask[mask_indices] = color

            # 알파 블렌딩
            if processed_mask.max() > 1:  # 0-255 범위인 경우
                mask_alpha = processed_mask.astype(np.float32) / 255.0
            else:  # 0-1 범위인 경우
                mask_alpha = processed_mask.astype(np.float32)

            # 3채널로 확장
            mask_alpha_3d = np.stack([mask_alpha] * 3, axis=-1)

            # 블렌딩 적용
            result_image = result_image.astype(np.float32)
            colored_mask = colored_mask.astype(np.float32)

            blend_alpha = mask_alpha_3d * alpha
            result_image = result_image * (1 - blend_alpha) + colored_mask * blend_alpha

            result_image = np.clip(result_image, 0, 255).astype(np.uint8)

            print(f"  ✅ 마스크 {i+1} 오버레이 완료")

        except Exception as e:
            print(f"  ❌ 마스크 {i+1} 처리 중 오류: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("✅ 세그멘테이션 전용 시각화 완료")
    return result_image


def generate_high_quality_colors(n: int) -> List[Tuple[int, int, int]]:
    """
    고품질의 구별되는 색상 팔레트 생성

    Args:
        n: 생성할 색상 개수

    Returns:
        List[Tuple[int, int, int]]: BGR 색상 리스트
    """
    if n == 0:
        return []

    colors = []

    # 미리 정의된 고품질 색상 팔레트 (BGR 순서)
    predefined_colors = [
        (255, 100, 100),   # 밝은 파랑
        (100, 255, 100),   # 밝은 초록
        (100, 100, 255),   # 밝은 빨강
        (255, 255, 100),   # 밝은 청록
        (255, 100, 255),   # 밝은 마젠타
        (100, 255, 255),   # 밝은 노랑
        (200, 150, 255),   # 라벤더
        (150, 255, 200),   # 민트
        (255, 200, 150),   # 복숭아
        (255, 150, 200),   # 핑크
        (150, 200, 255),   # 하늘색
        (200, 255, 150),   # 라임
    ]

    # 미리 정의된 색상을 우선 사용
    for i in range(min(n, len(predefined_colors))):
        colors.append(predefined_colors[i])

    # 추가 색상이 필요한 경우 HSV 공간에서 생성
    for i in range(len(predefined_colors), n):
        # 황금 비율을 사용한 색상 분산
        golden_ratio = 0.618033988749
        hue = (i * golden_ratio) % 1.0

        # HSV 값 설정 (채도와 밝기를 높게 설정)
        h = int(hue * 180)
        s = 255  # 최대 채도
        v = 255  # 최대 명도

        # HSV를 BGR로 변환
        hsv = np.array([[[h, s, v]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        colors.append(tuple(map(int, bgr[0][0])))

    return colors


def save_segmentation_only_result(image_path: str, masks: List[np.ndarray],
                                 output_path: str, alpha: float = 0.6,
                                 use_smooth_edges: bool = True) -> bool:
    """
    세그멘테이션 전용 결과를 파일로 저장

    Args:
        image_path: 원본 이미지 경로
        masks: 세그멘테이션 마스크 리스트
        output_path: 저장할 파일 경로
        alpha: 마스크 투명도
        use_smooth_edges: 부드러운 가장자리 처리 여부

    Returns:
        bool: 저장 성공 여부
    """
    try:
        # 원본 이미지 로딩
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 원본 이미지를 읽을 수 없습니다: {image_path}")
            return False

        # 세그멘테이션 전용 시각화
        result = visualize_segmentation_only(
            image, masks, alpha=alpha, use_smooth_edges=use_smooth_edges
        )

        # 결과 저장
        success = cv2.imwrite(output_path, result)

        if success:
            print(f"💾 세그멘테이션 전용 결과 저장: {output_path}")
        else:
            print(f"❌ 파일 저장 실패: {output_path}")

        return success

    except Exception as e:
        print(f"❌ 세그멘테이션 결과 저장 중 오류: {e}")
        return False


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
