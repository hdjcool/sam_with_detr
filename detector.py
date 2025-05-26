"""
DETR 객체 탐지 모듈
- MMDetection을 사용한 DETR 모델 로딩
- 바운딩 박스 탐지 및 시각화 기능
"""

import torch
import warnings
import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
from mmdet.apis import DetInferencer


class TorchLoadPatcher:
    """torch.load 함수를 안전하게 패치하는 클래스"""

    def __init__(self):
        self.original_torch_load = torch.load
        self.is_patched = False

    def patch(self, suppress_warnings: bool = True):
        """torch.load를 패치합니다"""
        if self.is_patched:
            return

        def safe_torch_load(*args, **kwargs):
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False

                if not suppress_warnings:
                    warnings.warn(
                        "torch.load with weights_only=False. "
                        "Please ensure you trust the source of the file.",
                        UserWarning
                    )
            return self.original_torch_load(*args, **kwargs)

        torch.load = safe_torch_load
        self.is_patched = True

    def unpatch(self):
        """패치를 해제합니다"""
        if self.is_patched:
            torch.load = self.original_torch_load
            self.is_patched = False


# 전역 패처 인스턴스
_torch_patcher = TorchLoadPatcher()


def setup_warnings_filter():
    """경고 필터를 설정합니다"""
    warnings.filterwarnings("ignore", message=".*weights_only.*")
    warnings.filterwarnings("ignore", message="Failed to search registry")
    warnings.filterwarnings("ignore", message="Failed to add.*LocalVisBackend")
    warnings.filterwarnings("ignore", message=".*state_dict.*match.*")


def validate_model_files(cfg_path: str, ckpt_path: str) -> None:
    """모델 파일들의 존재 여부를 검증합니다"""
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {cfg_path}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {ckpt_path}")


def generate_distinct_colors(n: int) -> List[Tuple[int, int, int]]:
    """구별되는 색상들을 생성하는 함수"""
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        color = tuple(map(int, bgr[0][0]))
        colors.append(color)
    return colors


def initialize(cfg_path: str, ckpt_path: str, device: str = "cpu") -> DetInferencer:
    """
    DETR 탐지 모델을 초기화합니다.

    Args:
        cfg_path: 설정 파일 경로
        ckpt_path: 체크포인트 파일 경로
        device: 사용할 디바이스 ('cpu', 'cuda', 'mps')

    Returns:
        DetInferencer: 초기화된 탐지 모델
    """
    try:
        # 경고 필터 설정
        setup_warnings_filter()

        # torch.load 패치
        _torch_patcher.patch(suppress_warnings=True)

        # 파일 존재 여부 검증
        validate_model_files(cfg_path, ckpt_path)

        # 디바이스 유효성 검사
        if device not in ['cpu', 'cuda', 'mps']:
            print(f"경고: 알 수 없는 디바이스 '{device}', 'cpu'로 설정합니다.")
            device = 'cpu'

        # 모델 초기화
        model = DetInferencer(
            model=cfg_path,
            weights=ckpt_path,
            device=device,
            show_progress=False
        )

        print(f"DETR 모델이 성공적으로 초기화되었습니다. (디바이스: {device})")
        return model

    except Exception as e:
        print(f"모델 초기화 중 오류 발생: {e}")
        raise RuntimeError(f"DETR 모델 초기화 실패: {e}")


def detect_objects(image_path: str, model: DetInferencer,
                  score_threshold: float = 0.5) -> dict:
    """
    이미지에서 객체를 탐지합니다.

    Args:
        image_path: 입력 이미지 경로
        model: 초기화된 탐지 모델
        score_threshold: 신뢰도 임계값

    Returns:
        dict: 탐지 결과 (bboxes, scores, labels)
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"입력 이미지를 찾을 수 없습니다: {image_path}")

        # 추론 수행
        results = model(
            inputs=image_path,
            out_dir=None,
            no_save_vis=True,
            pred_score_thr=score_threshold
        )

        # 결과 처리
        if not results or 'predictions' not in results or not results['predictions']:
            print("탐지된 객체가 없습니다.")
            return {'bboxes': [], 'scores': [], 'labels': []}

        predictions = results['predictions'][0]

        # 유효한 탐지 결과 필터링
        valid_indices = [
            i for i, score in enumerate(predictions['scores'])
            if score >= score_threshold
        ]

        if not valid_indices:
            print(f"신뢰도 {score_threshold} 이상인 객체가 없습니다.")
            return {'bboxes': [], 'scores': [], 'labels': []}

        # 결과 정리
        detection_result = {
            'bboxes': [predictions['bboxes'][i] for i in valid_indices],
            'scores': [predictions['scores'][i] for i in valid_indices],
            'labels': [predictions['labels'][i] for i in valid_indices] if 'labels' in predictions else []
        }

        # 결과 통계 출력
        avg_score = sum(detection_result['scores']) / len(detection_result['scores'])
        print(f"탐지 완료: {len(detection_result['bboxes'])}개 객체 (평균 신뢰도: {avg_score:.3f})")

        return detection_result

    except Exception as e:
        print(f"객체 탐지 중 오류 발생: {e}")
        return {'bboxes': [], 'scores': [], 'labels': []}


def visualize_detections(image: np.ndarray, detections: dict,
                        thickness: int = 2, font_scale: float = 0.7) -> np.ndarray:
    """
    탐지 결과를 이미지에 시각화합니다.

    Args:
        image: 입력 이미지 (BGR)
        detections: 탐지 결과 딕셔너리
        thickness: 박스 선 두께
        font_scale: 폰트 크기

    Returns:
        np.ndarray: 시각화된 이미지
    """
    if not detections['bboxes']:
        return image.copy()

    vis_image = image.copy()
    bboxes = detections['bboxes']
    scores = detections['scores']

    # 색상 생성
    colors = generate_distinct_colors(len(bboxes))

    # 각 바운딩 박스 그리기
    for i, (bbox, score) in enumerate(zip(bboxes, scores)):
        try:
            # 바운딩 박스 좌표 (정수로 변환)
            x1, y1, x2, y2 = map(int, bbox)
            color = colors[i % len(colors)]

            # 바운딩 박스 그리기
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)

            # 라벨 텍스트 준비
            label_text = f"Object_{i+1} {score:.2f}"

            # 텍스트 배경 박스 크기 계산
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # 텍스트 위치 계산
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10

            # 텍스트 배경 그리기
            cv2.rectangle(
                vis_image,
                (text_x, text_y - text_height - baseline),
                (text_x + text_width, text_y + baseline),
                color,
                -1
            )

            # 텍스트 그리기
            cv2.putText(
                vis_image,
                label_text,
                (text_x, text_y - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness - 1 if thickness > 1 else 1
            )

        except Exception as e:
            print(f"바운딩 박스 {i} 시각화 중 오류: {e}")
            continue

    return vis_image


def save_detection_result(image_path: str, output_path: str, model: DetInferencer,
                         score_threshold: float = 0.5) -> bool:
    """
    탐지 결과를 시각화하여 저장합니다.

    Args:
        image_path: 입력 이미지 경로
        output_path: 출력 이미지 경로
        model: 초기화된 탐지 모델
        score_threshold: 신뢰도 임계값

    Returns:
        bool: 저장 성공 여부
    """
    try:
        # 원본 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")

        # 객체 탐지
        detections = detect_objects(image_path, model, score_threshold)

        if not detections['bboxes']:
            print("탐지된 객체가 없어서 원본 이미지를 저장합니다.")
            return cv2.imwrite(output_path, image)

        # 시각화
        vis_image = visualize_detections(image, detections)

        # 저장
        success = cv2.imwrite(output_path, vis_image)
        if success:
            print(f"탐지 결과가 {output_path}에 저장되었습니다. (객체 수: {len(detections['bboxes'])})")

        return success

    except Exception as e:
        print(f"탐지 결과 저장 중 오류: {e}")
        return False


def cleanup():
    """리소스 정리 함수"""
    _torch_patcher.unpatch()


# 모듈 종료 시 자동으로 패치 해제
import atexit
atexit.register(cleanup)
