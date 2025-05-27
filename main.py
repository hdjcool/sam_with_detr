"""
DETR + SAM Instance Segmentation 메인 실행 파일
- DETR로 객체 탐지 (bbox + 클래스)
- SAM으로 정밀한 Instance Segmentation
- 통합 시각화 및 결과 저장
- 세그멘테이션 전용 깔끔한 출력 지원
"""

import os
import sys
import torch
import argparse
import cv2
import numpy as np
from pathlib import Path

try:
    import detector
    import sam
except ImportError as e:
    print(f"모듈 import 오류: {e}")
    print("detector.py와 sam.py 파일이 현재 디렉터리에 있는지 확인하세요.")
    sys.exit(1)


def get_optimal_device():
    """최적의 디바이스를 선택하는 함수"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def validate_file_exists(file_path: str) -> bool:
    """파일 존재 여부 확인"""
    if not os.path.exists(file_path):
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return False
    return True


def create_output_directory(output_dir: str = "results") -> str:
    """결과 저장용 디렉터리를 생성하는 함수"""
    Path(output_dir).mkdir(exist_ok=True)
    print(f"결과 디렉터리: {output_dir}")
    return output_dir


def check_model_files():
    """필요한 모델 파일들 확인"""
    print("🔍 모델 파일 확인 중...")

    # DETR 모델 파일
    detr_config = "configs/detr_r50_8xb2-150e_coco.py"
    detr_checkpoint = "weights/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth"

    # SAM 모델 파일
    sam_checkpoint = "weights/sam_vit_b_01ec64.pth"

    missing_files = []

    if not validate_file_exists(detr_config):
        missing_files.append(("DETR Config", detr_config))

    if not validate_file_exists(detr_checkpoint):
        missing_files.append(("DETR Checkpoint", detr_checkpoint))

    if not validate_file_exists(sam_checkpoint):
        missing_files.append(("SAM Checkpoint", sam_checkpoint))

    if missing_files:
        print("❌ 다음 모델 파일들이 필요합니다:")
        for name, path in missing_files:
            print(f"  - {name}: {path}")
        print("\n📥 모델 다운로드 방법:")
        print("python download_models.py  # DETR 모델")
        print("python simple_sam_download.py  # SAM 모델")
        return False

    print("✅ 모든 모델 파일이 준비되었습니다.")
    return True


def initialize_models(device: str):
    """DETR과 SAM 모델 초기화"""
    print(f"🤖 모델 초기화 중... (디바이스: {device})")

    # DETR 모델 초기화
    print("📊 DETR 모델 로딩 중...")
    detr_config = "configs/detr_r50_8xb2-150e_coco.py"
    detr_checkpoint = "weights/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth"

    detr_model = detector.initialize(detr_config, detr_checkpoint, device)
    if detr_model is None:
        print("❌ DETR 모델 초기화 실패")
        return None, None

    # SAM 모델 초기화
    print("🎯 SAM 모델 로딩 중...")
    sam_predictor = sam.initialize('vit_b', device)
    if sam_predictor is None:
        print("❌ SAM 모델 초기화 실패")
        return detr_model, None

    print("✅ 모든 모델 초기화 완료!")
    return detr_model, sam_predictor


def run_detection_only(input_path: str, detr_model, score_threshold: float, output_dir: str):
    """DETR 탐지만 수행 (SAM 없이)"""
    print("\n🔍 DETR 객체 탐지 수행 중...")

    # 객체 탐지
    detections = detector.detect_objects(input_path, detr_model, score_threshold)

    if not detections['bboxes']:
        print("❌ 탐지된 객체가 없습니다.")
        return False

    # 결과 출력
    print(f"✅ {len(detections['bboxes'])}개 객체 탐지됨")
    for i, (bbox, score, label) in enumerate(zip(
        detections['bboxes'],
        detections['scores'],
        detections.get('labels', [])
    )):
        if i < len(detections.get('labels', [])):
            class_name = detector.get_class_name(label)
            print(f"  {i+1}. {class_name}: {score:.3f}")
        else:
            print(f"  {i+1}. Object_{i+1}: {score:.3f}")

    # 시각화 저장
    image = cv2.imread(input_path)
    vis_image = detector.visualize_detections(image, detections)

    output_path = os.path.join(output_dir, f"detection_only_{Path(input_path).stem}.png")
    cv2.imwrite(output_path, vis_image)
    print(f"💾 DETR 결과 저장: {output_path}")

    return True


def run_full_pipeline(input_path: str, detr_model, sam_predictor,
                     score_threshold: float, output_dir: str, mode: str = "full"):
    """DETR + SAM 완전한 파이프라인 실행"""
    print(f"\n🚀 DETR + SAM Instance Segmentation 파이프라인 시작")
    print("=" * 60)

    # 1단계: DETR 객체 탐지
    print("1️⃣ DETR 객체 탐지 수행 중...")
    detections = detector.detect_objects(input_path, detr_model, score_threshold)

    if not detections['bboxes']:
        print("❌ 탐지된 객체가 없습니다.")
        return False

    bboxes = detections['bboxes']
    scores = detections['scores']
    labels = detections.get('labels', [])

    print(f"✅ DETR 탐지 완료: {len(bboxes)}개 객체")

    # 탐지 결과 출력
    for i, (bbox, score) in enumerate(zip(bboxes, scores)):
        if i < len(labels):
            class_name = detector.get_class_name(labels[i])
            print(f"  {i+1}. {class_name}: {score:.3f} {[int(x) for x in bbox]}")
        else:
            print(f"  {i+1}. Object_{i+1}: {score:.3f} {[int(x) for x in bbox]}")

    # 2단계: SAM Instance Segmentation
    print("\n2️⃣ SAM Instance Segmentation 수행 중...")
    masks = sam.inference_with_boxes(input_path, bboxes, sam_predictor)

    if not masks:
        print("❌ SAM 세그멘테이션 실패")
        return False

    successful_masks = [mask for mask in masks if mask.any()]
    print(f"✅ SAM 세그멘테이션 완료: {len(successful_masks)}/{len(masks)} 성공")

    # 3단계: 결과 시각화 및 저장
    print("\n3️⃣ 결과 시각화 및 저장 중...")

    image = cv2.imread(input_path)
    if image is None:
        print("❌ 원본 이미지를 읽을 수 없습니다.")
        return False

    # 다양한 시각화 결과 생성
    results = generate_visualization_results(
        image, bboxes, masks, scores, labels,
        input_path, output_dir, mode
    )

    print("✅ 모든 결과 저장 완료!")
    print(f"📂 결과 위치: {output_dir}")
    for desc, path in results:
        print(f"  - {desc}: {path}")

    return True


def generate_visualization_results(image, bboxes, masks, scores, labels,
                                 input_path, output_dir, mode="full"):
    """다양한 시각화 결과 생성"""
    results = []
    base_name = Path(input_path).stem

    # 모드에 따른 결과 생성
    if mode == "segmentation-only":
        # 세그멘테이션 전용 모드: 원본 이미지에 마스크만 오버레이
        print("🎨 세그멘테이션 전용 결과 생성 중...")

        # 1. 기본 세그멘테이션 전용 결과 (투명도 0.6)
        seg_only_path = os.path.join(output_dir, f"{base_name}_segmentation_only.png")
        success = sam.save_segmentation_only_result(
            input_path, masks, seg_only_path, alpha=0.6, use_smooth_edges=True
        )
        if success:
            results.append(("세그멘테이션 전용 결과", seg_only_path))

        # 2. 투명한 세그멘테이션 결과 (투명도 0.4)
        seg_light_path = os.path.join(output_dir, f"{base_name}_segmentation_light.png")
        success = sam.save_segmentation_only_result(
            input_path, masks, seg_light_path, alpha=0.4, use_smooth_edges=True
        )
        if success:
            results.append(("세그멘테이션 투명 결과", seg_light_path))

        # 3. 진한 세그멘테이션 결과 (투명도 0.8)
        seg_bold_path = os.path.join(output_dir, f"{base_name}_segmentation_bold.png")
        success = sam.save_segmentation_only_result(
            input_path, masks, seg_bold_path, alpha=0.8, use_smooth_edges=True
        )
        if success:
            results.append(("세그멘테이션 진한 결과", seg_bold_path))

    elif mode == "full":
        # 기존 전체 모드
        # 1. DETR 탐지 결과만 (bbox + 클래스명)
        detections = {
            'bboxes': bboxes,
            'scores': scores,
            'labels': labels
        }
        detr_vis = detector.visualize_detections(image, detections)
        detr_path = os.path.join(output_dir, f"{base_name}_1_detr_detection.png")
        cv2.imwrite(detr_path, detr_vis)
        results.append(("DETR 탐지 결과", detr_path))

        # 2. SAM 마스크만
        mask_vis = sam.visualize_masks(image, masks, alpha=0.6)
        mask_path = os.path.join(output_dir, f"{base_name}_2_sam_masks.png")
        cv2.imwrite(mask_path, mask_vis)
        results.append(("SAM 마스크", mask_path))

        # 3. 통합 결과 (bbox + mask + 클래스명)
        combined_vis = create_combined_visualization(
            image, bboxes, masks, scores, labels
        )
        combined_path = os.path.join(output_dir, f"{base_name}_3_combined_result.png")
        cv2.imwrite(combined_path, combined_vis)
        results.append(("통합 결과", combined_path))

        # 4. 세그멘테이션 전용 결과도 추가
        seg_only_path = os.path.join(output_dir, f"{base_name}_4_segmentation_only.png")
        success = sam.save_segmentation_only_result(
            input_path, masks, seg_only_path, alpha=0.6, use_smooth_edges=True
        )
        if success:
            results.append(("세그멘테이션 전용", seg_only_path))

        # 5. 개별 마스크 저장
        mask_dir = os.path.join(output_dir, f"{base_name}_individual_masks")
        mask_files = sam.save_masks(masks, mask_dir, f"{base_name}_mask")
        results.append((f"개별 마스크 ({len(mask_files)}개)", mask_dir))

    return results


def create_combined_visualization(image, bboxes, masks, scores, labels):
    """bbox + mask + 클래스명이 모두 포함된 통합 시각화"""
    # 먼저 마스크 오버레이
    result = sam.visualize_masks(image, masks, alpha=0.4)

    # bbox와 클래스명 추가
    colors = detector.generate_distinct_colors(len(bboxes))

    for i, (bbox, score) in enumerate(zip(bboxes, scores)):
        color = colors[i % len(colors)]

        # bbox 그리기 (더 굵게)
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)

        # 클래스명과 점수 표시
        if i < len(labels):
            class_name = detector.get_class_name(labels[i])
            label_text = f"{class_name}: {score:.2f}"
        else:
            label_text = f"Object_{i+1}: {score:.2f}"

        # 텍스트 배경 및 텍스트 그리기
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )

        text_x = x1
        text_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10

        # 텍스트 배경
        cv2.rectangle(
            result,
            (text_x, text_y - text_height - baseline),
            (text_x + text_width, text_y + baseline),
            color, -1
        )

        # 텍스트
        cv2.putText(
            result, label_text, (text_x, text_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )

    return result


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='DETR + SAM Instance Segmentation')
    parser.add_argument('--input', '-i', type=str, default='image_5.jpeg',
                       help='입력 이미지 경로')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                       help='DETR 탐지 신뢰도 임계값 (0.0-1.0)')
    parser.add_argument('--mode', choices=['full', 'detr-only', 'segmentation-only'],
                       default='full',
                       help='실행 모드: full (모든 결과), detr-only (DETR만), segmentation-only (세그멘테이션만)')
    parser.add_argument('--output', '-o', type=str, default='results',
                       help='결과 저장 디렉토리')
    parser.add_argument('--alpha', type=float, default=0.6,
                       help='세그멘테이션 마스크 투명도 (0.0-1.0, segmentation-only 모드용)')

    args = parser.parse_args()

    print("🤖 DETR + SAM Instance Segmentation")
    print("=" * 40)
    print(f"입력 이미지: {args.input}")
    print(f"신뢰도 임계값: {args.threshold}")
    print(f"실행 모드: {args.mode}")
    print(f"출력 디렉토리: {args.output}")
    if args.mode == 'segmentation-only':
        print(f"마스크 투명도: {args.alpha}")

    # 입력 파일 확인
    if not validate_file_exists(args.input):
        print("❌ 입력 이미지가 존재하지 않습니다.")
        return

    # 결과 디렉토리 생성
    output_dir = create_output_directory(args.output)

    # 모델 파일 확인
    if not check_model_files():
        return

    # 디바이스 설정
    device = get_optimal_device()
    print(f"🖥️ 사용 디바이스: {device}")

    try:
        # 모델 초기화
        detr_model, sam_predictor = initialize_models(device)

        if detr_model is None:
            print("❌ DETR 모델 초기화 실패")
            return

        # 실행 모드에 따른 파이프라인 실행
        if args.mode == 'detr-only' or sam_predictor is None:
            if sam_predictor is None:
                print("⚠️ SAM 모델 초기화 실패, DETR만 실행합니다.")

            success = run_detection_only(
                args.input, detr_model, args.threshold, output_dir
            )
        else:
            success = run_full_pipeline(
                args.input, detr_model, sam_predictor,
                args.threshold, output_dir, args.mode
            )

        if success:
            print(f"\n🎉 Instance Segmentation 완료!")
            print(f"📂 결과를 확인하세요: {output_dir}")
        else:
            print(f"\n❌ 처리 중 오류가 발생했습니다.")

    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
