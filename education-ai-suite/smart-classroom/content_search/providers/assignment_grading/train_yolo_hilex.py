from ultralytics import YOLO
from pathlib import Path
import shutil

HILEX_DIR = Path("C:/Users/user/jianfeng/EDU-AI/PR/HiLEx")
YOLO_DATA_DIR = HILEX_DIR / "HiLex_Yolo_Format"
DATA_YAML = YOLO_DATA_DIR / "data.yaml"

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "models" / "yolo_hilex"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fix_data_yaml_paths():
    print("\n[1/4] 修复data.yaml中的路径...")

    fixed_yaml_path = OUTPUT_DIR / "data.yaml"

    with open(DATA_YAML, 'r') as f:
        content = f.read()

    fixed_content = content.replace(
        "train: ../train/images",
        f"train: {YOLO_DATA_DIR / 'train' / 'images'}"
    ).replace(
        "val: ../valid/images",
        f"val: {YOLO_DATA_DIR / 'valid' / 'images'}"
    ).replace(
        "test: ../test/images",
        f"test: {YOLO_DATA_DIR / 'test' / 'images'}"
    )

    with open(fixed_yaml_path, 'w') as f:
        f.write(fixed_content)

    print(f"  已保存到: {fixed_yaml_path}")

    return fixed_yaml_path


def train_yolo11n(data_yaml, epochs=100, imgsz=640, batch=16, device='cpu'):
    print(f"\n[2/4] 加载YOLO11n预训练模型...")

    model = YOLO('yolo11n.pt')

    print(f"\n[3/4] 开始训练...")
    print(f"  数据集: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  图片尺寸: {imgsz}")
    print(f"  Batch: {batch}")
    print(f"  设备: {device}")
    print(f"  输出: {OUTPUT_DIR}")

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(OUTPUT_DIR),
        name='yolo11n_hilex',
        exist_ok=True,
        patience=10,
        save=True,
        save_period=10,
        cache=False,
        workers=4,
        verbose=True
    )

    print(f"\n[4/4] 训练完成！")
    print(f"  最佳权重: {OUTPUT_DIR / 'yolo11n_hilex' / 'weights' / 'best.pt'}")

    return results


def validate_model(data_yaml, weights_path):
    print("\n[验证] 在测试集上评估模型...")

    model = YOLO(str(weights_path))

    results = model.val(
        data=str(data_yaml),
        split='test'
    )

    print(f"\n验证结果:")
    print(f"  mAP@50: {results.box.map50:.3f}")
    print(f"  mAP@50-95: {results.box.map:.3f}")

    return results


def export_to_openvino(weights_path):
    print("\n[导出] 转换为OpenVINO格式...")

    model = YOLO(str(weights_path))

    ov_model_path = model.export(
        format='openvino',
        imgsz=640,
        half=True
    )

    print(f"  OpenVINO模型: {ov_model_path}")

    return ov_model_path


def check_gpu():
    import torch

    print("\n[GPU检测]")
    print(f"  PyTorch版本: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"  CUDA可用: True")
        print(f"  GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"  GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return '0'

    try:
        import torch_directml
        dml = torch_directml.device()
        print(f"  DirectML可用: True")
        print(f"  DML设备: {dml}")
        print(f"  注意: Ultralytics YOLO暂不支持DirectML，回退到CPU")
        print(f"  推荐: 训练后用OpenVINO在Intel GPU上推理加速")
        return 'cpu'
    except ImportError:
        print("  DirectML: 未安装")

    try:
        import intel_extension_for_pytorch as ipex
        print(f"  Intel Extension: {ipex.__version__}")
        if ipex.xpu.is_available():
            print(f"  Intel XPU可用: True")
            print(f"  XPU设备数: {ipex.xpu.device_count()}")
            return 'xpu'
    except ImportError:
        pass

    print("  GPU不可用，使用CPU训练")
    return 'cpu'


def main():
    print("="*80)
    print("YOLO11训练脚本 - HiLEx数据集（试卷布局检测）")
    print("="*80)

    device = check_gpu()

    print(f"\n数据集信息:")
    print(f"  位置: {HILEX_DIR}")
    print(f"  训练集: 1378张图片")
    print(f"  验证集: 388张图片")
    print(f"  测试集: 199张图片")
    print(f"  类别: Answer_Block, Description, Instruction,")
    print(f"        Question_Answer_Block, Question_Block, Question_Paper_Area")

    data_yaml = fix_data_yaml_paths()

    batch_size = 32 if device in ['0', 'xpu'] else 16

    print("\n" + "="*80)
    print("GPU训练说明:")
    print("  当前设备:", device)
    if device == 'cpu':
        print("  Intel Arc GPU无法直接用于PyTorch训练（Ultralytics不支持DirectML）")
        print("  建议:")
        print("    1. CPU快速测试: epochs=20 (约10-15分钟)")
        print("    2. 云端GPU训练: 上传train_colab.ipynb到Google Colab (免费T4 GPU)")
        print("    3. 完整CPU训练: epochs=100 (约1-2小时)")
        print("    4. 训练后用OpenVINO在Intel GPU上推理（已支持加速）")
    print("="*80)

    epochs_choice = input("\n选择训练epochs (20=快速测试, 100=完整训练, 回车=20): ").strip()
    epochs = int(epochs_choice) if epochs_choice else 20

    train_yolo11n(
        data_yaml=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        device=device
    )

    best_weights = OUTPUT_DIR / 'yolo11n_hilex' / 'weights' / 'best.pt'

    if best_weights.exists():
        validate_model(data_yaml, best_weights)

        export_to_openvino(best_weights)

        print("\n" + "="*80)
        print("全部完成！")
        print("="*80)
        print("\n下一步:")
        print("  1. 检查训练曲线: tensorboard --logdir", OUTPUT_DIR / 'yolo11n_hilex')
        print("  2. 测试推理:")
        print(f"     from ultralytics import YOLO")
        print(f"     model = YOLO('{best_weights}')")
        print(f"     results = model('test_image.jpg')")
        print("  3. 部署到B580:")
        print(f"     使用OpenVINO模型: {OUTPUT_DIR / 'yolo11n_hilex' / 'weights' / 'best_openvino_model'}")


if __name__ == '__main__':
    main()
