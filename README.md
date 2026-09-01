# WavLM-Conformer Baseline with Contrastive Segment Loss for Speech Editing Detection (OOP Edition)

Hệ thống baseline cho bài toán **Speech Editing Detection & Localization** sử dụng **WavLM-Large SSL Encoder**, **Conformer Backend** và **Contrastive Segment Loss**, được thiết kế tinh gọn theo chuẩn **Lập trình Hướng đối tượng (OOP)**.

---

## 1. Cấu trúc thư mục tối giản

```
/GuestShare_NAS/WorkingSpace/Personal/nghiadq/NCKH/WavLM/
├── config/
│   └── baseline.yaml                  # File cấu hình YAML chuẩn hóa
├── src/
│   ├── __init__.py                    # Export các class chính
│   ├── dataset.py                     # [OOP Data] PartialSpoofDataset & PartialSpoofDataModule
│   ├── model.py                       # [OOP Model] WavLMEncoder, ConformerEncoder, PoolHead, WavLMConformer
│   ├── criterion.py                   # [OOP Loss & Metrics] MaskedCrossEntropyLoss, ContrastiveSegmentLoss, TotalLoss, EERMetric, F1Metric
│   └── pipeline.py                    # [OOP Lightning Module] WavLMConformerPipeline
├── train.py                           # [Entrypoint] CLI Runner + tích hợp sẵn sanity check (--test-loss)
├── data/                              # Tài liệu & PDF tham khảo
├── logs/                              # Quản lý lượt chạy tự động (run1, run2, ...)
└── README.md
```

---

## 2. Thông tin Dữ liệu (PartialSpoof Protocol)

Dữ liệu được phân chia cố định theo giao thức chuẩn (Official Benchmark Protocol) để đảm bảo không bị rò rỉ người nói (speaker leakage):

| Phân vùng (Split) | Thư mục dữ liệu | File nhãn tương ứng | Số lượng file wav | Tỷ lệ trong dataset |
|---|---|---|---|---|
| **Train** (Huấn luyện) | `data/PartialSpoof/train/` | `train_seglab_0.16.npy` | **25,380 mẫu** | ~20.9% |
| **Validation / Dev** (Thẩm định) | `data/PartialSpoof/dev/` | `dev_seglab_0.16.npy` | **24,844 mẫu** | ~20.5% |
| **Test / Eval** (Đánh giá) | `data/PartialSpoof/eval/` | `eval_seglab_0.16.npy` | **71,237 mẫu** | ~58.6% |
| **Tổng cộng** | | | **121,461 mẫu** | **100%** |

---

## 3. Quản lý Logs & Lượt chạy (`run1`, `run2`, ...)

Hệ thống tự động tạo thư mục lượt chạy tăng dần bên trong `logs/wavlm_conformer_contrastive/`:
```
logs/wavlm_conformer_contrastive/
├── run1/
│   ├── train.log                      # File lưu toàn bộ log terminal & TQDM progress bar
│   ├── hparams.yaml                   # File lưu toàn bộ cấu hình tham số của lượt run
│   ├── test_results.txt               # Kết quả đánh giá cuối (Loss, EER, Acc, F1)
│   └── checkpoints/                   # Checkpoint mô hình tốt nhất
│       └── best_eer_epoch=XX_val_eer=YY.YY.ckpt
├── run2/
└── ...
```

---

## 4. Hướng dẫn Chạy Chi tiết

### Bước 0: Môi trường thực thi
Các lệnh huấn luyện cần được chạy trong môi trường **Docker (`pytorch_nghiadq`)** và môi trường **Conda (`sal`)**:
```bash
# Cách 1: Vào trực tiếp container
docker exec -it pytorch_nghiadq bash
conda activate sal
cd /GuestShare_NAS/WorkingSpace/Personal/nghiadq/NCKH/WavLM

# Cách 2: Chạy trực tiếp từ máy host
docker exec -it pytorch_nghiadq /GuestShare_NAS/WorkingSpace/Personal/nghiadq/miniconda3/envs/sal/bin/python /GuestShare_NAS/WorkingSpace/Personal/nghiadq/NCKH/WavLM/train.py <arguments>
```

---

### 1. Kiểm tra nhanh Loss & Gradient Flow (Sanity Check)
```bash
python3 train.py --test-loss
```

---

### 2. Chạy thử nghiệm nhanh (Debug Mode)
Giới hạn 50 mẫu mỗi split và chạy 2 epochs:
```bash
python3 train.py --debug --max_samples 50 --batch_size 4 --max_epochs 2
```

---

### 3. Huấn luyện Full Toàn bộ Dữ liệu (Full Training)

#### A. Chạy thông thường (GPU mặc định)
```bash
python3 train.py --config config/baseline.yaml
```

#### B. Huấn luyện Full có giới hạn CPU tối đa 100% (1 Core)
Khóa tiến trình chỉ chạy trên 1 core CPU duy nhất để tránh chiếm dụng tài nguyên máy chủ:
```bash
# Cách dùng taskset (Khuyên dùng):
taskset -c 0 python3 train.py --config config/baseline.yaml

# Hoặc giới hạn số thread của PyTorch/OpenMP:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_NUM_THREADS=1 python3 train.py --config config/baseline.yaml
```

#### C. Chỉ định GPU cụ thể (ví dụ GPU 0 hoặc GPU 1)
```bash
CUDA_VISIBLE_DEVICES=0 taskset -c 0 python3 train.py --config config/baseline.yaml
```

#### D. Chạy ngầm trong background (với nohup)
```bash
CUDA_VISIBLE_DEVICES=0 taskset -c 0 nohup python3 train.py --config config/baseline.yaml > run_stdout.log 2>&1 &
```
*(Tiến trình vẫn tự động ghi log chi tiết vào `logs/wavlm_conformer_contrastive/runX/train.log`)*

---

### 4. Đánh giá Mô hình từ Checkpoint (Test Only)
```bash
python3 train.py --config config/baseline.yaml --test_only --ckpt_path logs/wavlm_conformer_contrastive/run1/checkpoints/best_eer_epoch=29_val_eer=15.20.ckpt
```
