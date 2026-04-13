# CASE 2 — K3: PPE Compliance Monitoring Using Computer Vision

Proyek ini dibuat untuk memenuhi kebutuhan berikut:
- mendeteksi pekerja dari CCTV/video/webcam,
- mengenali PPE (helmet, vest, gloves, safety shoes, goggles),
- menilai kepatuhan PPE berdasarkan standar per zona,
- memberikan notifikasi pelanggaran real-time,
- membuat laporan K3 otomatis.

## 1) Teknologi yang digunakan
- **Python**
- **Ultralytics YOLO** untuk training dan inference object detection
- **OpenCV** untuk pengolahan video/CCTV
- **Pandas + OpenPyXL** untuk laporan CSV/XLSX
- **HTML report** untuk rekap cepat yang bisa dibuka di browser

## 2) Struktur file
```text
ppe_k3_case2/
│
├─ ppe_k3_monitoring.py
├─ requirements.txt
├─ ppe_policy_example.json
├─ dataset_example.yaml
├─ sample_expected_output.txt
└─ README.md
```

## 3) Hal yang sangat penting sebelum menjalankan
Agar sistem **benar-benar bisa mendeteksi seluruh PPE**, Anda harus menyiapkan atau melatih **model custom** yang memiliki class berikut:
- person
- helmet
- vest
- gloves
- safety_shoes
- goggles

Kalau Anda hanya memakai model umum seperti `yolo11n.pt`, biasanya model itu hanya bagus untuk mendeteksi `person`, bukan seluruh PPE. Jadi:
- **Deteksi pekerja** akan berjalan
- **Penilaian PPE penuh** baru akurat setelah Anda melatih model PPE sendiri

## 4) Persiapan dari nol (Windows, super detail)

### Langkah A — Install Python
1. Download Python dari situs resmi Python.
2. Saat instalasi, **centang**: `Add Python to PATH`.
3. Selesaikan instalasi.
4. Buka **Command Prompt** atau **PowerShell**.
5. Cek apakah Python sudah terbaca:
   ```powershell
   python --version
   ```

### Langkah B — Buat folder proyek
Misalnya Anda mau simpan di drive D:
```powershell
D:
mkdir finalpro
cd finalpro
```

Kalau Anda memakai **Git Bash**, pindah drive D: dengan cara:
```bash
cd /d/finalpro
```
Bukan mengetik `D:` seperti di CMD.

### Langkah C — Salin file proyek
Masukkan file-file berikut ke folder proyek:
- `ppe_k3_monitoring.py`
- `requirements.txt`
- `ppe_policy_example.json`
- `dataset_example.yaml`
- `README.md`

### Langkah D — Buat virtual environment
Sangat disarankan supaya library proyek tidak bercampur dengan Python global.

Di PowerShell/CMD:
```powershell
python -m venv .venv
```

Aktifkan environment:
- **PowerShell**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **CMD**
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **Git Bash**
  ```bash
  source .venv/Scripts/activate
  ```

Jika berhasil, biasanya terminal akan menampilkan `(.venv)` di depannya.

### Langkah E — Upgrade pip
```powershell
python -m pip install --upgrade pip
```

### Langkah F — Install semua library
```powershell
pip install -r requirements.txt
```

## 5) Persiapan dataset training

Agar sistem bisa mendeteksi PPE secara penuh, Anda perlu dataset berlabel object detection.

### Struktur dataset yang disarankan
```text
finalpro/
├─ datasets/
│  └─ ppe_dataset/
│     ├─ images/
│     │  ├─ train/
│     │  ├─ val/
│     │  └─ test/
│     └─ labels/
│        ├─ train/
│        ├─ val/
│        └─ test/
├─ dataset_example.yaml
└─ ppe_k3_monitoring.py
```

### Format label
Gunakan format YOLO `.txt` per gambar.

Contoh satu baris label YOLO:
```text
class_id x_center y_center width height
```
Semua nilai koordinat dinormalisasi 0 sampai 1.

### Mapping class
Pastikan class di dataset sama dengan YAML:
- 0 = person
- 1 = helmet
- 2 = vest
- 3 = gloves
- 4 = safety_shoes
- 5 = goggles

### Ganti isi `dataset_example.yaml`
Sesuaikan path dengan lokasi dataset Anda.

Contoh:
```yaml
path: ./datasets/ppe_dataset
train: images/train
val: images/val
test: images/test

names:
  0: person
  1: helmet
  2: vest
  3: gloves
  4: safety_shoes
  5: goggles
```

## 6) Training model custom PPE

### Perintah training dasar
```powershell
python ppe_k3_monitoring.py train --data dataset_example.yaml --model yolo11n.pt --epochs 50 --imgsz 960 --batch 8 --project runs/ppe_train --name ppe_case2
```

### Penjelasan parameter
- `--data`: file YAML dataset
- `--model`: model dasar YOLO
- `--epochs`: jumlah epoch training
- `--imgsz`: ukuran gambar training
- `--batch`: batch size
- `--project`: folder hasil training
- `--name`: nama eksperimen

### Setelah training selesai
Biasanya weights terbaik ada di:
```text
runs/ppe_train/ppe_case2/weights/best.pt
```

Itulah file model yang nanti dipakai saat monitoring.

## 7) Menjalankan monitoring realtime

### A. Pakai webcam
```powershell
python ppe_k3_monitoring.py monitor --source 0 --weights runs/ppe_train/ppe_case2/weights/best.pt --policy ppe_policy_example.json --zone-name general_factory --camera-name gate_cam_1 --output-dir output
```

### B. Pakai video file
```powershell
python ppe_k3_monitoring.py monitor --source sample_video.mp4 --weights runs/ppe_train/ppe_case2/weights/best.pt --policy ppe_policy_example.json --zone-name general_factory --camera-name cctv_line_1 --output-dir output
```

## 8) Parameter penting saat monitoring

### Confidence threshold
Kalau banyak false positive, naikkan confidence:
```powershell
python ppe_k3_monitoring.py monitor --source sample_video.mp4 --weights runs/ppe_train/ppe_case2/weights/best.pt --policy ppe_policy_example.json --zone-name general_factory --camera-name cctv_1 --output-dir output --conf 0.45
```

### Preprocessing untuk kondisi sulit
Untuk kondisi gelap, asap ringan, kontras rendah:
```powershell
python ppe_k3_monitoring.py monitor --source sample_video.mp4 --weights runs/ppe_train/ppe_case2/weights/best.pt --policy ppe_policy_example.json --zone-name general_factory --camera-name cctv_1 --output-dir output --preprocess all
```

Pilihan preprocessing:
- `none`
- `gamma`
- `clahe`
- `dehaze`
- `all`

### Kurangi notifikasi palsu
Pelanggaran baru dikonfirmasi jika terjadi beberapa frame berturut-turut.
Contoh 8 frame:
```powershell
python ppe_k3_monitoring.py monitor --source sample_video.mp4 --weights runs/ppe_train/ppe_case2/weights/best.pt --policy ppe_policy_example.json --zone-name general_factory --camera-name cctv_1 --output-dir output --violation-frames-threshold 8
```

### Matikan beep
```powershell
python ppe_k3_monitoring.py monitor --source sample_video.mp4 --weights runs/ppe_train/ppe_case2/weights/best.pt --policy ppe_policy_example.json --zone-name general_factory --camera-name cctv_1 --output-dir output --disable-beep
```

### Jangan simpan video anotasi
```powershell
python ppe_k3_monitoring.py monitor --source sample_video.mp4 --weights runs/ppe_train/ppe_case2/weights/best.pt --policy ppe_policy_example.json --zone-name general_factory --camera-name cctv_1 --output-dir output --no-save-video
```

## 9) Kebijakan PPE per zona industri
File `ppe_policy_example.json` bisa Anda ubah sesuai kebutuhan perusahaan.

Contoh:
- `general_factory`: helmet, vest, goggles, safety_shoes
- `welding_area`: helmet, gloves, goggles, safety_shoes
- `chemical_area`: helmet, goggles, gloves, safety_shoes, vest

Kalau kamera mengawasi area pengelasan, jalankan:
```powershell
python ppe_k3_monitoring.py monitor --source sample_video.mp4 --weights runs/ppe_train/ppe_case2/weights/best.pt --policy ppe_policy_example.json --zone-name welding_area --camera-name welding_cam_1 --output-dir output
```

## 10) Output yang dihasilkan
Setelah sistem berjalan, folder output akan berisi:

### A. Video anotasi
```text
output/videos/
```
Berisi video hasil dengan bounding box, status compliant/violation, dan ringkasan realtime.

### B. Evidence pelanggaran
```text
output/evidences/
```
Berisi crop gambar pekerja saat pelanggaran PPE terdeteksi.

### C. Laporan CSV
```text
output/reports/violation_events.csv
```
Berisi log semua event pelanggaran.

### D. Laporan Excel
```text
output/reports/k3_summary.xlsx
```
Berisi sheet rekap per jenis pelanggaran, tanggal, kamera, zona, dan track pekerja.

### E. Laporan HTML
```text
output/reports/k3_summary.html
```
Bisa dibuka di browser untuk ringkasan cepat.

### F. Log JSONL
```text
output/logs/violation_events.jsonl
```
Cocok untuk integrasi ke sistem lain.

## 11) Cara kerja logika sistem
1. Video dibaca frame per frame.
2. Frame dipreprocess untuk membantu kondisi low light/debu/asap ringan.
3. YOLO mendeteksi object pada frame.
4. Sistem memisahkan deteksi `person` dan deteksi PPE.
5. PPE di-associate ke setiap person berdasarkan posisi spasial dalam bounding box orang.
6. Sistem membaca policy per zona.
7. Jika ada PPE wajib yang hilang selama beberapa frame berturut-turut, sistem menandainya sebagai pelanggaran.
8. Sistem menyimpan bukti gambar, menulis event ke CSV/JSONL, dan membuat rekap laporan otomatis.

## 12) Contoh hasil terminal
```text
[INFO] Model classes tersedia: ['goggles', 'gloves', 'helmet', 'person', 'safety_shoes', 'vest']
[INFO] Required PPE untuk zone 'general_factory': ['helmet', 'vest', 'goggles', 'safety_shoes']
[INFO] Annotated video akan disimpan ke: output/videos/annotated_20260409_120000.mp4
[INFO] Tekan tombol 'q' untuk keluar.
[2026-04-09 12:01:14] VIOLATION | camera=cctv_line_1 | track=7 | missing=['helmet', 'goggles']
[2026-04-09 12:01:27] VIOLATION | camera=cctv_line_1 | track=11 | missing=['safety_shoes']
```

## 13) Contoh skenario lengkap dari awal sampai akhir
Misalnya:
- folder proyek: `D:\finalpro`
- dataset sudah siap di `D:\finalpro\datasets\ppe_dataset`
- Anda ingin train lalu jalankan monitoring video `sample_video.mp4`

Urutannya:

### 1. Buka PowerShell
### 2. Masuk folder proyek
```powershell
D:
cd finalpro
```

### 3. Buat environment
```powershell
python -m venv .venv
```

### 4. Aktifkan environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install library
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Cek file YAML dataset
Pastikan `dataset_example.yaml` menunjuk ke folder dataset yang benar.

### 7. Training model
```powershell
python ppe_k3_monitoring.py train --data dataset_example.yaml --model yolo11n.pt --epochs 50 --imgsz 960 --batch 8 --project runs/ppe_train --name ppe_case2
```

### 8. Setelah selesai, ambil weights terbaik
Lokasi:
```text
runs/ppe_train/ppe_case2/weights/best.pt
```

### 9. Jalankan monitoring
```powershell
python ppe_k3_monitoring.py monitor --source sample_video.mp4 --weights runs/ppe_train/ppe_case2/weights/best.pt --policy ppe_policy_example.json --zone-name general_factory --camera-name cctv_line_1 --output-dir output --preprocess all --conf 0.35 --violation-frames-threshold 5
```

### 10. Lihat output
Cek folder:
```text
output/
```

## 14) Troubleshooting

### `ModuleNotFoundError`
Berarti library belum terinstall di environment aktif.
Solusi:
```powershell
pip install -r requirements.txt
```

### Kamera tidak terbuka
Pastikan webcam/camera tidak dipakai aplikasi lain.
Coba ganti `--source 0` menjadi file video dulu untuk test.

### Tidak ada PPE yang terdeteksi
Biasanya karena:
- weights belum model PPE custom,
- class dataset tidak sesuai,
- label dataset salah,
- ukuran object terlalu kecil,
- resolusi CCTV terlalu rendah.

### Banyak false positive
Coba:
- naikkan `--conf` ke `0.45` atau `0.5`
- perbesar kualitas dataset
- tambah data kondisi gelap/asap/occlusion saat training
- gunakan CCTV resolusi lebih baik

### Gloves atau safety shoes sulit terdeteksi
Itu memang class yang relatif sulit karena:
- ukurannya kecil,
- sering tertutup mesin,
- sering blur pada CCTV.
Solusi terbaik: dataset yang lebih banyak, gambar lebih tajam, sudut kamera lebih tepat, dan train dengan imgsz lebih tinggi.

## 15) Batasan sistem
Sistem ini **membantu auditor K3**, tetapi **tidak menggantikan auditor K3**. Akurasi dipengaruhi oleh:
- kualitas CCTV,
- posisi kamera,
- pencahayaan,
- debu/asap,
- occlusion oleh mesin,
- kualitas dataset training,
- ketepatan standard policy per industri.

## 16) Saran pengembangan lanjutan
Jika ingin lebih kuat lagi, Anda bisa menambahkan:
- multi-camera dashboard,
- database PostgreSQL/MySQL,
- dashboard Streamlit/Flask,
- kirim alert ke Telegram/WhatsApp/Email,
- face blurring untuk privasi,
- area polygon per zona,
- per-person compliance score,
- integration ke BI dashboard.

---

Kalau Anda sudah punya dataset atau weights, jalankan bagian **monitor**.
Kalau belum punya weights PPE, jalankan bagian **train** terlebih dahulu.
