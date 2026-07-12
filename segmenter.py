# segmenter.py
import os
import sys
import torch
import torchvision
import numpy as np
import cv2
from PIL import Image
from transformers import (
    DFineForObjectDetection,
    RTDetrV2ForObjectDetection,
    RTDetrImageProcessor,
)
import argparse
from pathlib import Path
import json
from datetime import datetime

class DocumentSegmenterEngine:
    def __init__(self, default_device='cpu'):
        # == Device configuration ==
        self.device = default_device

        # == Model configurations ==
        self.MODELS = {
            "Docling Layout Egret XLarge": {
                "path": "ds4sd/docling-layout-egret-xlarge",
                "model_class": DFineForObjectDetection
            },
            "Docling Layout Egret Large": {
                "path": "ds4sd/docling-layout-egret-large",
                "model_class": DFineForObjectDetection
            },
            "Docling Layout Egret Medium": {
                "path": "ds4sd/docling-layout-egret-medium", 
                "model_class": DFineForObjectDetection
            },
            "Docling Layout Heron 101": {
                "path": "ds4sd/docling-layout-heron-101",
                "model_class": RTDetrV2ForObjectDetection
            },
            "Docling Layout Heron": {
                "path": "ds4sd/docling-layout-heron",
                "model_class": RTDetrV2ForObjectDetection
            }
        }

        # == Class mappings ==
        self.CLASSES_MAP = {
            0: "Caption", 1: "Footnote", 2: "Formula", 3: "List-item",
            4: "Page-footer", 5: "Page-header", 6: "Picture", 7: "Section-header",
            8: "Table", 9: "Text", 10: "Title", 11: "Document Index",
            12: "Code", 13: "Checkbox-Selected", 14: "Checkbox-Unselected", 
            15: "Form", 16: "Key-Value Region",
        }

        # == Global model variables (Sekarang menjadi State Instance) ==
        self.current_model = None
        self.current_processor = None
        self.current_model_name = None

    def colormap(self, N=256, normalized=False):
        """Generate dynamic colormap."""
        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)

        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << (7 - j))
                g = g | (bitget(c, 1) << (7 - j))
                b = b | (bitget(c, 2) << (7 - j))
                c = c >> 3
            cmap[i] = np.array([r, g, b])
        
        if normalized:
            cmap = cmap.astype(np.float32) / 255.0
        return cmap

    def iomin(self, box1, box2):
        """Intersection over Minimum (IoMin)."""
        x1 = torch.max(box1[:, 0], box2[:, 0])
        y1 = torch.max(box1[:, 1], box2[:, 1])
        x2 = torch.min(box1[:, 2], box2[:, 2])
        y2 = torch.min(box1[:, 3], box2[:, 3])
        inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        min_area = torch.min(box1_area, box2_area)
        
        return inter_area / min_area

    def nms_custom(self, boxes, scores, iou_threshold=0.5):
        """Custom NMS implementation using IoMin."""
        keep = []
        _, order = scores.sort(descending=True)

        while order.numel() > 0:
            i = order[0]
            keep.append(i.item())

            if order.numel() == 1:
                break

            box_i = boxes[i].unsqueeze(0)
            rest = order[1:]
            ious = self.iomin(box_i, boxes[rest])

            mask = (ious <= iou_threshold)
            order = order[1:][mask]

        return torch.tensor(keep, dtype=torch.long)

    def load_model(self, model_name):
        """Load the selected model."""
        if self.current_model_name == model_name and self.current_model is not None:
            print(f"✅ Model '{model_name}' already loaded")
            return True
        
        try:
            model_info = self.MODELS[model_name]
            model_path = model_info["path"]
            model_class = model_info["model_class"]
            
            print(f"📥 Loading model: {model_name}")
            print(f"   From: {model_path}")
            
            processor = RTDetrImageProcessor.from_pretrained(model_path)
            model = model_class.from_pretrained(model_path)
            model = model.to(self.device)
            model.eval()
            
            self.current_processor = processor
            self.current_model = model
            self.current_model_name = model_name
            
            print(f"✅ Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def process_image(self, input_path, model_name, conf_threshold=0.6, iou_threshold=0.5, nms_method="Standard IoU"):
        """Process image with document layout detection."""
        
        # Check if input file exists
        if not os.path.exists(input_path):
            print(f"❌ File not found: {input_path}")
            return None
        
        # Load model
        if not self.load_model(model_name):
            return None
        
        try:
            path = Path(input_path)
            ext = path.suffix.lower()
            input_img = None

            if ext == ".pdf":
                from pdf2image import convert_from_path

                pages = convert_from_path(
                    input_path,
                    dpi=300,
                    first_page=1,
                    last_page=1
                )

                if not pages:
                    print(f"❌ Error: Could not convert PDF: {input_path}")
                    return

                input_img = pages[0]

            elif ext in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                input_img = Image.open(input_path).convert("RGB")
            
            if input_img.mode != 'RGB':
                input_img = input_img.convert('RGB')
            
            print(f"   Image size: {input_img.size}")
            
            # Process with model
            print("🔍 Running layout detection...")
            inputs = self.current_processor(images=[input_img], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.current_model(**inputs)
            
            # Post-process results
            results = self.current_processor.post_process_object_detection(
                outputs,
                target_sizes=torch.tensor([input_img.size[::-1]]),
                threshold=conf_threshold,
            )
            
            if not results or len(results) == 0:
                print("ℹ️ No detections found.")
                return None
                
            result = results[0]
            boxes = result["boxes"]
            scores = result["scores"] 
            labels = result["labels"]
            
            if len(boxes) == 0:
                print(f"ℹ️ No detections above threshold {conf_threshold:.2f}.")
                return None
            
            # Apply NMS
            if iou_threshold < 1.0:
                if nms_method == "Custom IoMin":
                    keep_indices = self.nms_custom(boxes=boxes, scores=scores, iou_threshold=iou_threshold)
                else:
                    keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold)
                
                boxes = boxes[keep_indices]
                scores = scores[keep_indices]
                labels = labels[keep_indices]
            
            # === LOGIKA FALLBACK TITLE DARI SECTION-HEADER ===
            # Cek jika tidak ada label 10 ("Title") di dalam hasil deteksi tensor
            if not (labels == 10).any():
                # Cari indeks dari elemen yang berlabel 7 ("Section-header")
                section_indices = (labels == 7).nonzero(as_tuple=True)[0]
                
                if len(section_indices) > 0:
                    print("⚠️ 'Title' tidak ditemukan. Mencari 'Section-header' dengan area paling luas...")
                    
                    # Ambil koordinat box untuk section-header: [x1, y1, x2, y2]
                    section_boxes = boxes[section_indices]
                    
                    # Hitung lebar (w = x2 - x1) dan tinggi (h = y2 - y1)
                    widths = section_boxes[:, 2] - section_boxes[:, 0]
                    heights = section_boxes[:, 3] - section_boxes[:, 1]
                    
                    # Hitung luas area (lebar * tinggi)
                    areas = widths * heights
                    
                    # Temukan posisi indeks dengan nilai luas terbesar
                    largest_sub_idx = torch.argmax(areas)
                    highest_global_idx = section_indices[largest_sub_idx]
                    
                    # Ubah labelnya menjadi 10 ("Title")
                    labels[highest_global_idx] = 10
                    
                    luas_terpilih = areas[largest_sub_idx].item()
                    print(f"📌 Berhasil mengubah 'Section-header' paling luas (Luas: {luas_terpilih:.0f} px) "
                          f"di posisi (x={boxes[highest_global_idx, 0]:.0f}, y={boxes[highest_global_idx, 1]:.0f}) menjadi 'Title'.")
            # =================================================================

            print(f"✅ Found {len(boxes)} detections")
            
            return {
                'image': input_img,
                'boxes': boxes,
                'scores': scores,
                'labels': labels,
                'pdf_path': input_path,
                'model_name': model_name,
                'conf_threshold': conf_threshold,
                'iou_threshold': iou_threshold,
                'nms_method': nms_method
            }
                
        except Exception as e:
            print(f"❌ Processing error: {str(e)}")
            return None

    def save_detection_results(self, result_data, output_dir="output"):
        """Save detection results to files."""
        
        if result_data is None:
            return
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get base filename
        input_path = Path(result_data['pdf_path'])
        base_name = input_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save visualized image
        print(f"💾 Saving visualization...")
        vis_image = self.visualize_bbox(
            result_data['image'],
            result_data['boxes'],
            result_data['labels'],
            result_data['scores'],
            self.CLASSES_MAP,
            alpha=0.3,
            show_labels=True
        )
        vis_path = output_path / f"{base_name}_detection_{timestamp}.png"
        Image.fromarray(vis_image).save(vis_path)
        print(f"   Visualization saved: {vis_path}")
        
        # 2. Save detection metadata (JSON)
        print(f"💾 Saving metadata...")
        detections = []
        for i in range(len(result_data['boxes'])):
            box = result_data['boxes'][i]
            if torch.is_tensor(box):
                box = box.cpu().numpy()
            label = result_data['labels'][i]
            if torch.is_tensor(label):
                label = label.item()
            score = result_data['scores'][i]
            if torch.is_tensor(score):
                score = score.item()
            
            detections.append({
                'box': [float(x) for x in box],
                'label': int(label),
                'label_name': self.CLASSES_MAP.get(int(label), f"unknown_{int(label)}"),
                'confidence': float(score)
            })
        
        metadata = {
            'image': str(input_path),
            'image_size': result_data['image'].size,
            'model': result_data['model_name'],
            'parameters': {
                'conf_threshold': result_data['conf_threshold'],
                'iou_threshold': result_data['iou_threshold'],
                'nms_method': result_data['nms_method']
            },
            'total_detections': len(detections),
            'detections': detections,
            'timestamp': timestamp
        }
        
        json_path = output_path / f"{base_name}_metadata_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"   Metadata saved: {json_path}")
        
        # 3. Save text summary
        print(f"💾 Saving summary...")
        summary_path = output_path / f"{base_name}_summary_{timestamp}.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("DOCUMENT LAYOUT ANALYSIS RESULTS\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Image: {input_path}\n")
            f.write(f"Image Size: {result_data['image'].size}\n")
            f.write(f"Model: {result_data['model_name']}\n")
            f.write(f"Parameters: Conf={result_data['conf_threshold']}, IoU={result_data['iou_threshold']}, NMS={result_data['nms_method']}\n")
            f.write(f"Total Detections: {len(detections)}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("DETECTION DETAILS (by reading order)\n")
            f.write("-" * 70 + "\n\n")
            
            # Group by type
            type_counts = {}
            for det in detections:
                label_name = det['label_name']
                type_counts[label_name] = type_counts.get(label_name, 0) + 1
            
            f.write("DETECTION SUMMARY BY TYPE:\n")
            for label_name, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {label_name}: {count}\n")
            f.write("\n")
            
            f.write("DETAILED DETECTIONS (sorted by position):\n")
            f.write("-" * 70 + "\n")
            
            # Sort by y-position (top to bottom) for reading order
            sorted_detections = sorted(
                enumerate(detections),
                key=lambda x: x[1]['box'][1]  # Sort by y_min
            )
            
            for idx, (orig_idx, det) in enumerate(sorted_detections, 1):
                box = det['box']
                f.write(f"{idx:3d}. {det['label_name']:<20} ")
                f.write(f"Conf: {det['confidence']:.3f}  ")
                f.write(f"Box: [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]\n")
        
        print(f"   Summary saved: {summary_path}")
        
        # 4. Extract and save each region
        print(f"💾 Extracting regions using Graph-based Topological Sort (Human Reading Order)...")
        regions_dir = output_path / f"{base_name}_regions_{timestamp}"
        regions_dir.mkdir(parents=True, exist_ok=True)
        
        # ==================== PENDEKATAN BERBASIS GRAPH (ORDERING) ====================
        n = len(detections)
        adj_list = {i: [] for i in range(n)}
        indegree = {i: 0 for i in range(n)}
        
        for i in range(n):
            box_i = detections[i]['box']
            w_i = box_i[2] - box_i[0]
            center_y_i = (box_i[1] + box_i[3]) / 2.0  # Titik tengah vertikal i
            
            for j in range(n):
                if i == j:
                    continue
                box_j = detections[j]['box']
                center_y_j = (box_j[1] + box_j[3]) / 2.0  # Titik tengah vertikal j
                
                # 1. Cek lajur kolom yang sama (Overlap X > 30%)
                overlap_x = max(0, min(box_i[2], box_j[2]) - max(box_i[0], box_j[0]))
                is_same_column = overlap_x > 0.3 * min(w_i, box_j[2] - box_j[0])
                
                if is_same_column:
                    # REVISI: Menggunakan perbandingan titik tengah vertikal atau koordinat y_min
                    # untuk menentukan hubungan atas-bawah yang lebih aman dari overlap kecil
                    if box_i[1] < box_j[1] and center_y_i < center_y_j:
                        adj_list[i].append(j)
                        indegree[j] += 1
                else:
                    # 2. Cek lajur kolom yang berbeda (Kiri ke Kanan)
                    if box_i[2] <= box_j[0]:  # i benar-benar di kiri j
                        overlap_y = max(0, min(box_i[3], box_j[3]) - max(box_i[1], box_j[1]))
                        if overlap_y > 0 or (box_i[1] < box_j[3] and box_j[1] < box_i[3]):
                            adj_list[i].append(j)
                            indegree[j] += 1

        # Kahn's Algorithm untuk Topological Sort
        queue = [i for i in range(n) if indegree[i] == 0]
        queue.sort(key=lambda i: (detections[i]['box'][0], detections[i]['box'][1]))
        
        sorted_indices = []
        while queue:
            curr = queue.pop(0)
            sorted_indices.append(curr)
            
            for neighbor in adj_list[curr]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)
            
            queue.sort(key=lambda i: (detections[i]['box'][0], detections[i]['box'][1]))

        # Jika terjadi circular dependency (sangat jarang), fallback ke urutan asli
        if len(sorted_indices) < n:
            print("⚠️ Graph memiliki siklus, menggunakan urutan fallback spasial.")
            remaining = [i for i in range(n) if i not in sorted_indices]
            remaining.sort(key=lambda i: (detections[i].get('column', 0), detections[i]['box'][1]))
            sorted_indices.extend(remaining)
        # ===================================================================
        
        for order_idx, orig_idx in enumerate(sorted_indices, 1):
            det = detections[orig_idx]
            box = det['box']
            label_name = det['label_name']
            
            # Crop region
            x_min, y_min, x_max, y_max = [int(coord) for coord in box]
            region_img = result_data['image'].crop((x_min, y_min, x_max, y_max))
            
            # Save region
            region_filename = f"region_{order_idx:03d}_{label_name}_{timestamp}.png"
            region_path = regions_dir / region_filename
            region_img.save(region_path)
            
            # Save region text (metadata)
            txt_filename = f"region_{order_idx:03d}_{label_name}_{timestamp}.txt"
            txt_path = regions_dir / txt_filename
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"Reading Order: {order_idx}\n")
                f.write(f"Type: {label_name}\n")
                f.write(f"Confidence: {det['confidence']:.3f}\n")
                f.write(f"Position: ({box[0]:.0f}, {box[1]:.0f}) - Size: {box[2]-box[0]:.0f}x{box[3]-box[1]:.0f}\n")
                f.write(f"Original Image: {input_path}\n")
                
        print(f"   Regions saved: {regions_dir}")
        print(f"   Total regions: {len(detections)}")

        return regions_dir

    def visualize_bbox(self, image_input, bboxes, classes, scores, id_to_names, sorted_indices=None, alpha=0.3, show_labels=True):
        if isinstance(image_input, Image.Image):
            image = np.array(image_input)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                image = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
            else:
                image = image_input.copy()
        else:
            raise ValueError("Input must be PIL Image or numpy array")

        if len(bboxes) == 0:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        overlay = image.copy()
        cmap = self.colormap(N=len(id_to_names), normalized=False)

        # Buat mapping urutan baca berdasarkan indeks hasil topological sort graph
        reading_order_map = {}
        if sorted_indices:
            for order_idx, orig_idx in enumerate(sorted_indices, 1):
                reading_order_map[orig_idx] = order_idx

        for i in range(len(bboxes)):
            try:
                bbox = bboxes[i]
                if torch.is_tensor(bbox):
                    bbox = bbox.cpu().numpy()
                
                class_id = classes[i]
                if torch.is_tensor(class_id):
                    class_id = class_id.item()
                
                score = scores[i]
                if torch.is_tensor(score):
                    score = score.item()
                    
                x_min, y_min, x_max, y_max = map(int, bbox)
                class_id = int(class_id)
                class_name = id_to_names.get(class_id, f"unknown_{class_id}")

                color = tuple(int(c) for c in cmap[class_id % len(cmap)])

                cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 3)

                if show_labels:
                    # Gunakan urutan baca dari graph jika tersedia, jika tidak pakai urutan asli tensor (+1)
                    reading_order = reading_order_map.get(i, i + 1)
                    text = f"#{reading_order} {class_name}"
                    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    
                    cv2.rectangle(image, (x_min, y_min - text_height - baseline - 4), 
                                 (x_min + text_width + 8, y_min), color, -1)
                    cv2.putText(image, text, (x_min + 4, y_min - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            except Exception as e:
                print(f"Skipping box {i} due to error: {e}")

        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def merge_extracted_regions_by_column(self, regions_dir):
        """
        Fungsi khusus pasca-ekstraksi untuk menggabungkan file region (.png) 
        menjadi satu kesatuan file per kolom berdasarkan analisis sekuensial koordinat Y.
        Region berjenis 'Picture' akan diabaikan dari proses ini.
        """
        regions_dir = Path(regions_dir)
        if not regions_dir.exists():
            print(f"❌ Folder regions tidak ditemukan: {regions_dir}")
            return

        # 1. Ambil semua file txt secara urutan abjad ascending (sesuai urutan reading order graph)
        txt_files = sorted(list(regions_dir.glob("region_*.txt")))
        if not txt_files:
            print("ℹ️ Tidak ditemukan file metadata region (.txt) untuk proses merge kolom.")
            return

        print(f"\n🔄 Menjalankan Logika Merging Sekuensial Kolom di: {regions_dir.name}")
        
        columns_data = []  # Menyimpan daftar list region per kelompok kolom
        current_column_regions = []
        prev_y_min = None

        # 2 & 3. Berjalan sekuensial dan bandingkan koordinat Y untuk deteksi kolom
        for txt_path in txt_files:
            # -----------------------------------------------------------------
            # FILTER: Abaikan region yang mengandung '_Picture_' di tengah nama filenya
            # -----------------------------------------------------------------
            if "_Picture_" in txt_path.name:
                continue

            # Cari file PNG pasangannya
            png_path = txt_path.with_suffix(".png")
            if not png_path.exists():
                continue

            # Baca info koordinat posisi Y dari file TXT
            y_min = None
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if "Position:" in line:
                            # Contoh baris: Position: (103, 452) - Size: 300x120
                            pos_part = line.split("Position:")[1].split("-")[0].strip()
                            # Ambil koordinat y_min dari tuple string (103, 452)
                            y_min = float(pos_part.replace("(", "").replace(")", "").split(",")[1].strip())
                            break
            except Exception as e:
                print(f"⚠️ Gagal membaca koordinat dari {txt_path.name}: {e}")

            if y_min is None:
                continue

            # Logika Penentuan Kelompok Kolom:
            # Jika Y lebih kecil dari Y sebelumnya, dipastikan mata pembaca lompat ke atas (pindah kolom baru)
            if prev_y_min is not None and y_min < prev_y_min:
                if current_column_regions:
                    columns_data.append(current_column_regions)
                current_column_regions = []

            current_column_regions.append({
                'png_path': png_path,
                'txt_path': txt_path,
                'y_min': y_min
            })
            prev_y_min = y_min

        # Masukkan sisa kolom terakhir jika ada
        if current_column_regions:
            columns_data.append(current_column_regions)

        # 4. Terapkan Merging Gabungan Gambar per Kolom yang terdeteksi
        print(f"📦 Terdeteksi total {len(columns_data)} Lajur Kolom Koran.")
        
        # Buat folder khusus untuk menyimpan hasil gabungan kolom agar rapi
        merged_output_dir = regions_dir.parent / f"{regions_dir.name}_merged_columns"
        merged_output_dir.mkdir(parents=True, exist_ok=True)

        for col_idx, col_regions in enumerate(columns_data, 1):
            # Buka semua object gambar PIL dalam satu kolom ini
            images = [Image.open(reg['png_path']) for reg in col_regions]
            
            # Hitung ukuran kanvas baru untuk penggabungan vertikal
            max_width = max(img.width for img in images)
            total_height = sum(img.height for img in images)

            # Buat gambar putih kosong baru sebesar total akumulasi dimensi kelompok kolom
            merged_image = Image.new("RGB", (max_width, total_height), color=(255, 255, 255))

            # Tempel gambar satu per satu dari atas ke bawah
            current_y_offset = 0
            for img in images:
                merged_image.paste(img, (0, current_y_offset))
                current_y_offset += img.height
                img.close()  # Tutup resource gambar

            # Simpan file hasil gabungan kolom baru
            merged_png_filename = f"kolom_{col_idx:02d}.png"
            merged_png_path = merged_output_dir / merged_png_filename
            merged_image.save(merged_png_path)

            # Gabungkan teks isi metadata ke satu file baru per kolom
            merged_txt_filename = f"kolom_{col_idx:02d}.txt"
            merged_txt_path = merged_output_dir / merged_txt_filename
            with open(merged_txt_path, "w", encoding="utf-8") as out_f:
                out_f.write(f"=== LAYOUT GABUNGAN KOLOM {col_idx} ===\n")
                out_f.write(f"Terdiri dari {len(col_regions)} Sub-Region asli.\n\n")
                for reg in col_regions:
                    out_f.write(f"--- File Asal: {reg['txt_path'].name} ---\n")
                    with open(reg['txt_path'], "r", encoding="utf-8") as in_f:
                        out_f.write(in_f.read())
                    out_f.write("\n")

            print(f"   ✨ Berhasil menyimpan -> {merged_png_filename} (Gabungan {len(col_regions)} region)")

        print(f"📁 Semua hasil merge kolom tersimpan di folder: {merged_output_dir}")
        
        return merged_output_dir