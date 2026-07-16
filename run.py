# run.py
import pymupdf # sama dengan fitz
import glob
import os
import argparse
import shutil
import time  # <-- Ditambahkan untuk kalkulasi durasi waktu file
from pathlib import Path

from segmenter import DocumentSegmenterEngine
from lighton_ocr_client import ocr_image, is_server_ready, ServerNotReadyError, OcrRequestError

def main():
    temp_engine = DocumentSegmenterEngine()

    parser = argparse.ArgumentParser(
        description="Document Layout Analysis - Batch PDF Processing Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --input-dir "./documents"
  python run.py --input-dir "./documents" --output "./results" --model "Docling Layout Egret Large"
  python run.py --input-dir "./documents" --no-segment  # Langsung OCR tanpa pemotongan wilayah
        """
    )
    
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Path to the folder containing PDF files to process"
    )
    
    parser.add_argument(
        "--model",
        choices=list(temp_engine.MODELS.keys()),
        default="Docling Layout Egret XLarge",
        help="Model to use for layout detection (default: Docling Layout Egret XLarge)"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.6,
        help="Confidence threshold (default: 0.6)"
    )
    
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for NMS (default: 0.5)"
    )
    
    parser.add_argument(
        "--nms",
        choices=["Standard IoU", "Custom IoMin"],
        default="Standard IoU",
        help="NMS method (default: Standard IoU)"
    )
    
    parser.add_argument(
        "--output",
        default="output",
        help="Output directory (default: output)"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )

    # Menambahkan argument boolean dengan default=True menggunakan BooleanOptionalAction
    # Fitur ini menyediakan dua flag otomatis: --segment (default) dan --no-segment
    parser.add_argument(
        "--segment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Menentukan apakah PDF dipotong per wilayah kolom (default: True). Gunakan --no-segment untuk langsung OCR file PDF asli."
    )
    
    args = parser.parse_args()
    
    if args.list_models:
        print("\n📋 Available Models:")
        print("-" * 50)
        for i, model_name in enumerate(temp_engine.MODELS.keys(), 1):
            model_path = temp_engine.MODELS[model_name]["path"]
            print(f"{i:2d}. {model_name}")
            print(f"    Path: {model_path}")
        print()
        return

    # Validasi folder input
    input_path = Path(args.input_dir)
    if not input_path.exists() or not input_path.is_dir():
        print(f"❌ Folder input tidak ditemukan: {args.input_dir}")
        return

    # Siapkan folder output utama jika belum ada
    os.makedirs(args.output, exist_ok=True)

    # Siapkan subfolder khusus "ocr" di dalam folder output
    ocr_base_dir = os.path.join(args.output, "ocr")
    os.makedirs(ocr_base_dir, exist_ok=True)

    # =======================================================================
    # PROSES PEMBERSIHAN BERKALA (> EXPIRATION) DI AWAL PROSES
    # =======================================================================
    if os.path.exists(args.output):
        print(f"\n" + "-" * 70)
        print(f"🧹 Memeriksa dan membersihkan file/subfolder lama (> waktu expiration) di: {args.output} ...")
        
        now = time.time()
        expiration_in_seconds = 48 * 60 * 60  # 172800 detik

        # Iterasi item yang ada langsung di dalam root output folder
        for item in os.listdir(args.output):
            # JANGAN UTAK-ATIK subfolder "ocr"
            if item == "ocr":
                continue
                
            item_path = os.path.join(args.output, item)
            
            try:
                # Ambil waktu modifikasi terakhir dari file/folder tersebut
                item_mtime = os.path.getmtime(item_path)
                
                # Cek apakah usianya lebih dari 1 hari
                if (now - item_mtime) > expiration_in_seconds:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        print(f"   🗑️ Menghapus subfolder usang: {item}")
                    elif os.path.isfile(item_path):
                        os.remove(item_path)
                        print(f"   🗑️ Menghapus file usang: {item}")
            except Exception as e:
                print(f"   ⚠️ Gagal memeriksa/menghapus {item}: {e}")
                
        print("Base output folder siap.")

    # Pencarian file *.pdf secara rekursif ke seluruh subfolder
    pdf_files = sorted(list(input_path.rglob("*.pdf")))
    
    # Filter tambahan agar file yang sudah berhasil dipindahkan ke folder "done" tidak masuk antrean proses lagi
    pdf_files = [f for f in pdf_files if "done" not in f.parts]

    if not pdf_files:
        print(f"✨ Tidak ada file *.pdf yang ditemukan di folder: {args.input_dir}")
        return

    print(f"📚 Menemukan {len(pdf_files)} file PDF untuk diproses.")

    # 1. Inisialisasi Engine Utama Pipeline (Hanya jika proses segmentasi aktif)
    engine = None
    if args.segment:
        engine = DocumentSegmenterEngine(default_device='cpu')

    # =======================================================================
    # ITERASI PEMROSESAN PDF SATU PER SATU
    # =======================================================================
    for file_idx, pdf_path in enumerate(pdf_files, 1):
        print("\n" + "=" * 70)
        # Mendapatkan file path relatif terhadap input_dir untuk log (contoh: 1966/01/10/thisfile.pdf)
        relative_file_path = pdf_path.relative_to(input_path)
        print(f"🔄 [{file_idx}/{len(pdf_files)}] MEMPROSES FILE: {relative_file_path}")
        print("=" * 70)

        combined_ocr_text = []
        is_failed = False  # Penanda jika terjadi error/kegagalan proses OCR pada file ini

        try:
            if args.segment:
                # -------------------------------------------------------------
                # JALUR A: PROSES DENGAN SEGMENTASI REGION/KOLOM (DEFAULT)
                # -------------------------------------------------------------
                result = engine.process_image(
                    str(pdf_path),
                    args.model,
                    conf_threshold=args.conf,
                    iou_threshold=args.iou,
                    nms_method=args.nms
                )
                
                if result is None:
                    print(f"❌ Gagal memproses {pdf_path.name}: Segmentasi gagal atau tidak ada deteksi.")
                    continue
                
                # Simpan hasil segmentasi mentah ke folder output
                regions_directory = engine.save_detection_results(result, args.output)

                # Merge per kolom
                merged_output_dir = None
                if regions_directory and os.path.exists(regions_directory):
                    merged_output_dir = engine.merge_extracted_regions_by_column(regions_directory)

                # Jalankan proses OCR dari kolom yang berhasil di-merge
                if merged_output_dir and os.path.exists(merged_output_dir):
                    print("\n🔍 Memulai proses OCR untuk segmen kolom...")
                    column_images = sorted(glob.glob(os.path.join(str(merged_output_dir), "kolom_*.png")))

                    if not column_images:
                        print("⚠️ Tidak ditemukan potongan gambar (kolom_*.png) untuk di-OCR.")
                    else:
                        print(f"📋 Menemukan {len(column_images)} potongan kolom untuk diproses.")
                        for index, img_path in enumerate(column_images, 1):
                            filename = os.path.basename(img_path)
                            print(f"   [{index}/{len(column_images)}] OCR {filename} ... ", end="", flush=True)
                            try:
                                text_result = ocr_image(img_path)
                                combined_ocr_text.append(f"{text_result}\n")
                                print("✅ Sukses")
                            except (ServerNotReadyError, OcrRequestError, FileNotFoundError) as e:
                                print(f"❌ Gagal! Detail Error: {e}")
                                is_failed = True  # Tandai sebagai gagal
                                combined_ocr_text.append(f"{filename} ocr failed")
            else:
                # -------------------------------------------------------------
                # JALUR B: langsung OCR FILE PDF UTUH (TANPA SEGMENTASI)
                # -------------------------------------------------------------
                print(f"🔍 [Direct Mode] Memulai proses direct OCR untuk file PDF asli...")
                print(f"   Memproses {pdf_path.name} langsung ... ", end="", flush=True)
                try:

                    # Konversi pdf to image dengan asumsi pdf hanya berisi 1 halaman
                    doc = pymupdf.open(str(pdf_path))
                    pixmap = doc[0].get_pixmap(dpi=300)
                    img_path = os.path.join(args.output, f"{pdf_path.stem}.png")
                    pixmap.save(img_path)
                    doc.close()
                    doc = None

                    text_result = ocr_image(img_path)
                    combined_ocr_text.append(f"{text_result}\n")
                    print("✅ Sukses")
                except (ServerNotReadyError, OcrRequestError, FileNotFoundError) as e:
                    print(f"❌ Gagal! Detail Error: {e}")
                    is_failed = True  # Tandai sebagai gagal
                    combined_ocr_text.append(f"{pdf_path.name} ocr failed")
                finally:
                    if doc is not None:
                        doc.close()

            # -----------------------------------------------------------------
            # REPLIKASI STRUKTUR FOLDER YANG SAMA PADA FOLDER OUTPUT OCR
            # -----------------------------------------------------------------
            if combined_ocr_text:	
                final_txt_name = f"{pdf_path.stem}.txt"
                if is_failed:
                    # Jika ada error, arahkan folder target ke subfolder 'err' di path relatif masing-masing
                    ocr_target_dir = os.path.join(ocr_base_dir, relative_file_path.parent, "err")
                    os.makedirs(ocr_target_dir, exist_ok=True)
                    final_output_path = os.path.join(ocr_target_dir, final_txt_name)
                    print(f"\n⚠️ Proses OCR bermasalah, disimpan di folder err: {final_output_path}")
                else:
                    # Jika sukses, taruh di struktur relative path biasa
                    ocr_target_dir = os.path.join(ocr_base_dir, relative_file_path.parent)
                    os.makedirs(ocr_target_dir, exist_ok=True)
                    final_output_path = os.path.join(ocr_target_dir, final_txt_name)
                    print(f"\n📝 Hasil OCR gabungan sukses disimpan di: {final_output_path}")
                
                with open(final_output_path, "w", encoding="utf-8") as f_out:
                    f_out.write("\n".join(combined_ocr_text))

            # -----------------------------------------------------------------
            # 🛠️ PERBAIKAN: PEMINDAHAN DINAMIS PDF KE {input-dir}/done/{subfolder}
            # -----------------------------------------------------------------
            # relative_file_path.parent mengambil struktur '1966/01/10'
            file_done_dir = input_path / "done" / relative_file_path.parent
            os.makedirs(file_done_dir, exist_ok=True)
            
            dest_path = file_done_dir / pdf_path.name
            
            if dest_path.exists():
                os.remove(dest_path)
                
            shutil.move(str(pdf_path), str(dest_path))
            print(f"🚚 File asli berhasil dipindahkan ke: {dest_path}")

        except Exception as e:
            print(f"❌ Gagal memproses file {pdf_path.name}. Detail Error: {e}")

    print("\n✅ Semua file PDF di folder selesai diproses!")

if __name__ == "__main__":
    main()