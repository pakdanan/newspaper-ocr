# run.py
import pymupdf # sama dengan fitz
import glob
import os
import argparse
import shutil
import time  # <-- Ditambahkan untuk kalkulasi durasi waktu file
from pathlib import Path

from lighton_ocr_client import ocr_image, is_server_ready, ServerNotReadyError, OcrRequestError

def main():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --input-dir "./documents"
        """
    )
    
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Path to the folder containing PDF files to process"
    )
   
    parser.add_argument(
        "--output",
        default="output",
        help="Output directory (default: output)"
    )
    
    parser.add_argument(
        "--max-dim",
        type=int,
        default=2048,
        help="Batas maksimal dimensi sisi terpanjang gambar hasil render PDF (default: 2048 px)"
    )
    

    args = parser.parse_args()
    
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
        expiration_in_seconds = 3 * 60 

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
            # -------------------------------------------------------------
            # JALUR B: langsung OCR FILE PDF UTUH (TANPA SEGMENTASI)
            # -------------------------------------------------------------
            print(f"🔍 [Direct Mode] Memulai proses direct OCR untuk file PDF asli...")
            print(f"   Memproses {pdf_path.name} langsung ... ", end="", flush=True)
            try:

                # Konversi pdf to image dengan pembatasan dimensi maksimum (max-dim)
                doc = pymupdf.open(str(pdf_path))
                page = doc[0]
                rect = page.rect  # Ukuran halaman dalam point

                # Standar render tajam target ~300 DPI (300 / 72 pt ≈ 4.17)
                target_dpi = 300
                default_scale = target_dpi / 72.0

                max_side = max(rect.width, rect.height)
                # Jika dengan 300 DPI hasilnya melebihi args.max_dim (default 2048 px),
                # turunkan skalanya agar mentok di args.max_dim (aman dari timeout).
                # Jika tidak melebihi, pertahankan target 300 DPI (tidak di-upscale berlebihan).
                if max_side > 0 and (max_side * default_scale > args.max_dim):
                    scale = args.max_dim / max_side
                else:
                    scale = default_scale

                mat = pymupdf.Matrix(scale, scale)
                pixmap = page.get_pixmap(matrix=mat)

                img_path = os.path.join(args.output, f"{pdf_path.stem}.png")
                pixmap.save(img_path)
                print(f"({pixmap.width}x{pixmap.height} px) ... ", end="", flush=True)

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
                    ocr_target_dir = os.path.join(ocr_base_dir, "err", relative_file_path.parent)
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