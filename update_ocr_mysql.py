# update_ocr_mysql.py
import os
import argparse
from pathlib import Path

# Import class Database dari file database.py lokal Anda
from database import Database

def update_ocr_to_mysql(ocr_base_dir):
    ocr_path = Path(ocr_base_dir)
    
    # 1. Validasi folder ocr utama
    if not ocr_path.exists() or not ocr_path.is_dir():
        print(f"❌ Folder target tidak ditemukan: {ocr_base_dir}")
        return

    # 2. Ambil semua file .txt secara rekursif (.rglob)
    # Filter agar file teks yang berada di dalam folder "err" diabaikan
    all_txt_files = sorted(list(ocr_path.rglob("*.txt")))
    txt_files = [f for f in all_txt_files if "err" not in f.parts]

    if not txt_files:
        print(f"✨ Tidak ditemukan file *.txt (hasil OCR sukses) di folder: {ocr_base_dir}")
        return

    print(f"📚 Menemukan {len(txt_files)} file hasil OCR untuk diproses langsung ke database.\n")

    # 3. Inisialisasi Database menggunakan class lokal
    db = None
    try:
        db = Database()
        print("🔌 Berhasil terhubung ke database MySQL menggunakan modul database.py.")
        print("=" * 80)
        
        success_count = 0
        skipped_count = 0

        # 4. Looping untuk membaca file dan eksekusi UPDATE
        for idx, txt_path in enumerate(txt_files, 1):
            # relative_display mengambil path relatif (contoh: 1966/01/10/thisfile.txt)
            relative_display = txt_path.relative_to(ocr_path)
            
            print(f"🔄 [{idx}/{len(txt_files)}] Memproses: ocr/{relative_display}")
            
            # 🛠️ Mengubah ekstensi path dari .txt menjadi .pdf (contoh: 1966/01/10/thisfile.pdf)
            # Menggunakan as_posix() agar separator path selalu menggunakan '/' (standar database/Linux) bukan '\'
            pdf_relative_path = relative_display.with_suffix(".pdf").as_posix()
            
            # Membaca isi teks dari file OCR
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    ocr_content = f.read()
            except Exception as e:
                print(f"   ❌ Gagal membaca file teks: {e}")
                continue

            print(f"Mengupdate OCR_lighton berdasarkan {pdf_relative_path} di DB")
            # Menyiapkan query UPDATE dengan pencarian eksak (sama dengan (=)) pada kolom path
            # Silakan ganti 'nama_tabel_anda' dengan nama tabel asli di database Anda
            update_query = "UPDATE pdf_transcriptions t JOIN pdf_article_files p ON p.oldTarkID = t.oldTarkID SET t.OCR_lighton = %s WHERE p.Name = %s"
            
            # Eksekusi langsung ke database memanfaatkan class Database Anda
            result = db.execute(update_query, (ocr_content, pdf_relative_path))

            if result:
                print(f"   ✅ Sukses mengeksekusi perintah UPDATE untuk path: '{pdf_relative_path}'")
                success_count += 1
            else:
                print(f"   ⚠️ Gagal/Dilewati (Tidak ada baris yang cocok atau terjadi DB error).")
                skipped_count += 1
                
            print("-" * 60)

        print("\n" + "=" * 80)
        print(f"🏁 PROSES SINKRONISASI SELESAI!")
        print(f"   🟢 Berhasil diproses : {success_count} file")
        print(f"   🟡 Gagal/Dilewati    : {skipped_count} file")
        print("=" * 80)

    except Exception as e:
        print(f"❌ Terjadi kesalahan saat inisialisasi database: {e}")
    finally:
        if db:
            db.close()
            print("🔌 Koneksi database MySQL ditutup.")

def main():
    parser = argparse.ArgumentParser(
        description="Script Utility Sinkronisasi Hasil OCR ke Database MySQL berdasarkan Path Eksak PDF",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--ocr-dir",
        default="output/ocr",
        help="Path ke folder utama hasil OCR (default: output/ocr)"
    )
    
    args = parser.parse_args()
    
    update_ocr_to_mysql(args.ocr_dir)

if __name__ == "__main__":
    main()