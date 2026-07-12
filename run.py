# run.py
import glob
import os
import argparse
from pathlib import Path
from segmenter import DocumentSegmenterEngine
from lighton_ocr_client import ocr_image, is_server_ready, ServerNotReadyError, OcrRequestError

def main():
    # Mengambil instance engine untuk mengekstrak data konfigurasi MODELS bawaan asli
    temp_engine = DocumentSegmenterEngine()

    parser = argparse.ArgumentParser(
        description="Document Layout Analysis - Console Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py document.jpg
  python run.py document.jpg --model "Docling Layout Egret Large"
  python run.py document.jpg --conf 0.5 --iou 0.4 --output results
  python run.py document.jpg --nms "Custom IoMin"
        """
    )
    
    parser.add_argument(
        "image",
        help="Path to the document image file"
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
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        print("\n📋 Available Models:")
        print("-" * 50)
        for i, model_name in enumerate(temp_engine.MODELS.keys(), 1):
            model_path = temp_engine.MODELS[model_name]["path"]
            print(f"{i:2d}. {model_name}")
            print(f"    Path: {model_path}")
        print()
        return
    
    # 1. Inisialisasi Engine Utama Pipeline
    engine = DocumentSegmenterEngine(default_device='cpu')

    # Print header
    print("\n" + "=" * 70)
    print("📄 DOCUMENT LAYOUT ANALYSIS - CONSOLE APP")
    print("=" * 70)
    print(f"Device: {engine.device}")
    print(f"Image: {args.image}")
    print(f"Model: {args.model}")
    print(f"Parameters: Conf={args.conf}, IoU={args.iou}, NMS={args.nms}")
    print("=" * 70 + "\n")
    
    # Process image via Engine Class
    result = engine.process_image(
        args.image,
        args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        nms_method=args.nms
    )
    
    if result is None:
        print("\n❌ Processing failed or no detections found")
        return
    
    # Save results with reading mode (Menghasilkan folder region_xxx_xxx)
    regions_directory = engine.save_detection_results(result, args.output)

    # =======================================================================
    # PROSES JALAN PASCA-EKSTRAKSI: MERGING UTK PENGELOMPOKAN PER-KOLOM
    # =======================================================================
    merged_output_dir = None
    if regions_directory and os.path.exists(regions_directory):
        merged_output_dir = engine.merge_extracted_regions_by_column(regions_directory)
    # =======================================================================    

    # =======================================================================
    # INTEGRASI PROSES OCR JALAN KEDUA
    # =======================================================================
    if merged_output_dir and os.path.exists(merged_output_dir):
        print("\n" + "-" * 50)
        print("🔍 MEMULAI PROSES OCR DENGAN LIGHTON_OCR_CLIENT")
        print("-" * 50)

        # Cari segmen gambar hasil merge atau potongan kolom yang ada di dalam output folder
        # Prioritaskan file hasil merge_by_column (biasanya berformat terstruktur atau berada di folder terkait)
        # Menyesuaikan dengan output default segmenter, kita ambil file png yang valid
        # Diurutkan (sorted) alphabetis agar pasti berjalan runtut: kolom_01.png, kolom_02.png, dst.
        column_images = sorted(glob.glob(os.path.join(str(merged_output_dir), "kolom_*.png")))

        if not column_images:
            print("⚠️ Tidak ditemukan potongan gambar (.png) untuk di-OCR di folder hasil segmentasi.")
        else:
            print(f"📋 Menemukan {len(column_images)} potongan gambar untuk diproses.")
            
            combined_ocr_text = []
            
            for index, img_path in enumerate(column_images, 1):
                filename = os.path.basename(img_path)
                print(f"   [{index}/{len(column_images)}] Memproses {filename} ... ", end="", flush=True)
                
                try:
                    # Mengirim segmen gambar ke llama-server
                    text_result = ocr_image(img_path)
                    combined_ocr_text.append(f"{text_result}\n")
                    print("✅ Sukses")
                except (ServerNotReadyError, OcrRequestError, FileNotFoundError) as e:
                    print(f"❌ Gagal!\n   Detail Error: {e}")
                    combined_ocr_text.append(f"--- Bagian: {filename} (Gagal OCR) ---\n")

            # Menyimpan hasil gabungan OCR utuh ke dalam folder output utama
            source_filename_stem = Path(args.image).stem
            final_txt_name = f"{source_filename_stem}.txt"
            final_output_path = os.path.join(args.output, final_txt_name)
            
            with open(final_output_path, "w", encoding="utf-8") as f_out:
                f_out.write("\n".join(combined_ocr_text))
                
            print("-" * 50)
            print(f"📝 Hasil OCR gabungan berhasil disimpan di: {final_output_path}")



    print("\n✅ Processing complete!")
    print(f"📁 Results saved to: {args.output}\n" + "=" * 70 + "\n")

if __name__ == "__main__":
    main()