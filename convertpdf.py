# Konversi pdf to image dengan asumsi pdf hanya berisi 1 halaman

import os

import pymupdf

pdf_path = "27-Kliping_630024.pdf"
doc = pymupdf.open(str(pdf_path))
pixmap = doc[0].get_pixmap(dpi=300)
# img_path = os.path.join(args.output, f"{pdf_path.stem}.png")
pixmap.save("27-Kliping_630024.png")
doc.close()
doc = None
