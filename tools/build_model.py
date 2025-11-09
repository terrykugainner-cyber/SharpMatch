import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse, cv2, numpy as np
from sharp.shape_model import create, save

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True, help='input image path')
    ap.add_argument('--out', required=True, help='output model .npz path (e.g., part.smodel.npz)')
    ap.add_argument('--roi', type=int, nargs=4, metavar=('x','y','w','h'), help='ROI rect; if omitted use GUI')
    ap.add_argument('--low', type=int, default=60)
    ap.add_argument('--high', type=int, default=140)
    ap.add_argument('--step', type=int, default=2)
    ap.add_argument('--polarity', action='store_true', help='use gradient polarity')
    args = ap.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f'Cannot read {args.image}')
    
    if args.roi is None:
        # Try to use GUI, but handle case where GUI is not available
        try:
            r = cv2.selectROI('Select ROI', img, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow('Select ROI')
            x,y,w,h = map(int, r)
        except cv2.error as e:
            if 'not implemented' in str(e) or 'GUI' in str(e).upper():
                raise SystemExit(
                    f'錯誤：OpenCV 未編譯 GUI 支援，無法使用交互式 ROI 選擇。\n'
                    f'請使用 --roi 參數指定 ROI，格式：--roi x y w h\n'
                    f'範例：--roi 100 100 200 200\n'
                    f'圖像大小：{img.shape[1]}x{img.shape[0]}'
                )
            else:
                raise
    else:
        x,y,w,h = args.roi

    model = create(img, (x,y,w,h), edge_threshold=(args.low,args.high),
                   sampling_step=args.step, use_polarity=args.polarity)
    save(model, args.out)
    # quick preview
    tpl = model['template'].copy()
    color = cv2.cvtColor(tpl, cv2.COLOR_GRAY2BGR)
    pts, dirs = model['points'], model['dirs']
    for i in range(0, len(pts), max(1,int(len(pts)/500)+1)):
        px,py = int(pts[i,0]), int(pts[i,1])
        th = float(dirs[i])
        qx, qy = int(px + np.cos(th)*8), int(py + np.sin(th)*8)
        cv2.circle(color, (px,py), 1, (0,255,0), -1)
        cv2.line(color, (px,py), (qx,qy), (0,255,0), 1, cv2.LINE_AA)
    prev = args.out.replace('.npz','_preview.png')
    cv2.imwrite(prev, color)
    print(f'[OK] model saved → {args.out}\nPreview → {prev}')

if __name__ == '__main__':
    main()
