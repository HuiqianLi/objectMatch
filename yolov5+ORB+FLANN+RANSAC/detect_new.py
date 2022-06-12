# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np

import torch
import torch.backends.cudnn as cudnn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定第一块gpu

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from object_match import ORB, ByFlann, RANSAC


@torch.no_grad()
def run(
        weights=ROOT / 'weights/yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/VOC.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        # save_txt=True,  # save results to *.txt
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    fps = 0 # fps

    num_bbx = []    # 记录每帧bbx数
    xy = []         # 记录中心点坐标
    distance = []   # 记录距离
    preds = []       # 记录是否有预测框（漏检率误诊率用）
    count = 0       # 帧计数

    # 初始化当前帧的前两帧的预测信息
    lastFrame1 = None
    lastFrame2 = None

    # 初始化前5帧
    lastImg = [None]*5

    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # 如果第一二帧是None，对其进行初始化，计算第一二帧的不同
        if lastFrame2 is None:
            if lastFrame1 is None:
                lastFrame1 = pred
            else:
                lastFrame2 = pred
                # 前两帧的预测信息添加到pred里面
                pred = np.row_stack((pred.cpu().numpy(),lastFrame1.cpu().numpy()))
                pred = np.row_stack((pred,lastFrame2.cpu().numpy()))
                # numpy格式转换回gpu tensor
                pred = torch.from_numpy(pred).cuda()
            # continue
        # 当前帧设为下一帧的前帧,前帧设为下一帧的前前帧
        lastFrame1 = lastFrame2
        lastFrame2 = pred

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            # if的条件提出来
            FLAG = len(det)

            # ------------------------------------------后处理start--------------------------------------------
            # 去除假阳性，判断连续4个帧是否均为阳性  frame是当前帧数 seen计算总数的
            num_bbx.append(len(det))
            if count > 4:
                sum_bbox4 = sum(num_bbx[count-1-3:count-1])  # 4帧一组
                if sum_bbox4 == 0:       # 3帧全空
                    im0 = im             # 输出原图
                    FLAG = 0             # 跳过画框
            
            # 去除假阳性，并非连续的检测到息肉
            if count > 4:
                if num_bbx[count] != num_bbx[count-1]: # 01 10
                    if num_bbx[count-2] != num_bbx[count-1]: # 101 010
                        im0 = im
                        FLAG = 0
                    if num_bbx[count-2] == num_bbx[count-1] and num_bbx[count-3] != num_bbx[count-2]: # 1001 0110
                        im0 = im
                        FLAG = 0

            # 去除假阳性，计算连续两帧预测框中心点距离
            if FLAG:
                pred_xyxy = det[:, :4].cpu().numpy()[0].tolist()
                xy.append([pred_xyxy[2]-pred_xyxy[0],pred_xyxy[3]-pred_xyxy[1]])
            else:
                xy.append([0,0])
            if count > 1:       # dis = (x2-x1)^2+(y2-y1)^2
                dis = (xy[count][0] - xy[count-1][0])**2 + (xy[count][1] - xy[count-1][1])**2
                if xy[count-1][1] != xy[count-1][0] and xy[count-1][0] != 0: # 上一帧不是(0,0)
                    distance.append(dis)
                mean_dis = 41267.63      # 计算的整个视频的连续两帧均值(非零)
                if dis > mean_dis:  # 与上一帧的距离大于均值
                    im0 = im             # 输出原图
                    FLAG = 0             # 跳过画框
            
            # -----------------------------------特征匹配start-----------------------------------
            # 和前面第五帧进行特征匹配
            # ORB + Flann + RANSAC + 目标检测
            if lastImg[4] is None: # 0XXXX
                if lastImg[3] is None: # 00XXX
                    if lastImg[2] is None: # 000XX
                        if lastImg[1] is None:
                            if lastImg[0] is None: # 00000
                                lastImg[0] = im0s
                            else: # 00001
                                lastImg[1] = im0s
                        else: # 00011
                            lastImg[2] = im0s
                    else: # 00111
                        lastImg[3] = im0s
                else: # 01111
                    lastImg[4] = im0s
                continue
        
            img1 = lastImg[4]
            img2 = im0s

            # 当前帧设为下一帧的前帧,前帧设为下一帧的前前帧
            lastImg[4] = lastImg[3]
            lastImg[3] = lastImg[2]
            lastImg[2] = lastImg[1]
            lastImg[1] = lastImg[0]
            lastImg[0] = im0s

            if count>3 and (not FLAG) and (num_bbx[count-4]!=0):
                kp1, des1 = ORB(img1)
                kp2, des2 = ORB(img2)
                matches = ByFlann(img1, img2, kp1, kp2, des1, des2, "ORB")
                RANSAC(img1, img2, kp1, kp2, matches,count)
            # -----------------------------------特征匹配end-------------------------------------
            # break
            # ------------------------------------------后处理end--------------------------------------------

            # 统计预测框（漏检率误诊率用）
            if FLAG:
                preds.append(1)
            else:
                preds.append(0)

            if FLAG:
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # 在原图上画框 + 将预测到的目标剪切出来 保存成图片 保存在save_dir/crops下
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    
                    # 计算偏移
                    # x = xyxy[0].cpu().numpy() + xyxy[2].cpu().numpy()
                    # y = xyxy[1].cpu().numpy() + xyxy[3].cpu().numpy()
                    # print("+++++++++++++++++",x,y)

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
        # 计算fps
        deal_fps = 1 / (time_sync() - t1)
        # LOGGER.info(f'deal_fps: {deal_fps:.3f}')
        fps += deal_fps
        # cv2.putText(im0, "Processing FPS:  "+str(round(deal_fps, 2)), (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # # convert inference time in milliseconds to frames per second as well?
        # fpsm = 1 / (t3 - t2)
        # LOGGER.info(f'FPS: {fpsm:.1f}')
        count += 1       # 帧计数
        

    # print(sum(distance)/len(distance))
    # mean_dis = round(sum(distance)/len(np.nonzero(distance)[0]),2)
    # print("非零距离均值：", mean_dis)

    # # 统计预测框（漏检率误诊率用）
    # Note=open('cal/preds_n.txt',mode='w')
    # Note.write(str(preds)) #\n 换行符
    # Note.close()

    # # Print results
    # t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    # fps = fps / seen # 计算FPS
    # LOGGER.info(f'FPS: {fps:.1f}')
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    # 修改模型路径 yolov5_n
    parser.add_argument('--weights', nargs='+', type=str, default='/data/anhui-ai/lhq/yolov5/runs/train/exp2/weights/best.pt', help='model path(s)') # 修改
    # 修改数据路径
    parser.add_argument('--source', type=str, default='/data/anhui-ai/lhq/video/0209.mp4', help='file/dir/URL/glob, 0 for webcam') # 修改
    parser.add_argument('--data', type=str, default=ROOT / 'data/VOC.yaml', help='(optional) dataset.yaml path') # 修改
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.15, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    # 保存路径 default=ROOT / 'runs/detect'
    parser.add_argument('--project', default='/data/anhui-ai/lhq/yolov5/runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
