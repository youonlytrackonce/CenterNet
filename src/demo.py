from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2

from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)

  cntr = 1
  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    detector.pause = False
    while True:
        _, img = cam.read()
        file1 = open('/home/fatih/phd/CenterNet_DeepPBM/set0/exp/masked_txt/{:0>6d}.txt'.format(cntr),'w')
        cv2.imshow('input', img)
        width = cam.get(3)
        height = cam.get(4)
        ret = detector.run(img)
        # print(ret['results'].keys())
        print(len(ret['results'][1]))
        for ik in range(100):
            if ret['results'][1][ik][4] > 0.3:
                #print(ret['results'][1][ik])
                conf = ret['results'][1][ik][4]
                x1 = ret['results'][1][ik][0]
                if x1 < 0:
                    x1 = 0
                y1 = ret['results'][1][ik][1]
                if y1 < 0:
                    y1 = 0
                x2 = ret['results'][1][ik][2]
                if x2 > width:
                    x2 = width
                y2 = ret['results'][1][ik][3]
                if y2 > height:
                    y2 = height
                w = x2-x1
                h = y2-y1
                c1 = (x1 + w/2)/width
                c2 = (y1 + h/2)/height
                w = w/width
                h= h/height
                #cv2.rectangle(img,(int(ret['results'][1][ik][0]), int(ret['results'][1][ik][1])), (int(ret['results'][1][ik][2]), int(ret['results'][1][ik][3])),(255,5,5),1)
                file1.write('0 {} {} {} {} {}\n'.format(conf, c1, c2, w, h))
        #cv2.imwrite('/home/fatih/phd/CenterNet_DeepPBM/set0/exp/masked/{:0>6d}.jpg'.format(cntr), img)
        file1.close()
        cntr += 1
        print(cntr)
        time_str = ''
        for stat in time_stats:
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)
        if cv2.waitKey(1) == 27:
            return  # esc to quit
  else:
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]
    
    for (image_name) in image_names:
      ret = detector.run(image_name)
      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)
if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
