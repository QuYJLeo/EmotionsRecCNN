import time

import onnxruntime
import torch
import cv2
import numpy as np
from net_slim import Slim
from util import PriorBox, decode


Film_DICT = {
    0: "Anger",
    1: "Contempt",
    2: "Disgust",
    3: "Fear",
    4: "Happy",
    5: "Neural",
    6: "Sadness",
    7: "Surprise",

    "Anger":0,
    "Contempt": 1,
    "Disgust":2,
    "Fear":3,
    "Happy":4,
    "Neural": 5,
    "Sadness":6,
    "Surprise":7,
}

def main():
    net = Slim()
    net.load_state_dict(torch.load('../checkpoints/slim_Final.pth', weights_only=False), strict=False)
    net.cuda()
    net.eval()

    model = onnxruntime.InferenceSession('../checkpoints/emotionModel_noDrop0.4.onnx', providers=['CUDAExecutionProvider',])  # 'CUDAExecutionProvider',


    # cap = 'personID7_dis200_hgt160_Acap.avi'
    cap = 0
    vid = cv2.VideoCapture(cap)
    while True:
        ret, frame = vid.read()
        if frame is None or ret is not True:
            vid = cv2.VideoCapture(cap)
            continue


        try:

            h, w = frame.shape[:2]
            img = np.float32(frame)
            img -= (104, 117, 123)
            img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).cuda()  # 1, 3, 640, 480
            loc, conf, _ = net(img)
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            boxes = (decode(loc.cpu().data.squeeze(0), PriorBox((h, w)).forward().data) * torch.Tensor([w, h, w, h])).numpy()
            # ignore low scores
            inds = np.where(scores > 0.999)[0]
            boxes,  scores = boxes[inds],  scores[inds]
            # print(scores)
            bbox = boxes[scores.argsort()[::-1].tolist()[0], :].astype(np.int32).tolist()

            # cv2.rectangle(frame, (bbox[0]-10, bbox[1]-10), (bbox[2]+10, bbox[3]+10), (0, 0, 255), 6)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 6)
            face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]


            face = cv2.resize(face, (224, 224)) # (224, 224, 3)

            ## 方法2
            face = torch.from_numpy(face.transpose(2, 0,1 ))
            face = face.cuda() / 255

            face = torch.unsqueeze(face, dim=0) # 增加一个维度    # torch.Size([1, 3, 224, 224])
            face = face.cpu().numpy()
            tic = time.time()
            probably = model.run(None, {'input0': face},)
            end = time.time() - tic
            # print('net forward time: {:.4f}, fps:{}'.format(end, round(1 / end, 1)))
            probably = probably[0]  # (1, 8)
            probably = torch.from_numpy(probably)
            probably = torch.squeeze(probably, 0)
            probably = torch.softmax(probably, 0)
            print(probably)

            gap = int((h - 7 * 10) / 7)
            for (i, (emotion, prob)) in enumerate(zip(Film_DICT.values(), probably)):
                text = " {} : {:.3f}%".format(emotion, prob * 100)
                w = int(prob * 300)
                cv2.rectangle(frame, (7, (i * gap) + 10), (w, (i * gap) + gap), (0, 0, 255), -1)
                cv2.putText(frame, text, (10, (i * gap) + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except:
            pass

        cv2.imshow('raw', frame)
        cv2.waitKey(1)




if __name__ == "__main__":
    main()
