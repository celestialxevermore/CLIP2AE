import torch as th
import numpy as np
from PIL import Image
# pytorch=1.7.1
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
# pip install opencv-python
import cv2

class RawVideoExtractorCV2():
    def __init__(self, centercrop=False, size=224, framerate=-1, ):
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate
        self.transform = self._transform(self.size)
    ## Data preprocessing 
    def _transform(self, n_px):
        return Compose([
            #기본적인 내용ㅇㅇ
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    #재혁 : 
    def video_to_tensor(self, video_file, preprocess, sample_fp=0, start_time=None, end_time=None):
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                   and start_time > -1 and end_time > start_time
        assert sample_fp > -1

        # Samples a frame sample_fp X frames.
        cap = cv2.VideoCapture(video_file)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #초당 프레임 개수
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        #해당 비디오의 전체 길이
        #왜 그럴까...흠 그냥 frameCount // fps 하면 되는 것을. 
        #???? 
        total_duration = (frameCount + fps - 1) // fps
        start_sec, end_sec = 0, total_duration
        ###20220524 ipynb
        if start_time is not None:
            start_sec, end_sec = start_time, end_time if end_time <= total_duration else total_duration
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))

        interval = 1
        if sample_fp > 0:
            interval = fps // sample_fp
        else:
            sample_fp = fps
        if interval == 0: interval = 1

        inds = [ind for ind in np.arange(0, fps, interval)]
        assert len(inds) >= sample_fp
        #trunc스러움. ㅇㅇ 
        inds = inds[:sample_fp]

        ret = True
        images, included = [], []
        ###############20220524 ipynb ####################### 
        for sec in np.arange(start_sec, end_sec + 1):
            if not ret: break
            sec_base = int(sec * fps)
            for ind in inds:
                ##### cv2??????? 
                cap.set(cv2.CAP_PROP_POS_FRAMES, sec_base + ind)
                ret, frame = cap.read()
                if not ret: break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))

        cap.release()

        if len(images) > 0:
            video_data = th.tensor(np.stack(images))
        else:
            video_data = th.zeros(1)
        #### 정제를 해주는 과정을 말합니다. 대충!
        return {'video': video_data}
    
    def get_video_data(self, video_path, start_time=None, end_time=None):
        image_input = self.video_to_tensor(video_path, self.transform, sample_fp=self.framerate, start_time=start_time, end_time=end_time)
        return image_input

    #video가 들어오면 size()를 재정립한 이후에 tensor를 반환합니다. 
    def process_raw_data(self, raw_video_data):
        tensor_size = raw_video_data.size()
        ###20220524 ipynb bs ts H W C
        # bs에 대해서 객체 하나를 뜻하기 때문에 const한 숫자 1이 들어간 것 같습니다. (응엽)
        tensor = raw_video_data.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])
        return tensor
    ##### 논문에 순서 관련 내용이 있었음!!!! 20220524
    #순서 바꾸기
    def process_frame_order(self, raw_video_data, frame_order=0):
        # 0: ordinary order; 1: reverse order; 2: random order.
        if frame_order == 0:
            pass
        elif frame_order == 1:
            reverse_order = np.arange(raw_video_data.size(0) - 1, -1, -1)
            raw_video_data = raw_video_data[reverse_order, ...]
        elif frame_order == 2:
            random_order = np.arange(raw_video_data.size(0))
            np.random.shuffle(random_order)
            raw_video_data = raw_video_data[random_order, ...]

        return raw_video_data

# An ordinary video frame extractor based CV2
RawVideoExtractor = RawVideoExtractorCV2