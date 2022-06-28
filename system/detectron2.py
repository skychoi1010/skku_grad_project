from PIL import Image
import cv2
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("deepfashion_validation",)
cfg.DATASETS.TEST = ()
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 1500
cfg.SOLVER.STEPS = (1000, )
cfg.SOLVER.GAMMA = 0.05
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13
cfg.TEST.EVAL_PERIOD = 500
cfg.OUTPUT_DIR = './drive/MyDrive/detectron/data/DeepFashion2/'

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "./drive/MyDrive/detectron/data/DeepFashion2/model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55   # testing threshold -> 0.7 ?
cfg.DATASETS.TEST = ("deepfashion__validation", )
predictor = DefaultPredictor(cfg)

# predictor 정의  
predictor = detectron2.engine.defaults.DefaultPredictor(cfg)
img = cv2.imread("테스트할 이미지.jpg")
outputs = predictor(img)
instances = outputs["instances"].to('cpu')


# print(list(outputs["instances"].get_fields().keys()))
# output 항목들 : 'pred_boxes', 'scores', 'pred_classes', 'pred_masks'
# pred_masks에 mask 정보 - true, false값을 사용해 이미지의 한 픽셀픽셀이 객체의 부분인지를 말해줌 (다각형 꼭지점 좌표를 이용한 mask 데이터 아님)

# {'instances': Instances(num_instances=3, image_height=768, image_width=1024, 
# fields=[pred_boxes: Boxes(tensor([[664.5765,   1.5540, 912.3661, 258.9362],
#         [364.6794,   1.9076, 572.2790, 223.7474],
#         [143.4272,   8.6358, 385.4707, 245.0285]])), 
# scores: tensor([0.9951, 0.9943, 0.9771]), 
# pred_classes: tensor([0, 0, 0]), 
# pred_masks: tensor([[[False, False, False,  ..., False, False, False],
#          [False, False, False,  ..., False, False, False],
#          [False, False, False,  ..., False, False, False],
#          ...,
#          [False, False, False,  ..., False, False, False],
#          [False, False, False,  ..., False, False, False],
#          [False, False, False,  ..., False, False, False]],

#         [[False, False, False,  ..., False, False, False],
#          [False, False, False,  ..., False, False, False],
#          [False, False, False,  ..., False, False, False],
#          ...,
#          [False, False, False,  ..., False, False, False],
#          [False, False, False,  ..., False, False, False],
#          [False, False, False,  ..., False, False, False]],

#         [[False, False, False,  ..., False, False, False],
#          [False, False, False,  ..., False, False, False],
#          [False, False, False,  ..., False, False, False],
#          ...,
#          [False, False, False,  ..., False, False, False],
#          [False, False, False,  ..., False, False, False],
#          [False, False, False,  ..., False, False, False]]])])}
# 
# 공식문서 https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-output-format
#
# “instances”: Instances object with the following fields:
    # “pred_boxes”: Boxes object storing N boxes, one for each detected instance.
    # “scores”: Tensor, a vector of N confidence scores.
    # “pred_classes”: Tensor, a vector of N labels in range [0, num_categories).
    # “pred_masks”: a Tensor of shape (N, H, W), masks for each detected instance.



# 테스트 이미지 넣어서 인식 결과 표현된 이미지 출력 후 저장 (카테고리명, 예측확률, instance segmentation 결과 색칠, 바운딩박스 표현됨)
v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(instances)
cv2.imwrite("결과 표현된 이미지.jpg", v.get_image()[:, :, ::-1])


# 인식된 객체 카테고리명 추출, 바운딩박스 좌표대로 이미지 크롭
pred_classes = instances.pred_classes
boxes = instances.pred_boxes

if isinstance(boxes, detectron2.structures.boxes.Boxes):
    boxes = boxes.tensor.numpy()
else:
    boxes = np.asarray(boxes)

img_original = Image.open("테스트 할 이미지.jpg")
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
class_catalog = metadata.thing_classes

for idx, coordinates in enumerate(boxes):
    class_index = pred_classes[idx]
    class_name = class_catalog[class_index] # 인식된 객체 카테고리명 ex.'vest_dress'

    box = boxes[idx] # pred_boxes
    x_top_left = box[0]
    y_top_left = box[1]
    x_bottom_right = box[2]
    y_bottom_right = box[3]

    crop_img = img_original.crop((int(x_top_left), int(y_top_left), int(x_bottom_right), int(y_bottom_right)))
    crop_img.save("./drive/MyDrive/detectron/data/DeepFashion2/test/" + class_name +".jpg", "JPEG")
