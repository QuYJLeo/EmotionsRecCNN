import torch
from models import ResMasking

Film_DICT = {
    0: "Anger",
    1: "Contempt",
    2: "Disgust",
    3: "Fear",
    4: "Happy",
    5: "Neural",
    6: "Sad",
    7: "Surprise",

    "Anger":0,
    "Contempt": 1,
    "Disgust":2,
    "Fear":3,
    "Happy":4,
    "Neural": 5,
    "Sad":6,
    "Surprise":7,
}

def main():

    model = ResMasking(num_classes=8)
    model.load_state_dict(torch.load("./checkpoints/emotionModel_noDrop0.4.pth")["net"])

    model.cuda()
    model.eval()

    input_names = ["input0"]
    output_names = ["output0"]
    inputs = torch.randn(1, 3,  224, 224).cuda()

    dynamic_axes_23 = {  # 第2、3维动态
        'input0': [2, 3],
        'output0': [2, 3]
    }

    torch_out = torch.onnx._export(model,
                                inputs,
                                './checkpoints/emotionModel_noDrop0.4.onnx',
                                export_params=True,
                                verbose=False,
                                input_names=input_names,
                                output_names=output_names,
                                # dynamic_axes=dynamic_axes_23
                                )

if __name__ == "__main__":
    main()
