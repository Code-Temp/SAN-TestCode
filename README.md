# SAN-TestCode
SAN code for test

## Runtime Environment
- Using the dependencies
    ```
    pip install -r requirements.txt
    ```

## Model Result

Get the model training result from [WeiYun (passwd:zmcvf4)](https://share.weiyun.com/k7DUruUp).


## Evaluation

Web datatet:

Radical length : 33

- ABINet-TreeSim(SAN):
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config=configs/train_abinet.yaml --phase test --checkpoint=workdir/SAN-web-final/best-train-abinet.pth --test_root=data/web/web_val/ --model_eval=alignment --image_only
```

- VM-TreeSim:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config=configs/pretrain_vision_model.yaml --phase test --checkpoint=workdir/Vision-web-final/best-pretrain-vision-model.pth --test_root=data/web/web_val/ --model_eval=vision --image_only
```

Scene dataset:

Radical length : 39

- ABINet-TreeSim(SAN):
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config=configs/train_abinet.yaml --phase test --checkpoint=workdir/SAN-scene-final/best-train-abinet.pth --test_root=data/scene/scene_val/ --model_eval=alignment --image_only
```

- VM-TreeSim:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config=configs/pretrain_vision_model.yaml --phase test --checkpoint=workdir/Vision-scene-final/best-pretrain-vision-model.pth --test_root=data/scene/scene_val/ --model_eval=vision --image_only
```