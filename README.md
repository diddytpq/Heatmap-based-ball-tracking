# heatmap based tennis ball tracking

## 1. Make Conda env
        conda env create --file env.yaml

## 2. Run Demo video

        python src/predict_custom.py

## 3. Run realsensec Camera version

        python src/rs_predict_custom.py

## 4. Make Train data


## 5. Model Train

        python src/train_custom.py --multi_gpu=True --load_weight=weights/220304.tar --freeze=Ture

## 6. Run Test to Val data

        python src/val_model.py --load_weight=weights/custom_20.tar

<br>

## reference
        Tracknet-v2 : https://gitlab.com/lukelin/tracknetv2-pytorch
        Label Tool : https://hackmd.io/CQmL6OKKSGKY9xUvU8n0iQ
