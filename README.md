# ARG-DPLM

## Environment configuration
After downloading the environment.yml file, modify the installation path within it, and then use the following command to configure the environment.
```Bash
conda env create -f environment.yml
```

## ARGs Prediction
Please replace the file path with your own and use the following command to predict ARGs.
```Bash
python predict.py \
    --model_path /path/to/ARG-DPLM-MODEL.pth \
    --prot_t5_model_path  /path/to/prot_t5_xl_uniref50 \
    --input_dir  /path/to/input_dir \
    --output_dir  /path/to/output_dir \
    --batch_size 32 \
    --threshold 0.9
```
The download path for ARG-DPLM-MODEL.pth is as follows:
