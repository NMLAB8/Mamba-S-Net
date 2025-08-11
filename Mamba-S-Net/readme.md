

# **Mamba-S-Net**

A source separation network based on Mamba architecture.

## **Environment Setup**

```
pip install -r requirements.txt
```

## **Training** 

Modify the `conf/config.yaml` configuration file, then run:

```
python msnet/train.py --config_path ./conf/config.yaml --save_path ./result/
```

## **Inference**

```
python msnet/inference.py --input_dir /path/to/input --output_dir /path/to/output --config_path ./conf/config.yaml --checkpoint_path ./result/checkpoint.th
```

- `input_dir`: Input directory containing audio files
- `output_dir`: Directory to save separated audio files
- `checkpoint_path`: Path to the trained model file