# Experiment 3: sensitivity analysis for different feature dimensions

# feat_dim=2
python train.py --model-id cdip_0.2_1000_32x64_2_fire3_1.0_0.1 --samples-dir ~/samples/deeprec-cvpr20/cdip_0.2_1000_32x64 --feat-dim 2 --feat-layer fire3 --margin 1.0 --epochs 100 --learning-rate 0.1
python run.py --dataset D1 --model-id cdip_0.2_1000_32x64_2_fire3_1.0_0.1 --results-id D1_0.2_1000_32x64_2_fire3_1.0_0.1_3 --input-size 3000 64 --feat-dim 2 --feat-layer fire3 --vshift 3
python run.py --dataset D2 --model-id cdip_0.2_1000_32x64_2_fire3_1.0_0.1 --results-id D2_0.2_1000_32x64_2_fire3_1.0_0.1_3 --input-size 3000 64 --feat-dim 2 --feat-layer fire3 --vshift 3

# feat_dim=4
python train.py --model-id cdip_0.2_1000_32x64_4_fire3_1.0_0.1 --samples-dir ~/samples/deeprec-cvpr20/cdip_0.2_1000_32x64 --feat-dim 4 --feat-layer fire3 --margin 1.0 --epochs 100 --learning-rate 0.1
python run.py --dataset D1 --model-id cdip_0.2_1000_32x64_4_fire3_1.0_0.1 --results-id D1_0.2_1000_32x64_4_fire3_1.0_0.1_3 --input-size 3000 64 --feat-dim 4 --feat-layer fire3 --vshift 3
python run.py --dataset D2 --model-id cdip_0.2_1000_32x64_4_fire3_1.0_0.1 --results-id D2_0.2_1000_32x64_4_fire3_1.0_0.1_3 --input-size 3000 64 --feat-dim 4 --feat-layer fire3 --vshift 3

# feat_dim=8
python train.py --model-id cdip_0.2_1000_32x64_8_fire3_1.0_0.1 --samples-dir ~/samples/deeprec-cvpr20/cdip_0.2_1000_32x64 --feat-dim 8 --feat-layer fire3 --margin 1.0 --epochs 100 --learning-rate 0.1
python run.py --dataset D1 --model-id cdip_0.2_1000_32x64_8_fire3_1.0_0.1 --results-id D1_0.2_1000_32x64_8_fire3_1.0_0.1_3 --input-size 3000 64 --feat-dim 8 --feat-layer fire3 --vshift 3
python run.py --dataset D2 --model-id cdip_0.2_1000_32x64_8_fire3_1.0_0.1 --results-id D2_0.2_1000_32x64_8_fire3_1.0_0.1_3 --input-size 3000 64 --feat-dim 8 --feat-layer fire3 --vshift 3

# feat_dim=16
python train.py --model-id cdip_0.2_1000_32x64_16_fire3_1.0_0.1 --samples-dir ~/samples/deeprec-cvpr20/cdip_0.2_1000_32x64 --feat-dim 16 --feat-layer fire3 --margin 1.0 --epochs 100 --learning-rate 0.1
python run.py --dataset D1 --model-id cdip_0.2_1000_32x64_16_fire3_1.0_0.1 --results-id D1_0.2_1000_32x64_16_fire3_1.0_0.1_3 --input-size 3000 64 --feat-dim 16 --feat-layer fire3 --vshift 3
python run.py --dataset D2 --model-id cdip_0.2_1000_32x64_16_fire3_1.0_0.1 --results-id D2_0.2_1000_32x64_16_fire3_1.0_0.1_3 --input-size 3000 64 --feat-dim 16 --feat-layer fire3 --vshift 3

# feat_dim=32
python train.py --model-id cdip_0.2_1000_32x64_32_fire3_1.0_0.1 --samples-dir ~/samples/deeprec-cvpr20/cdip_0.2_1000_32x64 --feat-dim 32 --feat-layer fire3 --margin 1.0 --epochs 100 --learning-rate 0.1
python run.py --dataset D1 --model-id cdip_0.2_1000_32x64_32_fire3_1.0_0.1 --results-id D1_0.2_1000_32x64_32_fire3_1.0_0.1_3 --input-size 3000 64 --feat-dim 32 --feat-layer fire3 --vshift 3
python run.py --dataset D2 --model-id cdip_0.2_1000_32x64_32_fire3_1.0_0.1 --results-id D2_0.2_1000_32x64_32_fire3_1.0_0.1_3 --input-size 3000 64 --feat-dim 32 --feat-layer fire3 --vshift 3

# feat_dim=64
python train.py --model-id cdip_0.2_1000_32x64_64_fire3_1.0_0.1 --samples-dir ~/samples/deeprec-cvpr20/cdip_0.2_1000_32x64 --feat-dim 64 --feat-layer fire3 --margin 1.0 --epochs 100 --learning-rate 0.1
python run.py --dataset D1 --model-id cdip_0.2_1000_32x64_64_fire3_1.0_0.1 --results-id D1_0.2_1000_32x64_64_fire3_1.0_0.1_3 --input-size 3000 64 --feat-dim 64 --feat-layer fire3 --vshift 3
python run.py --dataset D2 --model-id cdip_0.2_1000_32x64_64_fire3_1.0_0.1 --results-id D2_0.2_1000_32x64_64_fire3_1.0_0.1_3 --input-size 3000 64 --feat-dim 64 --feat-layer fire3 --vshift 3

# feat_dim=256
python train.py --model-id cdip_0.2_1000_32x64_256_fire3_1.0_0.1 --samples-dir ~/samples/deeprec-cvpr20/cdip_0.2_1000_32x64 --feat-dim 256 --feat-layer fire3 --margin 1.0 --epochs 100 --learning-rate 0.1
python run.py --dataset D1 --model-id cdip_0.2_1000_32x64_256_fire3_1.0_0.1 --results-id D1_0.2_1000_32x64_256_fire3_1.0_0.1_3 --input-size 3000 64 --feat-dim 256 --feat-layer fire3 --vshift 3
python run.py --dataset D2 --model-id cdip_0.2_1000_32x64_256_fire3_1.0_0.1 --results-id D2_0.2_1000_32x64_256_fire3_1.0_0.1_3 --input-size 3000 64 --feat-dim 256 --feat-layer fire3 --vshift 3

# feat_dim=512
python train.py --model-id cdip_0.2_1000_32x64_512_fire3_1.0_0.1 --samples-dir ~/samples/deeprec-cvpr20/cdip_0.2_1000_32x64 --feat-dim 512 --feat-layer fire3 --margin 1.0 --epochs 100 --learning-rate 0.1
python run.py --dataset D1 --model-id cdip_0.2_1000_32x64_512_fire3_1.0_0.1 --results-id D1_0.2_1000_32x64_512_fire3_1.0_0.1_3 --input-size 3000 64 --feat-dim 512 --feat-layer fire3 --vshift 3
python run.py --dataset D2 --model-id cdip_0.2_1000_32x64_512_fire3_1.0_0.1 --results-id D2_0.2_1000_32x64_512_fire3_1.0_0.1_3 --input-size 3000 64 --feat-dim 512 --feat-layer fire3 --vshift 3