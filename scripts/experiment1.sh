# Experiment 1: single-reconstruction

# proposed: results are saved in results/proposed
python generate_samples.py --neutral-thresh 0.2 --dataset cdip --max-samples 1000 --sample-size 32 64 --save-dir ~/samples/deeprec-cvpr20/cdip_0.2_1000_32x64
python train.py --model-id cdip_0.2_1000_32x64_128_fire3_1.0_0.1 --samples-dir ~/samples/deeprec-cvpr20/cdip_0.2_1000_32x64 --feat-dim 128 --feat-layer fire3 --margin 1.0 --epochs 100 --learning-rate 0.1
# vshift=3
python run.py --dataset D1 --model-id cdip_0.2_1000_32x64_128_fire3_1.0_0.1 --results-id D1_0.2_1000_32x64_128_fire3_1.0_0.1_3 --input-size 3000 64 --feat-dim 128 --feat-layer fire3 --vshift 3
python run.py --dataset D2 --model-id cdip_0.2_1000_32x64_128_fire3_1.0_0.1 --results-id D2_0.2_1000_32x64_128_fire3_1.0_0.1_3 --input-size 3000 64 --feat-dim 128 --feat-layer fire3 --vshift 3
# vshift=0
python run.py --dataset D1 --model-id cdip_0.2_1000_32x64_128_fire3_1.0_0.1 --results-id D1_0.2_1000_32x64_128_fire3_1.0_0.1_0 --input-size 3000 64 --feat-dim 128 --feat-layer fire3 --vshift 0
python run.py --dataset D2 --model-id cdip_0.2_1000_32x64_128_fire3_1.0_0.1 --results-id D2_0.2_1000_32x64_128_fire3_1.0_0.1_0 --input-size 3000 64 --feat-dim 128 --feat-layer fire3 --vshift 0

# sib18: results are saved in results/sib18
python generate_samples.py --neutral-thresh 0.2 --dataset cdip --max-samples 1000 --sample-size 32 32 --save-dir ~/samples/deeprec-cvpr20/sib18
python train_sib18.py --model-id sib18 --samples-dir ~/samples/deeprec-cvpr20/sib18
# vshift=0
python run_sib18.py --dataset D2 --results-id D2-0 --vshift 0
python run_sib18.py --dataset D1 --results-id D1-0 --vshift 0
# vshift=10
python run_sib18.py --dataset D1 --results-id D1-10 --vshift 10
python run_sib18.py --dataset D2 --results-id D2-10 --vshift 10