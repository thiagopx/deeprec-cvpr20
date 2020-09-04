# Experiment 2: multi-reconstruction
python run_multi.py --dataset D1 --model-id cdip_0.2_1000_32x64_128_fire3_1.0_0.1 --results-id D1 --input-size 3000 64 --feat-dim 128 --feat-layer fire3 --vshift 3
python run_multi.py --dataset D2 --model-id cdip_0.2_1000_32x64_128_fire3_1.0_0.1 --results-id D2 --input-size 3000 64 --feat-dim 128 --feat-layer fire3 --vshift 3

python run_sib18_multi.py --dataset D1 --results-id D1  --vshift 10
python run_sib18_multi.py --dataset D2 --results-id D2  --vshift 10