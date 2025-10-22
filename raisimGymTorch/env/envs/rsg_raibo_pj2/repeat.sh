SET=$(seq 0 5)
for i in $SET
do
    python3 raisimGymTorch/env/envs/rsg_raibo_pj2/runner.py
done