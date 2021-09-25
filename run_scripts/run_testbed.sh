#!/bin/bash

RNG_SEED=0
NUM_LOOPS=1
NUM_MINUTES=1
WORKER_MEMORY=4
POLICY="cremebrulee-oracle"
QPM=6


cd ../src-testbed

if true ; then
  echo "Generating workload"
  (
    cd ../src-workload 
    python -u parse_azure.py \
      --rng_seed $RNG_SEED \
      --num_minutes 1 \
      --max_quantile 0.95 \
      --remove_one_hit_wonders \
      --downsample_events 0.001 \
      --input_models_file ../measurements/models.short.json \
      --keep_real_models \
  )
else
  echo "Skipping workload generation"
fi

common_args="
    --rng_seed 0
    --run_series $(git rev-parse --verify HEAD | head -c 7)
    --real_model_repo_path ../models/models
    --cost_function cost-direct
    --num_workers_to_add 1
    --show_debug
    --workload_file ../workload/workload.txt
    --model_description_file ../workload/models.json
    "

    
echo "Policy under test: $POLICY-$WORKER_MEMORY-random"
#(docker kill $(docker container ls | awk '{print $1}') || docker system prune -f ) >/dev/null
python3 -u runEdgeController.py            \
  $common_args                        \
  --worker_memory $WORKER_MEMORY \
  --model_eviction_algorithm $POLICY    \
  --run_identifier "$POLICY.${WORKER_MEMORY}gb.random" \
    

#docker kill $(docker container ls | tail -n+2 | awk '{print $1}')