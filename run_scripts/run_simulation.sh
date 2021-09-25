#!/bin/bash

declare -a eviction_algos=(
    #"random"
    "belady"
    #"cremebrulee-oracle"
    #"cremebrulee"
    #"cremebrulee --oracle"
    #"popularity"
    #"loadtime"
    #"recent"
  ) 

RNG_SEED=0
POLICY="cremebrulee-oracle"
WORKER_MEMORY=1024

cd ../src-simulator


if true ; then
  echo "Generating workload"
  (
    cd ../src-workload 
    python -u parse_azure.py \
      --rng_seed $RNG_SEED \
      --num_minutes 1 \
      --max_quantile 0.95 \
      --remove_one_hit_wonders \
      --downsample_events 0.1 \
      --input_models_file ../measurements/models.short.json \
      --keep_real_models \
  )
else
  echo "Skipping workload generation"
fi


common_args="
  --run_series $(git rev-parse --verify HEAD | head -c 7)${additional_tags}
  --rng_seed $RNG_SEED
  --workload_file ../workload/workload.txt
  --model_description_file ../workload/models.json
  --num_workers_to_add 1
  --results_dir results
  --cost_function cost-direct
  --worker_memory ${WORKER_MEMORY}
"

echo "Policy under test: $POLICY $WORKER_MEMORY"
python simulation.py \
          $common_args \
          --model_eviction_algorithm $POLICY \
          --run_identifier ${POLICY}.${WORKER_MEMORY}gb \
          2>&1 #/dev/null"
echo "-----------------------"


exit 0
