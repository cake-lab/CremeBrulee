#!env python

import logging
logging.basicConfig(level=logging.NOTSET)
log = logging.getLogger("app")
log.setLevel(logging.DEBUG)

import glob
import argparse
import json
import collections
import itertools
import math
import os

import numpy as np
import pandas as pd

#import sys
#sys.path.append('..')
#import common

#common.getLogger(hide_debug=False)

INDEX_COLUMN = 'HashFunction'


def getFilenames(dir="../datasets/azure"):
  invocations_files = sorted(glob.glob(f"{dir}/invocations_per_function_md*"))
  latency_files = sorted(glob.glob(f"{dir}/function_durations_percentiles*"))
  memory_files = sorted(glob.glob(f"{dir}/app_memory_percentiles*"))
  return (invocations_files, latency_files, memory_files)

def processMinute(df_minute, minute, rng, evenly_distribute=True):
  log.info(f"processMinute {minute}")
  
  # Get just invocations that actually occur (although this is fairly pointless due to later steps)
  df_invocations = df_minute[df_minute > 0]
  
  # Calculate how often invocations occur on average
  time_between_invocations = 60. / df_invocations.sum()

  hash_counts = {} #collections.defaultdict(int)
  # Repeat the hash of the name as many times as the function occurs
  func_invocations = []
  for func_hash, count in zip(df_invocations.index.values, df_invocations.values):
    func_invocations.extend([func_hash for _ in range(count)])
    hash_counts[func_hash] = count
  
  # Combine all lists, correcting for the minute
  invocation_events = list(zip( np.arange((minute-1)*60, minute*60, time_between_invocations), rng.permutation(func_invocations)))
  
  # Sort events by time (they're out of order due to the sampling above)
  return sorted(invocation_events, key=(lambda s: float(s[0]))), hash_counts

def getWorkloadEvents(flags, filenames, rng):
  num_minutes_collected = 0
  hash_counts = collections.defaultdict(int)
  events = []
  
  for day, (f) in enumerate(filenames):
    log.info(f"Parsing file for day {day}...")
    df = pd.read_csv(f, index_col=INDEX_COLUMN)
    
    if not flags.all_triggers:
      df = df[df["Trigger"]=="http"]
    
    for minute in range(1, 1441):
      df_for_minute = df[f"{minute}"]
      new_events, counts = processMinute(df_for_minute, minute+(day*1440), rng=rng)
      events.extend(new_events)
      for (h, c) in counts.items():
        hash_counts[h] += c
      num_minutes_collected += 1
      
      if num_minutes_collected >= flags.num_minutes:
        return events, hash_counts, num_minutes_collected
      
      # Check every hours
      if flags.stop_early:
        if minute % 60 == 0:
          if len(hash_counts.keys()) > flags.min_uniq_funcs:
            return events, hash_counts, num_minutes_collected
          if len(events) > flags.min_uniq_invocations:
            return events, hash_counts, num_minutes_collected
  log.debug(hash_counts)
  return events, hash_counts, num_minutes_collected

def getRealModels(flags):
  # Load models
  with open(flags.input_models_file) as fid:
    models = json.load(fid)
  
  if flags.cost_function == "cost-direct":
    def cost(m): return m["avg_load_latency"]
  elif flags.cost_function == "cost-increase":
    def cost(m): return m["avg_load_latency"] / m["avg_exec_latency"]
  
  models = sorted(models, key=(lambda m: cost(m)), reverse=flags.reverse_correlation)
  log.debug(models)
  return models

def removeOutlierHashes(flags, events, hash_counts):
  #log.debug(events)
  #log.debug(hash_counts)
  #log.debug(f"np.array(hash_counts.values(): {np.array(list(hash_counts.values()))}")
  lower_limit, upper_limit = np.quantile(list(hash_counts.values()), [flags.min_quantile, flags.max_quantile])
  hashes_included = {
    h : c
    for (h, c) in hash_counts.items()
    if ((lower_limit <= c) and (c <= upper_limit))
      and not (flags.remove_one_hit_wonders and c == 1)
  }
  
  
  
  log.debug(f"len(hash_counts) before: {len(hash_counts)}")
  log.debug(f"len(events) before: {len(events)}")
  hash_counts = hashes_included
  events = list(filter((lambda e: e[1] in hash_counts), events))
  log.debug(f"len(hash_counts) after: {len(hash_counts)}")
  log.debug(f"len(events) after: {len(events)}")
  
  
  return events, hash_counts

def mergeModelsAndEvents(flags, real_models, events, hash_counts):
  log.debug(f"Model pairing method: {flags.model_pairing_method}")
  if flags.model_pairing_method == 'random':
    def model_pairing_method():
      while True:
        m = dict(flags.rng_seed.choice(real_models, p=p))
        #log.debug(f"Next model (r): {m['name']}")
        yield m
  elif flags.model_pairing_method == 'round_robin':
    def model_pairing_method():
      i = 0
      while True:
        _ = dict(flags.rng_seed.choice(real_models, p=p)) # Burns a random
        m = real_models[i%len(real_models)]
        #log.debug(f"Next model (rr): {m['name']}")
        i += 1
        yield m
  elif flags.model_pairing_method == 'quantiles':
    def model_pairing_method():
      i = 0
      while True:
        _ = dict(flags.rng_seed.choice(real_models, p=p)) # Burns a random
        curr_index = int( len(real_models) * (i / len(hash_counts)))
        m = real_models[curr_index]
        #log.debug(f"Next model (q): {m['name']}")
        i += 1
        yield m
  get_model = model_pairing_method()
  
  #models_by_hash = { h : dict(flags.rng_seed.choice(real_models)) for (_, h) in events }
  models_by_hash = {}
  # Walk through the hashes by decreasing popularity
  ## This originally just made the names pretty by having more popular models have lower acsension numbers
  for i, h in enumerate(sorted(hash_counts.keys(), key=hash_counts.get, reverse=True)):
    # if flags.keep_correlation:
    #   percentile = int(len(real_models) * (float(i) / len(hash_counts)))
    #   p = np.array([1 if i!=percentile else len(real_models) for i in range(len(real_models))])
    # else:
    #   p = np.array([1 for i in range(len(real_models))])
    
    p = np.array([1 for i in range(len(real_models))])
    p = p / np.sum(p)
    
    if flags.keep_real_models:
      # We'll be using totally real models, down to the name
      #m = dict(flags.rng_seed.choice(real_models, p=p))
      m = next(get_model)
      
    else:
      # We'll be using a fake model name
      model_name = f"model_{i:0{math.ceil(math.log(len(hash_counts.keys()), 10))}d}"
      if flags.mix_model_statistics:
        # Then we want to combine little pieces of real models to make semi-real models
        m = {
          "avg_exec_latency"  : flags.rng_seed.choice(real_models, p=p)["avg_exec_latency"],
          "avg_load_latency"  : flags.rng_seed.choice(real_models, p=p)["avg_load_latency"],
          "avg_unload_latency": flags.rng_seed.choice(real_models, p=p)["avg_unload_latency"],
          "loaded_size"       : flags.rng_seed.choice(real_models, p=p)["loaded_size"],
        }
      else:
        # Then we want to use a single real model as a statistic, but we'll be using a fake name
        m = dict(next(get_model))
      m["name"] = model_name
      log.debug(m)
    
    # Add the selected, or generated, model to the mapping
    models_by_hash[h] = m
  
  models_by_name = {m["name"] : m for m in models_by_hash.values()}
  models = list(sorted(models_by_name.values(), key=(lambda m: m["name"])))
  events = list(map((lambda e: (e[0], models_by_hash[e[1]])), events))
  
  #models = [m.update( {"avg_exec_latency" : (m["avg_exec_latency"]*flags.scale_runtime)} ) for m in models]
  models = [{**m, "avg_exec_latency" : (m["avg_exec_latency"]*flags.scale_runtime) } for m in models]
  
  return models, events
  

def mapHashesToNames(flags, events, models=None):
  log.info("mapHashesToNames")
  hash_name_dict = {}
  hash_popularities = collections.Counter([e[1] for e in events])
  hashes_by_popularity = [h for (h, _) in  hash_popularities.most_common()]
  
  #hash_model_pairs = zip(hashes_by_popularity, itertools.cycle(models))
  hash_model_pairs = zip(hashes_by_popularity, flags.rng_seed.choice(models, size=len(hashes_by_popularity), replace=True))
  hash_name_dict = { h : model["name"] for (h, model) in hash_model_pairs}
  
  return hash_name_dict

  
def updateEventNames(flags, events, hash_name_dict):
  updated_events = [(e[0], hash_name_dict[e[1]]) for e in events]
  return updated_events

def makeRepresentative(flags, events, hash_counts):
  q = (np.arange(1,flags.num_representative_quantiles+1)/flags.num_representative_quantiles)
  counts = np.array(list(hash_counts.values()))
  breaks = np.quantile(counts, q=q)
  intervals = zip(breaks[:-1], breaks[1:])
  
  representative_hashes = []
  for low, high in intervals:
    log.debug(f"({low}, {high})")
    # Note, this is inclusive on both size to ensure that corner cases can be selected still in case there aren't enough requests
    possible_hashes = list(filter( (lambda k: low <= hash_counts[k] and hash_counts[k] <= high), hash_counts.keys() ))
    selected_hashes = flags.rng_seed.choice(list(possible_hashes), size=min([flags.num_from_each_quantile, len(possible_hashes)]), replace=False)
    representative_hashes.extend(selected_hashes)
  
  set_of_hashes = set(representative_hashes)
  events = list(filter((lambda e: e[1] in set_of_hashes), events))
  hash_counts = {
    h: hash_counts[h]
    for h in set_of_hashes
  }
  
  return events, hash_counts
  





def getParser(add_help=False, include_parents=False):
  parser = argparse.ArgumentParser(add_help=add_help,
    #parents=([common.getParser(add_help=False)] if include_parents else [])  
  )
  
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--rng_seed', dest='rng_seed', default=None, type=int)
  
  parser.add_argument('--num_minutes', default=1, type=int)
  parser.add_argument('--stop_early', action="store_true")
  parser.add_argument('--min_uniq_funcs', type=int, default=1000)
  parser.add_argument('--min_uniq_invocations', type=int, default=10000)
  
  
  parser.add_argument('--all_triggers', action="store_true")
  parser.add_argument('--downsample_events', default=1.0, type=float)
  
  parser.add_argument('--input_models_file', default="./real_models.json")
  parser.add_argument('--reverse_correlation', action="store_false")

  #parser.add_argument('--keep_correlation', action="store_true", help="If set then there will be a correlation between popularity and real model selection")
  
  parser.add_argument('--workload_file', default="../workload/workload.txt")
  parser.add_argument('--model_json', default="../workload/models.json")
  
  parser.add_argument('--min_quantile', default=0.0, type=float,
                        help="Minimum quantile of function popularity to include.")
  parser.add_argument('--max_quantile', default=1.0, type=float,
                        help="Minimum quantile of function popularity to include.")
  parser.add_argument('--remove_one_hit_wonders', action="store_true")
  
  parser.add_argument('--mix_model_statistics', action="store_true", help="Uses parts of multiple models as a basis, rather than one per model")
  
  
  parser.add_argument('--scale_runtime', default=1.0, type=float, help="Scale runtime of models by this amount to simulate GPU")
  
  parser.add_argument('--qpm', default=None, type=float, help="Queries per minute (used for testbed)")
  parser.add_argument('--num_requests', default=None, type=int, help="Total number of requests to use.  Overrides qpm for use with restrict_requests in client.py")
  parser.add_argument('--keep_real_models', action="store_true", help="Only uses the original number of models, and assigns functions to them.")
  
  parser.add_argument('--path_to_azure_dataset', default="../datasets/azure")
  
  parser.add_argument('--model_pairing_method', choices=['random', 'round_robin', 'quantiles'], default='random', help="Method by which to select models for a given function when using real models")
  parser.add_argument(
      "--cost_function",
      choices=[
        "cost-direct",
        "cost-increase"
      ],
      default="cost-direct",
      help="Sets the cost function to use between the two approaches explained in 2021-02-17.md"
    )
  
  parser.add_argument('--make_representative', action="store_true", help="As per FaasCache")
  parser.add_argument('--num_representative_quantiles', default=400, type=int)
  parser.add_argument('--num_from_each_quantile', default=1, type=int)
  
  return parser
  
def getFlags():
  flags = getParser().parse_args()
  if not os.path.exists(flags.path_to_azure_dataset):
    log.error("No files in dataset.  Have you downloaded them?")
    exit(8)
  flags.min_uniq_invocations = int(flags.min_uniq_invocations / flags.downsample_events)
  
  flags.rng_seed = np.random.default_rng(flags.rng_seed)
  
  return flags

def removeUnused(models, events):
  model_hashes = {m[f"{INDEX_COLUMN}"] for m in models}
  events = [e for e in events if e[1] in model_hashes]
  
  requested_model_hashes = {e[1] for e in events}
  models = [m for m in models if m[f"{INDEX_COLUMN}"] in requested_model_hashes]
  return models, events

def main():
  
  flags = getFlags()
  
  rng = flags.rng_seed
  
  invocations_files, latency_files, memory_files = getFilenames(flags.path_to_azure_dataset)

  # Find events that contain the selected hashes
  events, hash_counts, num_minutes_collected = getWorkloadEvents(flags, invocations_files[:1], rng)
  
  
  # Load models so combine later
  real_models = getRealModels(flags)
  
  if flags.make_representative:
    events, hash_counts = makeRepresentative(flags, events, hash_counts)
  else:
    events, hash_counts = removeOutlierHashes(flags, events, hash_counts)
  
  # Merge models and events
  models, events = mergeModelsAndEvents(flags, real_models, events, hash_counts)
  
  
  log.debug(f"There are now {len(models)} models")
  log.debug(f"There are now {len(events)} events")
  
  
  
  
  
  # log.info("Downsampling models")
  # if flags.downsample_models:
  #   # This is where we downsample based on models
  #   if flags.downsample_method == "most_popular":
  #     log.info("Fiding most popular models")
  #     #hash_popularities = collections.Counter([e[1] for e in events])
  #     hashes_by_popularity = sorted(hash_counts.keys(), key=hash_counts.get, reverse=True)
  #     log.info("Removing non-popular models")
  #     hashes_to_keep = hashes_by_popularity[:flags.downsample_number]
  #     #models_kept = hashes_found[:flags.downsample_number]
  #     models = [m for m in models if m[INDEX_COLUMN] in hashes_to_keep]
  #   #models, events = removeUnused(models, events)
  # else:
  #   log.info("Not downsampling models")
  # log.debug(f"There are now {len(models)} models")
  # log.debug(f"There are now {len(events)} events")
  
  # log.info("Downsampling workload")
  if (flags.downsample_events < 1.0) and (flags.qpm is None):
    # Downsample as needed by always sample at least one
    num_to_sample = max([1, int(flags.downsample_events * len(events))])
    sampled_events = rng.choice(events, num_to_sample, replace=False)
    # Correcting a weirdness where the samples were both converted to strings
    events = list(filter((lambda _: rng.uniform() < flags.downsample_events), events))
    #models, events = removeUnused(models, events)
  else:
    log.info("Not downsampling events")
  log.debug(f"There are now {len(models)} models")
  log.debug(f"There are now {len(events)} events")
  
  
  # if flags.num_requests is not None:
  #   events = list(sorted(rng.choice(events, size=flags.num_requests), key=(lambda s: s[0])))
  # elif flags.qpm is not None:
  #   pass
  
  if flags.qpm is not None:
    events = list(sorted(rng.choice(events, size=int(flags.qpm*num_minutes_collected)), key=(lambda s: s[0])))
  
  
  
  log.info("Writing out results")
  with open(flags.workload_file, 'w') as fid:
    for i, (event_time, model) in enumerate(events):
      fid.write(f"{i} {event_time:0.5f} {model['name']}\n")
  
  for i, (event_time, event) in enumerate(events[:10]):
    print(f"{event_time:0.5f} - {event['name']}")

  with open(flags.model_json, 'w') as fid:
    fid.write(json.dumps(models, indent=4))
  
  event_resources = map((lambda e: (e[0], e[1]["loaded_size"])), events)
  usages = [np.sum([e[1] for e in events]) for second, events in itertools.groupby(event_resources, key=(lambda e: int(e[0])))]
  log.debug(f"max_usage: {np.max(usages)}")
  log.debug(f"average: {np.average(usages)}")


  return
  
 
  
if __name__ == '__main__':
  main()

