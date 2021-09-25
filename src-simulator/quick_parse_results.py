#!env python

import sys
import json
import pandas as pd

log_file = sys.argv[1]
stats_file = sys.argv[2]

def parseStats(model_stats_json):
  stats = json.load(open(model_stats_json))
  return { m["name"] : m for m in stats}

model_stats = parseStats(stats_file)

with open(log_file) as fid:
  responses = []
  for line in fid.readlines():
    line = line.strip()
    line = line.replace('\'', '\"')
    line = line.replace('True', '"True"')
    line = line.replace('False', '"False"')
    #logging.debug(line)
    responses.append(json.loads(line))
  df = pd.DataFrame(responses)

#print(df.head())
#print(df.mean()["model_miss"])
#print(f'load avg: {df[["model", "model_miss"]].apply((lambda row: model_stats[row[0]]["avg_load_latency"] if row[1] else 0), axis=1).mean():0.3f}')
#print(f'increase avg: {df[["model", "model_miss"]].apply((lambda row: (model_stats[row[0]]["avg_load_latency"] / model_stats[row[0]]["avg_exec_latency"]) if row[1] else 1), axis=1).mean():0.3f}')

#print(f'load std: {df[["model", "model_miss"]].apply((lambda row: model_stats[row[0]]["avg_load_latency"] if row[1] else 0), axis=1).std():0.3f}')
#print(f'increase std: {df[["model", "model_miss"]].apply((lambda row: (model_stats[row[0]]["avg_load_latency"] / model_stats[row[0]]["avg_exec_latency"]) if row[1] else 1), axis=1).std():0.3f}')



print(f"miss rate: {df.mean()['model_miss']:0.2%}")
print(f"avg queue delay: {df[['model', 'model_miss']].apply((lambda row: model_stats[row[0]]['avg_load_latency'] if row[1] else 0), axis=1).mean():0.3f}s")
print(f"avg increase: {df[['model', 'model_miss']].apply((lambda row: (model_stats[row[0]]['avg_load_latency'] / model_stats[row[0]]['avg_exec_latency']) if row[1] else 1), axis=1).mean():0.3f}x")
