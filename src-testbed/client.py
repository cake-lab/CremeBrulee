#!env bin


import gevent.monkey
gevent.monkey.patch_socket()


import argparse
import time
import sched
import gevent
import gevent.lock
from gevent import subprocess
from io import BytesIO

import numpy as np

import requests
import json

import sys

# Mine
import os
import common
#logger = common.getLogger(f"{os.path.basename(__file__).replace('.py', '')}")
import logging

URL = "http://frontend:5000"
#URL = "http://localhost:32788"

quantiles = [0.5, 0.9, 0.95]

response_times = []
responses = []
responses_by_model = {}


output_semaphore = gevent.lock.Semaphore(value=1)
output_fid = sys.stdout
  
def runInference(id_num, model_name, data):
  
  ts = time.time()
  response = requests.post(f"{URL}/infer/{model_name}", params={"id" : f"{id_num}"}) #, data={"data": data})
  te = time.time()
  try:
    response_json = json.loads(response.text)
    responses.append( response_json )
    sys.stdout.write('.')
    try:
      responses_by_model[model_name].append(response_json)
    except KeyError:
      responses_by_model[model_name] = [response_json]
    recordResponse(response_json)
  except json.decoder.JSONDecodeError:
    logging.error(f"request for {model_name} failed")
    logging.error(response.text)
    sys.stdout.write('X')
    exit(8)
  sys.stdout.flush()
  response_times.append( (te-ts) )

def recordResponse(response):
  output_semaphore.acquire()
  output_fid.write(f"{json.dumps(response)}\n")
  output_fid.flush()
  output_semaphore.release()

def getParser(add_help=True, include_parents=True):
  parser = argparse.ArgumentParser(add_help=add_help,
    parents=([common.getParser(add_help=False)] if include_parents else [])  
  )
  
  parser.add_argument('--restrict_requests', action="store_true")
  parser.add_argument('--identifier', default="test")
  
  return parser


def main():
  args = getParser().parse_args()
  
  data = common.getData()
  
  workload = []
  with open(args.workload_file, 'r') as workload_fid:
    workload_events = [tuple(s.strip().split(' ')) for s in workload_fid.readlines()]
    workload_events = list(map( (lambda e: (e[0], float(e[1]), e[2])), workload_events ))
  
  global output_fid
  output_fid = open(os.path.join("/etc/results/", f"{args.identifier}.log"), 'w')
  
  
  if args.restrict_requests:
    for id_num, event_time, model_name in workload_events:
      runInference(id_num, model_name, data)
  else:
    threads = []
    for id_num, event_time, model_name in workload_events:
      threads.append(gevent.spawn_later(event_time, runInference, id_num, model_name, data))
    gevent.joinall(threads)
  sys.stdout.write('\n')
  sys.stdout.flush()
  
  #print(responses)
  #print(np.quantile(np.array(response_times), [0.5, 0.9, 0.95]))
  
  print("Overall latency")
  print(np.quantile(np.array([r["overall_latency"] for r in responses]), quantiles))
  for model_name, model_responses in responses_by_model.items():
    print(f"{model_name} : {np.quantile(np.array([r['overall_latency'] for r in model_responses]), quantiles)}")
  
  for q in quantiles:
    print(f"{q} : {np.mean(np.array([np.quantile(np.array([r['overall_latency'] for r in model_responses]), [q]) for model_responses in responses_by_model.values()]))}")
  
  print("")
  print("Queue delay")
  print(np.quantile(np.array([r["queue_delay"] for r in responses]), quantiles))
  for model_name, model_responses in responses_by_model.items():
    print(f"{model_name} : {np.quantile(np.array([r['queue_delay'] for r in model_responses]), quantiles)}")
  
  for q in quantiles:
    print(f"{q} : {np.mean(np.array([np.quantile(np.array([r['queue_delay'] for r in model_responses]), [q]) for model_responses in responses_by_model.values()]))}")
  
  
  print(f"Average: {np.average(np.array([r['queue_delay'] for r in responses]))}")
  
  output_fid.close()
  return
  
  
  

if __name__ == '__main__':
  main()