#!env python

import logging
import argparse

import abstractions

import common
import time

import random


def getParser(add_help=True, include_parents=True):
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--models_to_measure', nargs='+')
  parser.add_argument('--worker_url', default='worker_0:50051')
  
  parser.add_argument('--num_cycles', default=10, type=int)
  parser.add_argument('--num_requests', default=10, type=int)
  
  
  return parser




def main():
  flags = getParser().parse_args()
  
  models_to_measure = [abstractions.ModelAbstraction(model_name) for model_name in flags.models_to_measure]
  
  worker = abstractions.WorkerAbstraction(flags.worker_url)
  
  data = common.getData()
  
  for i in range(flags.num_cycles):
    logging.info(f"Test loop {i}")
    print(f"Test loop {i}")
    random.shuffle(models_to_measure)
    for model in models_to_measure:
      logging.info(f"  model: {model}")
      print(f"  model: {model}")
      worker.loadModel(model)
      time.sleep(0.1)
      for j in range(flags.num_requests):
        logging.info(f"    Submitting request {j+1} of {flags.num_requests}")
        request = common.InferenceRequest(model.model_name, data)
        worker.infer(request)
        request.complete.wait()
      time.sleep(0.1)
      worker.unloadModel(model)
      time.sleep(0.1)
      
  print("Complete")
  

if __name__ == '__main__':
  main()
  print("exiting...")
  exit()