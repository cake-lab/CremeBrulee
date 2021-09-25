#!env python

import argparse
import threading
import redis
import requests
import collections
import re
import time

import numpy as np

# Mine
from worker_interface import WorkerClient
import exceptions

import common
logger = common.getLogger(f"{__file__.replace('.py', '')}")



class MetricManager(object):
  def __init__(self, redis_server, redis_port, time_between_updates, *args, **kwargs):
    self.db = redis.Redis(host=redis_server, port=redis_port)
    self.time_between_updates = time_between_updates
    
    self.worker_clients_by_url = {}
    self.updateWorkers()
    
    
    # Set "updating" flag to indicate it is not currently updating
    self.db.mset({common.UPDATING_FLAG_NAME : 0})
  
  def updateWorkers(self):
    redis_worker_urls = [common.fixWorkerURL(w.decode()) for w in self.db.smembers('workers')]
    new_worker_urls = set(redis_worker_urls) - set(self.worker_clients_by_url.keys())
    
    for worker_url in new_worker_urls:
      self.worker_clients_by_url[worker_url] = WorkerClient(worker_url)
      
    
  def updateMetrics(self):
    logger.info("Enter updateMetrics()")
    # Schedule next update
    if self.time_between_updates > 0:
      threading.Timer(self.time_between_updates, self.updateMetrics).start()
    
    # Check whether the previous metrics have been updated.  If not, skip this time
    if self.db.get(common.UPDATING_FLAG_NAME) == 1:
      logger.warning("Previous updateMetrics() still has update lock")
      return
      
    # Update the metrics
    ## Set "updating" flag in redis
    logger.info("Acquiring lock")
    self.db.mset({common.UPDATING_FLAG_NAME : 1})
    
    self.updateWorkers()
    
    try:
      
      worker_stats = {}
      model_stats = {}
      
      ## Get a List of current workers
      #metrics_by_worker = {
      #    worker_url : worker_client.getMetrics()
      #    for (worker_url, worker_client) in self.worker_clients_by_url.items()
      #}
      
      #fields_to_get = [""]
      
      for worker_url in self.worker_clients_by_url.keys():
        worker_prefix = f"{common.WORKER_STAT_PREFIX}{common.stripWorkerURL(self.worker_url)}"
        
      
      logger.debug(f"metrics_by_worker: {metrics_by_worker}")
      
      
      ############################
      # Process Model Statistics #
      ############################
      
      ## Get a list of models used
      model_names = []
      for worker_metrics in metrics_by_worker.values():
        model_names.extend(worker_metrics["model_info"].keys())
      model_names = list(set(model_names))
      
      logger.debug(f"model_names: {model_names}")
      
      
      ############################
      ############################
      
      ## Push Results to redis
      self.updateRedis(worker_stats, model_stats)
    finally:
      ## Unset "updating" flag in redis
      self.db.mset({common.UPDATING_FLAG_NAME : 0})
      logger.info("Released lock")
  
  def updateRedis(self, worker_stats, model_stats):
    logger.info("updateRedis()")
    logger.debug(f"worker_stats: {worker_stats}")
    logger.debug(f"model_stats: {model_stats}")
    
    ### Open a pipeline so all updates can happen at once
    pipe = self.db.pipeline()
      
    for worker in worker_stats.keys():
      # For each worker, grab all the statistics and submit them all at once
      for statistic in worker_stats[worker]:
        if not isinstance(worker_stats[worker][statistic], dict):
          pipe.set( f"{common.WORKER_STAT_PREFIX}{worker}{common.DB_FIELD_SEPARATOR}{statistic}", worker_stats[worker][statistic])
        
    for model in model_stats.keys():
      # For each worker, grab all the statistics and submit them all at once
      for statistic in model_stats[model]:
        pipe.set( f"{common.MODEL_STAT_PREFIX}{model}{common.DB_FIELD_SEPARATOR}{statistic}", model_stats[model][statistic] )
    
    results = pipe.execute()
    logger.debug(f"Pipelined setting results: {results}")
      
  
      


def parseArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument('--redis_server', default="redis-server",
            help='Hostname of the REDIS server')
  parser.add_argument('--redis_port', default=6379, type=int,
            help='Port of the REDIS server')
  
  parser.add_argument('--time_between_updates', default=1,
            help="Time between updates in seconds.")
  
  args = parser.parse_args()
  return args


def main():
  args = parseArgs()
  
  metrics_manager = MetricManager(args.redis_server, args.redis_port, args.time_between_updates)
  
  # Start update cycle
  metrics_manager.updateMetrics()


if __name__ == '__main__':
  main()