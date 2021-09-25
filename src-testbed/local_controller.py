import base64
import subprocess
import json
from pathlib import Path
import random
import requests
import redis
from redis import Redis
import time
import threading
import collections
from threading import Lock
from threading import Event
from queue import Queue

# Mine
import exceptions
import abstractions

import logging

import os
import common




class LocalController:
  
  def __init__(self, redis_server='redis-server', redis_port=6379):
    
    # Set Up Redis database connection
    self.db = Redis(host=redis_server, port=redis_port)
    try:
      self.db.ping()
    except redis.exceptions.ConnectionError as e:
      print(f"Error connecting to Redis server @ {redis_server}.  Is it started?")
      print(e)
      exit(8)
    
    # Get basic information from Redis
    ## Note: might not be available
    self.model_names = [w.decode() for w in self.db.smembers(f"{common.MODEL_NAME_KEY}")]
    
    # Set up structures
    self.models_by_name = {}
    
    
    # Set up lists of models and workers
    self.workers_by_url = {}
    self.model_placements = common.ModelPlacements()
    
    ## Set up locks for models and workers
    self.workers_lock = threading.RLock()
    self.model_lock = threading.RLock()
    
    
    
    # Events for when models are available
    # TODO: purge this half-assed approach
    self.model_available_event = {
      model_name : Event()
      for model_name in self.model_names
    }
    
    
    # Subscribe to future updates to models and workers
    pubsub = self.db.pubsub()
    pubsub.subscribe(**{f'{common.KEYSPACE_PREFIX}{common.WORKER_URL_KEY}': self.updateWorkerList})
    pubsub.psubscribe(**{f'{common.KEYSPACE_PREFIX}{common.MODEL_PLACEMENT_PREFIX}*': self.updateModelPlacement})
    # Note: There is an option sleep_time argument to run_in_thread that I'm not totally sure what it does
    self.subThread = pubsub.run_in_thread()
    
    self.metrics = common.Metrics(self.db, common.MODEL_STAT_PREFIX, ["requests_submitted", "requests_completed", "open_requests"], ["requests_submitted"])
    
    print("Controller Initialized.")
    
  
  
  ######################
  ## Worker Functions ##
  ######################
  def addWorker(self, worker_url):
    logging.info("local_controller: addWorker start")
    logging.info(f"Adding worker: {worker_url}")
    
    worker_url = common.fixWorkerURL(worker_url)
    
    if worker_url in self.workers_by_url.keys() and self.workers_by_url[worker_url] is not None:
      logging.warn(f"Already added worker @ '{worker_url}'")
      return
      
    new_worker = abstractions.WorkerAbstraction(worker_url)
    # TODO: Add error checking for the new worker.  Check to see if it is active for instance
    # TODO: Get list of models in the repo on the new worker
    self.workers_by_url[worker_url] = new_worker
    
    logging.info("local_controller: addWorker end")
      
  def delWorker(self, worker_url):
    logging.debug("delWorker start")
    logging.debug(f"Deling worker: {worker_url}")
    #self.workers_by_url[worker_url].removeWorker()
    del self.workers_by_url[worker_url]
  ######################
  
  
  #####################
  ## Model Functions ##
  #####################
  def getModelByName(self, model_name):
    try:
      model = self.models_by_name[model_name]
    except KeyError:
      model = abstractions.ModelAbstraction(model_name)
      self.models_by_name[model_name] = model
    return model
  
  def loadModelOntoWorker(self, model_name, worker_url):
    logging.info("loadModelOntoWorker() start")
    worker_url = common.fixWorkerURL(worker_url)
    model = self.getModelByName(model_name)
    
    if self.workers_by_url[worker_url].loadModel(model):
      self.model_placements.addModelToWorker(worker_url, model.model_name)
    else:
      logging.error(f"Failed to load model '{model.model_name}' to worker @ {worker_url}")
      

  def unloadModelFromWorker(self, model_name, worker_url):
    logging.info("unloadModelFromWorker() start")
    worker_url = common.fixWorkerURL(worker_url)
    model = self.getModelByName(model_name)
    
    if self.workers_by_url[worker_url].unloadModel(model):
      self.model_placements.removeModelFromWorker(worker_url, model.model_name)
    else:
      logging.error(f"Failed to unload model '{model.model_name}' to worker @ {worker_url}")
    
    
  #####################
  
  
  #########################
  ## Statistic Functions ##
  def recordRequestEntry(self, model_requested):
    self.metrics.incrementMetricBy("requests_submitted", model_requested)
    self.metrics.incrementMetricBy("open_requests", model_requested, +1)
    
  def recordRequestExit(self, model_requested):
    self.metrics.incrementMetricBy("requests_completed", model_requested)
    self.metrics.incrementMetricBy("open_requests", model_requested, -1)
  #########################
  
  
  #########################
  ## Inventory Functions ##
  #########################
  def updateWorkerList(self, message=None):
    logging.debug("updateWorkerList start")
    
    orig_message = message
    message = orig_message["data"].decode()
    
    if message not in ["sadd", "srem"]:
      logging.error(f"Unknown message received: {message}")
      logging.error(f"Full message: {orig_message}")
      return
    
    logging.debug("Acquiring workers_lock")
    try:
      with self.workers_lock:
        redis_worker_urls = list(map( (lambda b: common.fixWorkerURL(b.decode())), self.db.smembers(f'{common.WORKER_URL_KEY}')))
          
        if message == "sadd" or message is None:
          # Workers were added
          worker_urls_add = list(set(redis_worker_urls) - set(self.workers_by_url.keys()))
          for worker_url in worker_urls_add:
            self.addWorker(worker_url)
          
        elif message == "srem" or message == "del":
          # Workers were removed
          worker_urls_del = list(set(self.workers_by_url.keys()) - set(redis_worker_urls))
          for worker_url in worker_urls_del:
            self.delWorker(worker_url)

    finally:
      logging.debug("Releasing workers_lock")
  
  
  def updateModelPlacement(self, message):
    logging.debug("updateModelPlacement start")
    
    orig_message = message
    message = orig_message["data"].decode()
    channel = orig_message["channel"].decode()
    
    if message not in ["sadd", "srem"]:
      logging.warn(f"Unknown message received: {message}")
      logging.warn(f"Full message: {orig_message}")
      return
    
    model_name = channel.replace(f"{common.KEYSPACE_PREFIX}{common.MODEL_PLACEMENT_PREFIX}","")
    logging.info(f"Update for model: {message} {model_name}")
    
    try:
      redis_model_placements = list(map( (lambda b: common.fixWorkerURL(b.decode())), self.db.smembers(f"{common.MODEL_PLACEMENT_PREFIX}{model_name}")))
      current_placements = self.model_placements.getWorkersFromModel(model_name)
      logging.info(f"redis_model_placements: {redis_model_placements}")
      logging.info(f"current_placements: {current_placements}")
      
      logging.debug("Acquiring model lock")
      with self.model_lock:
        if message == "sadd":
          placements_to_add = list(set(redis_model_placements) - set(current_placements))
          for worker_url in placements_to_add:
            logging.info(f"Adding {model_name} to {worker_url}")
            self.loadModelOntoWorker(model_name, worker_url)
        elif message == "srem":
          placements_to_del = list(set(current_placements) - set(redis_model_placements))
          for worker_url in placements_to_del:
            logging.debug(f"Del'ing {model_name} from {worker_url}")
            self.unloadModelFromWorker(model_name, worker_url)
    finally:
      logging.debug("Releasing model lock")
  #########################
  
  
  @common.timing
  def infer(self, inference_request):
    logging.info(f"infer({inference_request.model_name})")
    self.recordRequestEntry(inference_request.model_name)
    
    logging.info(f"inference_request: {inference_request}")
    
    model = self.getModelByName(inference_request.model_name)
    
    accepted_by_worker = False
    while not accepted_by_worker:
      while len(model.placements) == 0:
        self.requestModelPlacement(inference_request)
        if not model.is_available.wait(common.PLACEMENT_POLL_INTERVAL):
          logging.warning(f"Waiting on placement for {model}")
      if len(model.placements) > 0:
        worker = random.choice(list(model.placements))
        accepted_by_worker = worker.infer(inference_request)
      else:
        logging.warning("Worker picked already reassigned, requests are coming in too fast?")
    
    if not inference_request.complete.wait(common.TIMEOUT_IN_SECONDS):
      raise exceptions.InferenceFailedException("Inference Failed to respond")
    logging.info(f"inference request after inference: {inference_request}")
    
    self.recordRequestExit(inference_request.model_name)
    logging.info(f"response: {inference_request}") #.getResponse()}")
    return inference_request.getResponse()
    
  @common.gather_info
  def requestModelPlacement(self, inference_request):
    logging.info(f"requestModelPlacement({inference_request.model_name})")
    # In case oracle is being used
    self.db.set(f"latest_request_id", f"{inference_request.id}")
      
    # Request model placement
    results = self.db.sadd(f"{common.PLACEMENT_REQUEST_KEY}", f"{inference_request.model_name}")
    if results >= 1:
      # If this is the first time this model has been requested
      inference_request.markModelMiss()
      
    
    
  
  
if __name__ == '__main__':
  common.getLogger(f"{os.path.basename(__file__).replace('.py', '')}")
  local_controller = LocalController()
  while True:
    time.sleep(0.001)
