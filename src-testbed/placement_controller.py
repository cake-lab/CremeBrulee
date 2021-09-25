#!env python

import os
import sys
import argparse
import time
import signal
import numpy as np

import collections
import threading
from threading import RLock as Lock
import copy
import queue

import functools

## Mine
import exceptions

import logging
import common

import eviction_policies

class RedisInterface_PlacementController(common.RedisInterface):
  def __init__(self, placement_controller, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.placement_controller = placement_controller
    
    
    self._got_state = False
    self.getStateFromRedis()
    
    self.syncRedis()
    
    pubsub = self.db.pubsub()
    pubsub.subscribe(**{f'__keyspace@0__:{common.PLACEMENT_REQUEST_KEY}': self.acuteUpdate})
    
    pubsub.subscribe(**{f'{common.KEYSPACE_PREFIX}{common.WORKER_URL_KEY}': self.updateWorkerList})
    pubsub.subscribe(**{f'{common.KEYSPACE_PREFIX}{common.MODEL_NAME_KEY}': self.updateModelList})
    
    self.subThread = pubsub.run_in_thread()
  
  def acuteUpdate(self, message, *args, **kwargs):
    """
    Reacts to the need for a model placement immediately.
    If no workers are in system then will exit early.
    Additionally, if there has already been a worker allocated for the model then will exit early.
    """
    logging.info(f"acuteUpdate()")
    
    action = message["data"].decode()
    
    if action == "sadd":
      models_to_add = [m.decode() for m in self.db.smembers(f"{common.PLACEMENT_REQUEST_KEY}")]
    else:
      return
    
    
    self.placement_controller.setModelInfo(self.getModelInfo())
    try:
      request_id = int(self.db.get(f"latest_request_id").decode())
    except AttributeError:
      request_id = 0
    models_not_placed = self.placement_controller.requestToAddModels(models_to_add, request_id)
    
    models_added = list(set(models_to_add) - set(models_not_placed))
    
    pipe = self.db.pipeline()
    for model_name in models_added:
      pipe.srem(f"{common.PLACEMENT_REQUEST_KEY}", f"{model_name}")
    results = pipe.execute()
    logging.info(f"Removed {len(results)} from {common.PLACEMENT_REQUEST_KEY}")
    
    # Push decisions to redis
    self.syncRedis()
  
    
  ###########################
  ## Information Functions ##
  ###########################
  
  def getModelInfo(self):
    model_info = {}
    for model in self.placement_controller.model_placements.getModels():
      open_requests = self.db.get(f"{common.MODEL_STAT_PREFIX}{model}{common.DB_FIELD_SEPARATOR}open_requests")
      last_used = self.db.get(f"{common.MODEL_STAT_PREFIX}{model}{common.DB_FIELD_SEPARATOR}last_used")
      requests_submitted = self.db.get(f"{common.MODEL_STAT_PREFIX}{model}{common.DB_FIELD_SEPARATOR}requests_submitted")
      expected_load_latency = self.db.get(f"{common.MODEL_STAT_PREFIX}{model}{common.DB_FIELD_SEPARATOR}avg_load_latency")
      expected_exec_latency = self.db.get(f"{common.MODEL_STAT_PREFIX}{model}{common.DB_FIELD_SEPARATOR}avg_exec_latency")
      loaded_size = self.db.get(f"{common.MODEL_STAT_PREFIX}{model}{common.DB_FIELD_SEPARATOR}loaded_size")
      
      model_info[model] = {
        "open_requests"     : 0 if open_requests is None else float(open_requests.decode()),
        "last_used"         : 0 if last_used is None else float(last_used.decode()),
        "requests_submitted"        : 0 if requests_submitted is None else float(requests_submitted.decode()),
        "placement_count"   : len(self.placement_controller.model_placements.getWorkersFromModel(model)),
        "load_latency"      : 0 if expected_load_latency is None else float(expected_load_latency.decode()),
        "exec_latency"      : 0 if expected_exec_latency is None else float(expected_load_latency.decode()),
        "loaded_size"      : 0 if loaded_size is None else float(loaded_size.decode()),
      }

      model.load_latency = 0 if expected_load_latency is None else float(expected_load_latency.decode())
      model.loaded_size = 0 if loaded_size is None else float(loaded_size.decode())
    logging.debug(f"model_info: {model_info}")
    return model_info  
    
  
  # TODO: These all need to be integrated better
  
  #####################
  ## REDIS Functions ##
  #####################
  def getStateFromRedis(self):
    """
    Pulls current state from redis.
    Should only be called once at the very beginning
    """
    if not self._got_state:
      self.updateWorkerList()
      self.updateModelList()
      
      for model in self.placement_controller.model_placements.getModels():
        for worker in map( (lambda b: b.decode()), self.db.smembers(f"{common.MODEL_PLACEMENT_PREFIX}{model}")):
          self.placement_controller.model_placements.addModelToWorker(worker, model)
      self._got_state = True
    else:
      logging.warn("Should only get state on initial setup")
    
  def syncRedis(self):
    """
    Pushes the changes made locally to redis.  Note, we are the golden source, except for the first sync (which is handled in __init__.
    We just push our changes blindly.
    """
    logging.info("syncRedis(..) start")
    
    self.placement_controller.model_placements.sync()
    
    ## First find workers removed from each model
    pipe = self.db.pipeline()
    while not self.placement_controller.model_placements.removals.empty():
      (worker, model) = self.placement_controller.model_placements.removals.get()
      pipe.srem(f"{common.MODEL_PLACEMENT_PREFIX}{model}", f"{worker}")
    results = pipe.execute()
    
    
    ## Next find workers added to each model
    pipe = self.db.pipeline()
    while not self.placement_controller.model_placements.additions.empty():
      (worker, model) = self.placement_controller.model_placements.additions.get()
      logging.info(f"Adding {model} to {worker}")
      pipe.sadd(f"{common.MODEL_PLACEMENT_PREFIX}{model}", f"{worker}")
    results = pipe.execute()
    
    
    logging.info("syncRedis(..) end")
  
  def updateWorkerList(self, message=None):
    logging.info(f"updateWorkerList({message})")
    for worker in [w.decode() for w in self.db.smembers(f"{common.WORKER_URL_KEY}")]:
      self.placement_controller.model_placements.addWorker(worker)
  def updateModelList(self, message=None):
    for model_name in [m.decode() for m in self.db.smembers(f"{common.MODEL_NAME_KEY}")]:
      self.placement_controller.model_placements.addModel(common.ModelPlacements.Model(model_name))
  
class PlacementController(object):
  request_indices = collections.defaultdict(collections.deque)
  rng = None
  
  def __init__(self, flags, *args, **kwargs):
    logging.info("PlacementController: __init__(...)")
    
    self.flags = flags
    logging.debug(f"Seeding with '{flags.rng_seed}'")
    self.rng = np.random.default_rng(flags.rng_seed)
    self.model_placements = common.ModelPlacements()
    
    # This is for collecting whatever information is the latest available on models
    self.model_info = {}
    self.latest_request_id = -1
    self.models_kept_alive = []
    self.placement_lock = threading.Lock()
    
    
    eviction_policies.State.setCacheSize(self.flags.worker_memory)
    
    
    self.do_proactive = kwargs["do_proactive"] if "do_proactive" in kwargs else False
    self.do_reactive = kwargs["do_reactive"] if "do_reactive" in kwargs else False
    if (not self.do_reactive) and (not self.do_proactive):
      logging.warning(f"{self.__class__.__name__} will not doing anything!")
    
    #####################
    ## Model Valuation ##
    #####################
    # TODO: This should actually be set here, but model_info is still blank.  This should be available in the real version
    self.ideal_caches = None
    
    if flags.scale_func == "rank":
      self.scale_func = (lambda v, vals: sorted(vals).index(v) / (len(vals) - 1) )
    elif flags.scale_func == "minmax":
      self.scale_func = (lambda v, vals: (v - vals.min()) / (vals.max() - vals.min()))
    
    if flags.weight_func == "identity":
      self.weight_func = (lambda v, vals: v) # Identify function
    elif flags.weight_func == "favor_large":
      self.weight_func = (lambda v, vals: (np.exp(vals) / np.sum(np.exp(vals), axis=0))[vals.index(v)] )
    elif flags.weight_func == "favor_small":
      self.weight_func = (lambda v, vals: (((1./(1 + np.exp(-vals))) - 0.5) / ((1./(1 + np.exp(-1))) - 0.5))[vals.index(v)] )
    
      
    
    ##################
    ### Prefetching ##
    ##################
    
    self.do_keep_alive = kwargs["keep_alive"] if "keep_alive" in kwargs else False
    self.do_model_cooling = kwargs["model_cooling"] if "model_cooling" in kwargs else False
    self.do_model_warming = kwargs["model_warming"] if "model_warming" in kwargs else False
    
    self.do_model_annealing = kwargs["model_annealing"] if "model_annealing" in kwargs else False
    self.force_remove_periodic = kwargs["force_remove_periodic"] if "force_remove_periodic" in kwargs else False
      
    if self.do_proactive:
      self.periodicThread = threading.Timer(common.PLACEMENT_PERIOD, self.periodicUpdate)
      self.periodicThread.start()
    ###########################
    
    
    # Gather information for oracle
    with open(flags.workload_file) as fid:
      self.models_that_will_be_requested = [s.strip().split()[2] for s in fid.readlines()]
    
    for index, model in enumerate(self.models_that_will_be_requested):
      self.__class__.request_indices[model].append(index)
    
    
  ###################
  ## API Functions ##
  ###################
  def setModelInfo(self, model_info):
    self.model_info = model_info
    self.model_info["last_updated"] = time.time()
  
  @common.timing
  def requestToAddModels(self, models_to_add, request_id=-1):
    """
    addModels(self, models_to_add):
    
    Called to ask for a number of models to be placed.
    
    Prior:
      Model information should be updated
    After:
      Model placements will be current for all the models requested to be added.
    """
    logging.info(f"Request: {request_id}, models_to_add: {[str(m) for m in models_to_add]}")
    
    self.latest_request_id = max([request_id, self.latest_request_id])
    
    if time.time() - self.model_info["last_updated"] > common.MODEL_INFO_FRESHNESS:
      logging.warn("Model Information out of date!  Did you remember to call setModelInfo(...)?")
    
    with self.placement_lock:
      
      # Update information about the cluster
      self.updateKeepAliveModels()
      models_in_cache = self.model_placements.getModelsInCache() # [model_name for (model_name, model_placements) in self.model_placements.items() if len(model_placements) > 0]
      
      new_models = list(set(models_to_add) - set(models_in_cache))
      # Check whether models to add are already in the cache
      if len(new_models) == 0:
        return []
      
      
      if len(self.model_placements.getWorkers()) == 0:
        # Weirdly, we have no workers to place model on yet.  Dunno, weird, right?
        # We are leaving the request in the queue, so we should maybe come back to it in the future
        # Or call it during the periodic update
        # TODO
        logging.warning("No workers available for model placement")
        return new_models
      
      
      models_not_placed = []
      
      ## We are looping through all requested models that are not in the cache.  
      ## It is expected that we will only have a single model requested at a time.
      ## We are doing this because multiple models might be requested at once, if say we need to use the results all at one time.
      ## We may need to think more deeply about this in the future, if we do chain models, but for right now
      ##   we are just going to refuse to remove any model that is being requested to be added during the evict model phase.
      # TODO: Sort new_models by size
      for model_to_add in new_models:
        logging.debug(f"Looking for placement for {model_to_add}")
        # Call in case we switch between using each in a loop
        
        model_size = self.getSize(model_to_add)
        
        available_workers = []
        
        ## Notes:
        ##  We may need to separate out selecting the available works from the eviction
        
        ###########
        ## Stage 1: 
        ## look for empty workers ##
        logging.info("Stage 1")
        available_workers = self.model_placements.getEmptyWorkers()
        logging.debug(f"available_workers: {available_workers}")
        
        if len(available_workers) > 0:
          worker_selected =  self.pickWorker(available_workers)
          self.model_placements.addModelToWorker(worker_selected, model_to_add)
          continue
        ##############
        
        
        ###########
        ## Stage 2: 
        ### get workers that have free space ###
        logging.info("Stage 2")
        available_workers = self.findWorkersWithFreeSpace(model_size)
        logging.debug(f"available_workers: {available_workers}")
        
        if len(available_workers) > 0:
          worker_selected =  self.pickWorker(available_workers)
          self.model_placements.addModelToWorker(worker_selected, model_to_add)
          continue
        ############
        
        
        ###########
        ## Stage 3:
        ## find a model to evict and then select a worker to remove it from ##
        logging.info("Stage 3")
        # TODO: GetModelsToEvict should specifically reference worker-model pairs, since we would want this degree of control
        models_to_evict = self.getModelsToEvict(model_to_add)
        available_workers = self.findWorkersWithModels(models_to_evict)
        logging.debug(f"available_workers: {available_workers}")
        
        if len(available_workers) > 0:
          worker_selected =  self.pickWorker(available_workers, models_to_evict)
          for model in models_to_evict:
            self.model_placements.removeModelFromWorker(worker_selected, model)
          self.model_placements.addModelToWorker(worker_selected, model_to_add)
          continue
          
        ###########
        
        
        logging.error(f"Cannot find available worker for model '{model_to_add}'")
        models_not_placed.append(model_to_add)
        
    logging.info(f"acuteUpdate() end")
    return models_not_placed
  
  def getModelsByWorker(self):
    return self.model_placements.getModelsByWorker()
  def getWorkersByModel(self):
    return self.model_placements.getWorkersByModel()
  
  
  #########################
  ## Placement Functions ##
  #########################
  def periodicUpdate(self, message=None, *args, **kwargs):
    logging.info(f"periodicUpdate()")
    
    #self.updateInfo()
    
    model_idle_times = self.getModelIdleTimes()
    if self.do_model_annealing:
      try:
        time_until_next_check = min( [(common.KEEP_ALIVE_IN_SECONDS - t) for t in model_idle_times.values() if t < common.KEEP_ALIVE_IN_SECONDS] )
      except ValueError:
        time_until_next_check = common.PLACEMENT_PERIOD
    else:
      time_until_next_check = common.PLACEMENT_PERIOD
    
    logging.debug(f"time_until_next_check: {time_until_next_check}")
    
    # Star timer to next check
    self.periodicThread = threading.Timer(time_until_next_check, self.periodicUpdate)
    self.periodicThread.start()
    
    
    with self.placement_lock:
      if self.force_remove_periodic:
        # Expire all models
        logging.debug("Removing all models")
        expired_models = [model_name for (model_name, idle_time) in model_idle_times.items()]
        self.expireModels(expired_models, do_sync=False)
      elif self.do_model_cooling:
        logging.debug("Doing model cooling")
        expired_models = [model_name for (model_name, idle_time) in model_idle_times.items() if idle_time >= common.KEEP_ALIVE_IN_SECONDS]
        self.expireModels(expired_models, do_sync=False)
        
      if self.do_model_warming:
        logging.debug("Doing model warming")
        self.fillWithModels(do_sync=False)
        
      if self.do_model_cooling or self.do_model_warming:
        self.syncRedis()

  
  #########################
  #########################
  
  
  ####################
  ## Misc Functions ##
  ####################
  
  def getModelsToEvict(self, model_requested, *args, **kwargs):
    logging.info(f"getModelsToEvict(model_requested={model_requested})")

    models_in_cache = self.model_placements.getModelsInCache()
    logging.debug(f"models_in_cache: {[str(m) for m in models_in_cache]}")
    
    size_of_new_model = self.getSize(model_requested)
    size_of_models_in_cache = sum([self.getSize(m) for m in models_in_cache])
    space_to_free = abs((self.flags.worker_memory - size_of_models_in_cache) - size_of_new_model)
    
    def minmax_scaler(w):
      if np.max(w) == np.min(w):
        return w
      return (w - np.min(w)) / (np.max(w) - np.min(w))
      #return (w - (0.99*np.min(w))) / (np.max(w) - np.min(w))
    def max_scaler(w):
      return w / np.max(w)
    
    
    # Shuffle models to ensure that we don't pick based on location in process memory
    models_to_consider = list(self.rng.permutation(models_in_cache))
    
    if False:
      pass
    elif self.flags.model_eviction_algorithm == "popularity" or (self.flags.model_eviction_algorithm == "random" and self.flags.random_weights == "popularity"):
      model_weights = self.getWeights_popularity(models_to_consider)
      
    elif self.flags.model_eviction_algorithm == "loadtime" or (self.flags.model_eviction_algorithm == "random" and self.flags.random_weights == "loadtime"):
      model_weights = self.getWeights_cost(models_to_consider)
      
    elif self.flags.model_eviction_algorithm == "recent" or (self.flags.model_eviction_algorithm == "random" and self.flags.random_weights == "recent"):
      model_weights = self.getWeights_recent(models_to_consider)
      
    elif self.flags.model_eviction_algorithm == "belady" or (self.flags.model_eviction_algorithm == "random" and self.flags.random_weights == "belady"):
      model_weights = self.getWeights_belady(models_to_consider)
      
    elif self.flags.model_eviction_algorithm == "belady-amortized" or (self.flags.model_eviction_algorithm == "random" and self.flags.random_weights == "belady-amortized"):
      model_weights = self.getWeights_beladyam(models_to_consider, scaler_func=max_scaler, cost_scale=self.flags.cost_scale, boundary_scale=self.flags.boundary_scale, use_size=(not self.flags.do_not_use_size))
      
    elif self.flags.model_eviction_algorithm == "smart" or (self.flags.model_eviction_algorithm == "random" and self.flags.random_weights == "smart"):
      model_weights = self.getWeights_smart(models_to_consider, scaler_func=max_scaler, cost_scale=self.flags.cost_scale, boundary_scale=self.flags.boundary_scale, use_size=(not self.flags.do_not_use_size))
      
    else:
      model_weights = self.getWeights_naive(models_to_consider)
    
    logging.debug(f"raw weights: {np.min(model_weights):0.3f} {np.mean(model_weights):0.3f} {np.max(model_weights):0.3f}")
    logging.debug(f"raw weights: {model_weights}")
    # First, scale to data to always be between 0 and 1 and reverse the order of importance
    model_weights = 1. - (model_weights / np.max(model_weights))
    # Next, remove any NANs that might've been caused by np.max(model_weights) being 0
    np.nan_to_num(model_weights, copy=False, nan=1.0)
    
    models_to_evict = []
    while sum([self.getSize(m) for m in models_to_evict]) < space_to_free:
      # Check to make sure the models aren't all 0, but if they are then give them all an equal chance of being selected
      if np.sum(model_weights) == 0:
        model_weights = model_weights + 1
      # Normalize the weights to unit
      model_weights = model_weights / np.sum(model_weights)
      
      logging.debug("loop")
      logging.debug(f"{sum([self.getSize(m) for m in models_to_evict])} < {space_to_free}")
      logging.debug(f"model_weights: {model_weights}")
      if self.flags.model_eviction_algorithm == "random":
        chosen_index = self.rng.choice(range(0, len(model_weights)), p=model_weights)
      else:
        chosen_index = np.argmax(model_weights)
      logging.debug(f"chosen_index: {chosen_index}")
      logging.debug(f"len(models_to_consider: {len(models_to_consider)}")
      models_to_evict.append(models_to_consider[chosen_index])
      del models_to_consider[chosen_index]
      model_weights = np.delete(model_weights, chosen_index)
    
    models_to_evict = collections.deque(sorted(models_to_evict, key=(lambda m: self.getSize(m)), reverse=True))
    models_to_evict_prime = []
    while sum([self.getSize(m) for m in models_to_evict_prime]) < space_to_free:
      models_to_evict_prime.append(models_to_evict.popleft())
    
    logging.debug(f"models_to_evict_prime: {[str(m) for m in models_to_evict_prime]}")
    
    return models_to_evict_prime
  
  
  # A note on model weights
  ## Our goal is to have higher weights be better.
  ## For instance, a high cost model will have more value, or a more popular model will have more value.
  ## We should stick to these pattern whenever possible.
  ## For compound weights (e.g. smart or beladyam) we should combine other approaches, and return the output in the same order
  def getModelWeights(self, list_of_models, value_func=(lambda m: 0.), normalize=False, *args, **kwargs):
    w = np.array([value_func(m) for m in list_of_models])
    return w
  
  # Direct weight functions
  def getWeights_popularity(self, list_of_models, *args, **kwargs):
    return self.getModelWeights(list_of_models, value_func=self.getPopularity, *args, **kwargs)
  def getWeights_cost(self, list_of_models, *args, **kwargs):
    return self.getModelWeights(list_of_models, value_func=self.getCost, *args, **kwargs)
  def getWeights_naive(self, list_of_models, *args, **kwargs):
    return self.getModelWeights(list_of_models, *args, **kwargs)
  def getWeights_belady(self, list_of_models, *args, **kwargs):
    # Note: this is 1 / w because the further out the higher the belady boundary and the lower the utility
    return self.getModelWeights(list_of_models, value_func=(lambda m: (1. / self.getBeladyBoundary(m, self.latest_request_id))), *args, **kwargs)
  def getWeights_recent(self, list_of_models, *args, **kwargs):
    return self.getModelWeights(list_of_models, value_func=self.getLastUsed, *args, **kwargs)
  def getWeights_size(self, list_of_models, *args, **kwargs):
    return self.getModelWeights(list_of_models, value_func=self.getSize, *args, **kwargs)
  
  # Compound weight functions
  def getWeights_beladyam(self, list_of_models, cost_scale=1.0, boundary_scale=1.0, use_size=True, *args, **kwargs):
    if "scaler_func" in kwargs:
      scaler_func = kwargs["scaler_func"]
    else:
      scaler_func = (lambda w: w)
      
    w_belady = np.power(scaler_func(self.getWeights_belady(list_of_models, *args, **kwargs)), boundary_scale)
    w_cost = np.power(scaler_func(self.getWeights_cost(list_of_models, *args, **kwargs)), cost_scale)
    w_size = self.getWeights_size(list_of_models, *args, **kwargs)
    logging.debug(f"w_belady: {np.min(w_belady):0.3f} {np.mean(w_belady):0.3f} {np.max(w_belady):0.3f}")
    logging.debug(f"w_cost: {np.min(w_cost):0.3f} {np.mean(w_cost):0.3f} {np.max(w_cost):0.3f}")
    logging.debug(f"w_size: {np.min(w_size):0.3f} {np.mean(w_size):0.3f} {np.max(w_size):0.3f}")
    if use_size:
      return (w_belady * w_cost) / w_size
    else:
      return (w_belady * w_cost)
    
  def getWeights_smart(self, list_of_models, cost_scale=1.0, boundary_scale=1.0, use_size=True, *args, **kwargs):
    w_popularity = self.getWeights_popularity(list_of_models, *args, **kwargs)
    if "scaler_func" in kwargs:
      scaler_func = kwargs["scaler_func"]
    else:
      scaler_func = (lambda w: w)
      
    w_cost = np.power(scaler_func(self.getWeights_cost(list_of_models, *args, **kwargs)), cost_scale)
    w_size = self.getWeights_size(list_of_models, *args, **kwargs)
    
    # (1. / w_popularity) -> estimated normalized inter-arrival time
      # This gives a number between 0 and 1 that is related to the interarrival time
    # ((1+self.latest_request_id) / w_popularity) -> estimated inter-arrival time
      # This is an estimate of the belady boundary
    # (1. / ((1+self.latest_request_id) / w_popularity)) -> estimated boundary cost
      # inverse of the previous to match what `getBeladyWeights(...)` returns
    # (w_popularity / (1+self.latest_request_id)
      # simplified version of the previous
      # Note that this means that the higher percentage of requests that exist for this model, the higher its value
    w_estimated_belady = np.power(scaler_func(w_popularity / (1+self.latest_request_id)), boundary_scale)
    if use_size:
      return (w_estimated_belady * w_cost) / w_size
    else:
      return (w_estimated_belady * w_cost)
  
  
  
  @classmethod
  def getBeladyBoundary(cls, model, request_id):
    while len(cls.request_indices[model]) > 0:
      if request_id >= cls.request_indices[model][0]:
        # Then we've gone past this request and we should pare down the list
        cls.request_indices[model].popleft()
      else:
        # Then the next request is in the list and we can calculate how far until we get to it
        return cls.request_indices[model][0] - request_id
    # There were no more usages in the list, so we just remove the list and return `inf`
    del cls.request_indices[model]
    return float('inf')
  
  @classmethod
  def get_num_occurances(cls, model, curr_request_id, limit_request_id):
    while len(cls.request_indices[model]) > 0:
      if curr_request_id >= cls.request_indices[model][0]:
        # Pop requests off
        cls.request_indices[model].popleft()
      else:
        break
    num_occurances = 0
    for i in range(len(cls.request_indices[model])):
      if cls.request_indices[model][i] < limit_request_id:
        num_occurances += 1
    return num_occurances
    
  
  def getModelAttribute(self, model_name, attribute_name, reverse=False):
    try:
      attr_val = self.model_info[model_name][attribute_name]
    except KeyError:
      raise exceptions.ModelInformationMissingException
    
    if not reverse:
      return attr_val
    else:
      return -1 * attr_val
  def getCost(self, m, *args, **kwargs):
    if self.flags.cost_function == "cost-direct":
      return self.getLoadTime(m)
    elif self.flags.cost_function == "cost-increase":
      return self.getLoadTime(m) / self.getExecTime(m)
    #return (self.getLoadTime(m) - self.getExecTime(m)) / (1+self.getExecTime(m))
    #return (self.getModelAttribute(m, "load_latency", *args, **kwargs) / self.getModelAttribute(m, "exec_latency", *args, **kwargs))
  def getLoadTime(self, m, *args, **kwargs):
    return self.getModelAttribute(m, "load_latency", *args, **kwargs)
  def getExecTime(self, m, *args, **kwargs):
    return self.getModelAttribute(m, "exec_latency", *args, **kwargs)
  def getPopularity(self, m, *args, **kwargs):
    return self.getModelAttribute(m, "requests_submitted", *args, **kwargs)
  def getSize(self, m, *args, **kwargs):
    return self.getModelAttribute(m, "loaded_size", *args, **kwargs)
  def getLastUsed(self, m, *args, **kwargs):
    return self.getModelAttribute(m, "last_used", *args, **kwargs)
  
  
  
  ##################################
  ## Worker Information Functions ##
  ##################################
  def findWorkersWithFreeSpace(self, space_needed):
    return [worker for worker in self.model_placements.getWorkers() if self.hasEnoughFreeSpace(worker, space_needed)]
  
  def hasEnoughFreeSpace(self, worker_url, space_needed):
    # TODO: Make this actually calculate, instead of just saying it does.
    if self.flags.worker_memory is not None:
      return space_needed + sum([m.getSize() for m in self.model_placements.getModelsFromWorker(worker_url)]) <= self.flags.worker_memory
    return len(self.model_placements.getModelsFromWorker(worker_url)) < self.flags.max_concurrent_models
  
  def findWorkersWithModels(self, models_to_evict):
    # This statement finds all the workers that host both the selected models.
    # This statement translates to finding the intersection of all the placements of each of the models that are being evicted
    if len(models_to_evict) == 0:
      return []
    
    return list(set.intersection( *([set(self.model_placements.getWorkersFromModel(model_name)) for model_name in models_to_evict]) ))
  
  def pickWorker(self, potential_workers, models_to_evict=[], *args, **kwargs):
    """
    Picks a worker from a list to evict
    
    If given models_to_evict it can take this into consideration
    """
    return self.rng.choice(potential_workers)
  ##################################
  
  
  #####################
  ## Model Functions ##
  #####################
  def getIdleTime(self, model_name):
    if self.model_info[model_name]['last_used'] == 0:
      # TODO: is this accurate?  If a model is never used this might be inaccurate
      # Instead would we need to tag when we add models to the placement?
      return self.model_info[model_name]['last_used']
    else:
      return (time.time() - self.model_info[model_name]['last_used'])
      
  def getModelIdleTimes(self, *args, **kwargs):
    live_models = self.model_placements.getModelsInCache()
    model_idle_times = { m : self.getIdleTime(m) for m in live_models }
    return model_idle_times
    
  def updateKeepAliveModels(self, *args, **kwargs):
    logging.info(f"updateKeepAliveModels()")
    
    model_idle_times = self.getModelIdleTimes()
    self.models_kept_alive = [ model_name for (model_name, time_idle) in model_idle_times.items() if time_idle < common.KEEP_ALIVE_IN_SECONDS]
    
    return self.models_kept_alive
  
  def expireModels(self, newly_expired_models, do_sync=True, *args, **kwargs):
    logging.info(f"expireModels({newly_expired_models})")
    # TODO: This should be done on a per-worker basis
    for model_name in newly_expired_models:
      self.model_placements[model_name] = set([])
  
  def fillWithModels(self, do_sync=True, *args, **kwargs):
    logging.info(f"fillWithModels()")
    # TODO: Make this do more than just add models that are not currently placed
    potential_models_to_add = list(set(self.model_names) - set(self.model_placements.getModelsInCache()))
    # TODO: reimplement this to actually choose models to add
    #models_to_add = self._modelEvictionAlgorithm(potential_models_to_add, self.model_info,)[::-1]
    models_to_add = []
    
    for model_to_add in models_to_add:
      space_needed = 1 #self.model_info[model_to_add]["model_size"]
      available_workers = self.findWorkersWithFreeSpace(space_needed)
      if len(available_workers) == 0:
        break
      self.model_placements.addModelToWorker(available_workers[0], model_to_add)
    
  def getModelsSetsThatMakeEnoughSpace(self, models_that_can_be_removed, space_to_free=1.):
    """
    Function that will eventually return all sets of models that could be removed to replace the current model.
    
    That is, given m1, m2, and m3 with sizes 0.5, 0.5, and 1 GB respectively, and a need to free up space for a model
       that is 1GB in size then [ [m1, m2], [m3] ] would be returned.
    Therefore we want it to be the set of models that would need to be removed, without extras being added beyond the base set.
    (This might be difficult eventually, but right now I get to be lazy).
    """
    return [[m] for m in models_that_can_be_removed]
    
  #####################
  #####################
    
    

def getParser(add_help=True, include_parents=True):
  parser = argparse.ArgumentParser(add_help=add_help,
    parents=([common.getParser(add_help=False)] if include_parents else [])  
  )
  
  #############
  ## General ##
  
  parser.add_argument('--time_between_updates', default=10,
            help="Time between updates in seconds.")
  
  parser.add_argument(
      "--cost_function",
      choices=[
        "cost-direct",
        "cost-increase"
      ],
      default="cost-direct",
      help="Sets the cost function to use between the two approaches explained in 2021-02-17.md"
    )
  #############
  
  #####################
  ## Eviction Policy ##
  parser.add_argument('--disable_reactive', action='store_false', dest="do_reactive",
              help=f"If set then models will expire, be removed and be replaced via model-warming.")
  
  parser.add_argument('--oracle', action='store_true', 
              help="If set then oracle variation of eviction algorithm will be used")
  
  parser.add_argument('--model_eviction_algorithm', 
              choices=['random', 
                      'belady',
                      'belady-amortized',
                      'popularity', 
                      'loadtime', 
                      'smart',
                      'recent',
                    ],
              default='random',
              help="Algorithm to use to select model to evict"
              )
  
  
  parser.add_argument("--scale_func",
                      choices=[
                        "rank",
                        "minmax",
                      ],
                      default="minmax",
                      help="Function to scale values to [0,1]")
  parser.add_argument("--weight_func",
                      choices=[
                        "identity",
                        "favor_large",
                        "favor_small",
                      ],
                      default="identity",
                      help="Function to weight values in [0,1].  Identity is nothing, increase uses 'softmax', decrease uses 'sigmoid'")
  
  parser.add_argument("--random_weights", 
    choices=[
      "naive", 
      "popularity", 
      "loadtime", 
      "belady", 
      "belady-amortized", 
      "smart"]
  )
  
  #####################
  
  ###############
  ## Proactive ##
  parser.add_argument('--do_proactive', action='store_true', dest="do_proactive",
              help=f"If set then models will expire, be removed and be replaced via model-warming.")
  
  parser.add_argument('--keep_alive', action='store_true',
              help=f"If set then keep alive time of {common.KEEP_ALIVE_IN_SECONDS}s will be used")
  parser.add_argument('--model_cooling', action='store_true',
              help=f"If set then model-cooling will remove models that are alive after {common.KEEP_ALIVE_IN_SECONDS}s")
  
  parser.add_argument('--model_warming', action='store_true',
              help=f"If set then model-warming will add models when it can.")
  
  parser.add_argument('--model_annealing', action='store_true',
              help=f"If set then models will expire, be removed and be replaced via model-warming.")
  parser.add_argument('--force_remove_periodic', action='store_true',
              help=f"If set then models will all be removed on periodic updates.")
  
  ###############
  
  parser.add_argument('--cost_scale', default=1.0, type=float, help="Modifier to be applied to the cost of a model.")
  parser.add_argument('--boundary_scale', default=1.0, type=float, help="Modifier to be applied to the belady boundary of a model.")
  parser.add_argument('--do_not_use_size', action="store_true", help="Set to not use size in smart and beladyam")
  
  return parser


def main():

  import abstractions
  
  common.getLogger(f"{os.path.basename(__file__).replace('.py', '')}")
  
  signal.signal(signal.SIGTERM, (lambda *_: sys.exit(0)))
  
  flags = getParser().parse_args()
  
  placement_controller = PlacementController(flags,
                        flags.time_between_updates,
                        keep_alive=flags.keep_alive,
                        model_cooling=flags.model_cooling,
                        model_warming=flags.model_warming,
                        model_annealing=flags.model_annealing,
                        do_reactive=flags.do_reactive,
                        do_proactive=flags.do_proactive,
                        force_remove_periodic=flags.force_remove_periodic,
                        model_eviction_algorithm=flags.model_eviction_algorithm,
                      )
  
  redis_controller = RedisInterface_PlacementController(placement_controller, flags.redis_server, flags.redis_port)
  

if __name__ == '__main__':
  main()