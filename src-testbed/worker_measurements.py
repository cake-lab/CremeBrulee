#!env python

import argparse
import docker
import time
import socket
import platform
import enum

import threading
from threading import Event
import queue
#from queue import Queue

import functools

import redis
from redis import Redis

import signal

# My imports
import exceptions
from worker_interface import WorkerServer
import models

import logging
import os
import common


NETWORK_NAME = "worker_network"

class DataBase(object):
  def __init__(self, redis_server="redis-server", redis_port=6379, testing=False):
    self.testing = testing
    if not self.testing:
      self.db = Redis(host=redis_server, port=redis_port)
      logging.debug("Trying to connect to redis server")
      try:
        self.db.ping()
      except redis.exceptions.ConnectionError as e:
        logging.critical(f"Error connecting to Redis server @ {redis_server}.  Is it started?")
        logging.critical(e)
        exit(8)
      logging.debug("Connected to redis")
    else:
      self.db = None
      
  def smembers(self, s):
    if self.testing:
      return ["resnet50_netdef_a", "resnet50_netdef_b", "resnet50_netdef_c"]
    else:
      return self.db.smembers(s)
  
  def get(self, s):
    if self.testing:
      return 0.1 # sso
    else:
      return self.db.get(s)
  def set(self, *args, **kwargs):
    if self.testing:
      pass
    else:
      return self.db.set(*args, **kwargs)
  
  def pipeline(self):
    return self.db.pipeline()
  

@functools.total_ordering
class WorkerEvent(Event):
  
  def __init__(self, model_name, is_model_change, func, args=[], kwargs={}):
    super().__init__()
    self.model_name = model_name
    self.is_model_change = is_model_change
    
    self.enqueue_time = time.time()
    self.execution_start_time = None
    self.execution_end_time = None
    
    self.func = func
    self.args = args if type(args) is list else [args]
    self.kwargs = kwargs
    
    self.response = None
  
  def executeInThread(self):
    threading.Thread(target=self.execute).run()
    
  def execute(self):
    logging.debug("Starting Execution")
    self.execution_start_time = time.time()
    self.response = self.func(*self.args, **self.kwargs)
    self.execution_end_time = time.time()
    logging.debug(f"Execution Complete: {self.response}")
    self.set()
    return self.response
    
  def __repr__(self):
    return f"WorkerEvent<{self.func}, {self.enqueue_time}>"
    
  # For ordering in priority queue
  def __eq__(self, other):
    return self.enqueue_time == other.enqueue_time
  def __lt__(self, other):
    return self.enqueue_time < other.enqueue_time
  

  
class Worker(object):
  
  def __init__(self, flags, redis_server, redis_port, model_repo_path, host_repo_path, container_id, worker_name=None, dummy_load=True, *args, **kwargs):
    
    self.flags = flags
    
    # Set Up Redis database connection
    self.db = DataBase(redis_server, redis_port, flags.testing)
    
    # Set up docker connection
    ## Note, we might push this to be model controlled, but it is pretty integrated.  Hmm
    logging.debug("Connecting to docker")
    try:
      self.docker_client = docker.from_env()
    except FileNotFoundError as e:
      logging.critical("Cannot find docker file.  Is it running?")
      logging.critical(e)
      exit(8)
    logging.debug("Connected to docker")
    
    # For labeling network and workers
    self.container_id = container_id
    self.container = self.docker_client.containers.get(self.container_id)
    self.container_prefix = self.container_id[:5]
    self.network_name = f"{self.container_prefix}-{NETWORK_NAME}"
    self.network = self.docker_client.networks.create(name=f"{self.network_name}", internal=False, check_duplicate=True)
    self.network.connect(self.container)
    
    # For reporting metrics
    self.worker_name = worker_name
    
    # Model Path information
    self.host_repo_path = host_repo_path # To pass to individual models
    #self.model_repo_path = model_repo_path # To know what models are available
    self.models_in_repo = os.listdir(model_repo_path) # Available models
    
    self.dummy_load = dummy_load
    
    self.max_concurrent_execs = flags.max_concurrent_execs
    self.num_concurrent_execs = 0
    self.current_execution_list = []
    
    # Set up models
    if self.dummy_load:
      logging.debug("Using Dummy models")
      Model = models.DummyModel
      self.models_in_repo = [w.decode() for w in self.db.smembers(f"{common.MODEL_NAME_KEY}")]
      logging.info(f"self.models_in_repo: {self.models_in_repo}")
      
    else:
      logging.debug("Using Real models")
      Model = models.TFSServerModel
      models.TFSServerModel.initServer(self.docker_client, self.host_repo_path, flags.worker_memory, self.network_name, arm=flags.use_arm)
    common.getModelInfo(self.db)
    logging.info(f"Running with {Model}")
    self.models_by_name = {
      model_name : Model(model_name, os.path.join(self.host_repo_path, model_name), self.docker_client, self.network_name, self.container_prefix)
      for model_name in self.models_in_repo
    }
    
    self.metrics = common.Metrics(self.db, f"{common.WORKER_STAT_PREFIX}{self.worker_name}", ["requests_submitted", "requests_completed", "open_requests", "model_loads", "model_unloads"])
    
    self.event_queue = queue.PriorityQueue()
    self.processing_thread = threading.Thread(target=self.processEventQueue)
    self.processing_thread.start()
    
    self._unused_queue = queue.PriorityQueue()
    
  def processEventQueue(self):
    while True:
      # (sso) TODO: uncomment the below line
      #self.getStatistics()
      event = self.event_queue.get()
      
      # Check to see if event changes the models in memory
      if not event.is_model_change:
        # Check to see if request could potentially be serviced
        if not self.models_by_name[event.model_name].isAvailable():
          self._unused_queue.put(event)
          continue
      time_start = time.time()
      event.execute()
      #event.set()
      # (sso) TODO: maybe only update after model loads/unloads?
  
  def processEventQueue_new(self):
    while True:
      event = self.event_queue.get()
      
      # Check if event is serviceable
      if (not event.is_model_change) and (not self.models_by_name[event.model_name].isAvailable()):
        # If not, then put in the unused queue to be mixed back in after model load
        self._unused_queue.put(event)
        continue
      
      # We know that the event can be executed
      logging.debug(f"Next Event: {event.func.__name__}")
      event.executeInThread()
      self.current_execution_list.append(event)
      
      while len(self.current_execution_list) == self.max_concurrent_execs:
        # Loop until some of the executions decrease in number
        ## That is, just stay here until something completes
        self.current_execution_list = list(filter((lambda r: not r.is_set()), self.current_execution_list))
      
      # Finally updated redis stats
      self.updateRedisModelStats()
        
        
  
  def stopWorker(self):
    logging.debug("Stopping Worker")
    for model in self.models_by_name.values():
      model.shutdown(shutdown=True)
    self.network.disconnect(self.container)
    self.network.remove()
    sys.exit(0)
  
  def getStatistics(self):
    stats_by_model = {
      model_name : model.getStatistics()
      for model_name, model in self.models_by_name.items()
    }
    return stats_by_model
      
  
  ########################
  ## External Functions ##
  ########################
  @common.timing
  def loadModel(self, model_name, *args, **kwargs):
    logging.info(f"loadModel() requested for {model_name}")
    if model_name not in self.models_by_name:
      raise exceptions.ModelNotInRepoException("Model not available in model repository")
    if not self._isThereFreeSpaceToLoad(model_name):
      # TODO: with queued loading/unloading, this becomes more complicated
      raise exceptions.NotEnoughFreeSpace("Not enough free space to load model")
    
    load_event = WorkerEvent(model_name, is_model_change=True, func=self._loadModel, args=[model_name])
    self.event_queue.put(load_event)
    
    self.recordModelLoad(model_name)
    
    
  @common.timing
  def unloadModel(self, model_name, *args, **kwargs):
    logging.info(f"unloadModel() requested for {model_name}")
    if model_name not in self.models_by_name:
      raise exceptions.ModelNotInRepoException("Model not available in model repository")
      
    unload_event = WorkerEvent(model_name, is_model_change=True, func=self._unloadModel, args=[model_name])
    self.event_queue.put(unload_event)
    
    self.recordModelUnload(model_name)
  
  @common.timing
  def requestInference(self, inference_request, *args, **kwargs):
    logging.info(f"requestInference() for model '{inference_request.model_name}'")
    logging.info(f"request '{inference_request}'")
    self.recordRequestEntry(inference_request.model_name)
    
    inference_request.markAssignment()
    logging.debug(f"Assigning: {inference_request}")
    infer_event = WorkerEvent(inference_request.model_name, 
                  is_model_change=False, 
                  func=self.models_by_name[inference_request.model_name].runInference, 
                  args=[inference_request])
    self.event_queue.put(infer_event)
    
    logging.debug(f"Waiting: {inference_request}")
    infer_event.wait(common.TIMEOUT_IN_SECONDS)
    self.recordRequestExit(inference_request.model_name)
    
    if infer_event.response is None:
      raise exceptions.ModelInferenceException(f"Inference model ({model_name}) did not respond")
    
    return infer_event.response
  ########################
  ########################
  
  
  ####################
  ## Core Functions ##
  ####################
  @common.timing
  def _loadModel(self, model_name):
    model_stats = self.getModelInformation(model_name)
    self.models_by_name[model_name].startModel(model_stats)
    while not self._unused_queue.empty():
      self.event_queue.put(self._unused_queue.get())
    
  @common.timing
  def _unloadModel(self, model_name):
    self.models_by_name[model_name].stopModel()
  
  
  ####################
  ####################
  
  
  #############################
  ## Informational Functions ##
  #############################
  def _isThereFreeSpaceToLoad(self, model_name, *args, **kwargs):
    return True
    if self.dummy_load:
      return True
    else:
      running_models = [m for m in self.models_by_name.values() if m.isAvailable()]
      logging.debug(running_models)
      logging.info(f"Free space: {self.flags.worker_memory - sum([m.getModelSize() for m in running_models])}")
      logging.info(f"Space needed: {self.models_by_name[model_name].getModelSize()}")
      return (sum([m.getModelSize() for m in running_models]) + self.models_by_name[model_name].getModelSize() )<= self.flags.worker_memory
  
  #############################
  #############################
  
  
  ########################
  ## Redis interactions ##
  ########################
  def getModelInformation(self, model_name, *args, **kwargs):
    expected_exec_latency = self.db.get(f"{common.MODEL_STAT_PREFIX}{model_name}{common.DB_FIELD_SEPARATOR}avg_exec_latency")
    expected_load_latency = self.db.get(f"{common.MODEL_STAT_PREFIX}{model_name}{common.DB_FIELD_SEPARATOR}avg_load_latency")
    expected_unload_latency = self.db.get(f"{common.MODEL_STAT_PREFIX}{model_name}{common.DB_FIELD_SEPARATOR}avg_unload_latency")
    loaded_size = self.db.get(f"{common.MODEL_STAT_PREFIX}{model_name}{common.DB_FIELD_SEPARATOR}loaded_size")
    
    model_stats = {
        "expected_exec_latency"     : common.DUMMY_EXEC_LATENCY if expected_exec_latency is None else float(expected_exec_latency.decode()),
        "expected_load_latency"     : common.DUMMY_LOAD_LATENCY if expected_load_latency is None else float(expected_load_latency.decode()),
        "expected_unload_latency"   : common.DUMMY_UNLOAD_LATENCY if expected_unload_latency is None else float(expected_unload_latency.decode()),
        "loaded_size"               : 0 if loaded_size is None else float(loaded_size.decode()),
      }
    return model_stats
  
  def updateRedisModelStats(self):
    if not self.flags.update_redis:
      return
    logging.info(f"updateRedisModelStats()")
    pipe = self.db.pipeline()
    for model_name, model in self.models_by_name.items():
      for stat_name, stat in model.getStatistics().items():
        #logging.debug(f"{model_name}:{stat_name} : {stat}")
        pipe.set( f"{common.MODEL_STAT_PREFIX}{model_name}{common.DB_FIELD_SEPARATOR}{stat_name}", stat )
    results = pipe.execute()
    return results
    
      
  
  ########################
  ########################
  
  
  #################################
  ## Metrics recording Functions ##
  def recordRequestEntry(self, model_requested):
    self.metrics.incrementMetricBy("requests_submitted", model_requested)
    self.metrics.incrementMetricBy("open_requests", model_requested, +1)
    
  def recordRequestExit(self, model_requested):
    logging.info(f"recordRequestExit({model_requested})")
    self.metrics.incrementMetricBy("requests_completed", model_requested)
    self.metrics.incrementMetricBy("open_requests", model_requested, -1)
    
  def recordModelLoad(self, model_requested):
    self.metrics.incrementMetricBy("model_loads", model_requested)
    
  def recordModelUnload(self, model_requested):
    self.metrics.incrementMetricBy("model_unloads", model_requested)
  #################################
  
    
    

def getParser(add_help=True, include_parents=True):
  parser = argparse.ArgumentParser(add_help=add_help,
    parents=([common.getParser(add_help=False)] if include_parents else [])  
  )
  
  parser.add_argument('--model_repo_path', default="/tmp/models",
              help='Path to model repo')
  parser.add_argument('--real_model_repo_path', default="/Users/ssogden/research/2020-project-EdgeController/triton-inference-server/docs/examples/model_repository.limited",
              help='Path to model repo')
  parser.add_argument('--running_in_docker', action="store_true",
              help="Setting to help system determine if running within docker.")
  
  parser.add_argument('--worker_name',
              help="Name of worker")
  
  parser.add_argument('--use_arm', action="store_true")
  parser.add_argument('--testing', action="store_true")
  
  parser.add_argument('--update_redis', action="store_true")
  
  parser.add_argument('--max_concurrent_execs', default=1, type=int, help="Number of concurrent executions that can be started")
  
  return parser



def main():
  
  
  flags = getParser().parse_args()
  
  flags.use_arm = (platform.processor() == 'arm64')
  
  common.getLogger() #f"{os.path.basename(__file__).replace('.py', '')}")
  
  # we can assume we are running in Docker basically
  container_id = socket.gethostname()
  logging.debug(f"container_id: {container_id}")
  
  
  worker = Worker(flags, 
                  flags.redis_server, 
                  flags.redis_port, 
                  flags.model_repo_path, 
                  host_repo_path=(flags.real_model_repo_path if flags.running_in_docker else flags.model_repo_path), 
                  container_id=container_id, 
                  worker_name=flags.worker_name, 
                  dummy_load=flags.dummy_load)
  signal.signal(signal.SIGTERM, worker.stopWorker)
  logging.info("Worker set up.  Beginning serving gRPC")
  worker_server = WorkerServer.serve(worker)
  logging.info("Finished serving, wrapping up")
  worker.stopWorker()
  

  

if __name__ == '__main__':
  main()