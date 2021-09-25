#!env python

import logging as logger
import threading

from threading import Lock
from queue import Queue
from threading import Event


import common
from worker_interface import WorkerClient
import exceptions

class WorkerAbstraction():
  def __init__(self, worker_url):
    self.worker_url = worker_url
    self.client = WorkerClient(worker_url)
    self.lock = Lock()
    self.models_by_name = {}
    self.queue = Queue()
    
    self.queue_thread = threading.Thread(target=self.processQueue)
    self.queue_thread.start()
    
    self.shutting_down = False
  
  def startProcessingQueue(self):
    if not self.queue_thread.is_alive():
      self.queue_thread = threading.Thread(target=self.processQueue)
      self.queue_thread.start()
    
  def __hash__(self):
    return hash(self.worker_url)
    
  def loadModel(self, model):
    logger.info(f"loadModel({self.worker_url}, {model.model_name})")
    with self.lock:
      self.queue.put( (self._loadModel, [model], {}) )
      self.startProcessingQueue()
      self.models_by_name[model.model_name] = model
      model.addPlacement(self)
    return True
    
  def _loadModel(self, model):
    try:
      self.client.loadModel(model.model_name)
    except exceptions.ModelNotInRepoException as e:
      # We should remove the model from the possible models on the worker
      logger.error("Model not in repository!")
      logger.error(e)
    except exceptions.NotEnoughFreeSpace as e:
      # We need to free up space
      logger.error("Not enough free space on model!")
      logger.error(e)
  
  def unloadModel(self, model):
    logger.info(f"unloadModel({self.worker_url}, {model.model_name})")
    with self.lock:
      self.queue.put( (self._unloadModel, [model], {}) )
      self.startProcessingQueue()
      del self.models_by_name[model.model_name]
      model.removePlacement(self)
    return True
      
  def _unloadModel(self, model):
    try:
      self.client.unloadModel(model.model_name)
    except exceptions.ModelNotInRepoException as e:
      logger.error("Model was not in worker repo!")
      logger.error(e)
  
  def infer(self, inference_request):
    logger.info(f"infer({self.worker_url}, {inference_request.model_name})")
    with self.lock:
      logger.info(f"self.models_by_name: {self.models_by_name}")
      if inference_request.model_name not in self.models_by_name:
        logger.warning("Cannot add inference request to queue, model not loaded.")
        return False
      self.queue.put( (self._infer, [inference_request], {}) )
      self.startProcessingQueue()
      return True
  
  @common.timing
  def _infer(self, inference_request):
    logger.info(f"_infer({self.worker_url}, {inference_request.model_name}")
    try:
      inference_request.mergeRequests(self.client.infer(inference_request))
      inference_request.complete.set()
    except exceptions.WorkerException as e:
      logger.error("Inference failed")
      logger.error(e)
      raise exceptions.InferenceFailedException("Inference failed on worker")
    logger.info(f"_infer({self.worker_url}, {inference_request.model_name} complete")
    
  
  def processQueue(self):
    while not self.queue.empty():
      (func_call, args, kwargs) = self.queue.get()
      logger.info(f"Got {func_call.__name__} from queue")
      func_call(*args, **kwargs)
      logger.info(f"Executed {func_call.__name__} from queue")
    
        
      

class ModelAbstraction():
  def __init__(self, model_name):
    self.model_name = model_name
    self.is_available = Event()
    self.unassigned_requests = Queue()
    self.placements = set()
  def __hash__(self):
    return hash(self.model_name)
  def __repr__(self):
    return f"<{self.__class__.__name__}, {self.model_name}, {self.is_available.isSet()}, {self.placements}>"
  def addPlacement(self, worker):
    self.placements.add(worker)
    self.is_available.set()
  def removePlacement(self, worker):
    self.placements.remove(worker)
    if len(self.placements) == 0:
      self.is_available.clear()
  
  #def addRequest(self, inference_request):
  #    self.unassigned_requests.put(inference_request)


class RedisAbstraction():
  def __init__(self, redis_server="redis-server", redis_port=6379, simulation=False, *args, **kwargs):
    self.simulation = simulation
    
    if not self.simulation:
      self.db = redis.Redis(host=redis_server, port=redis_port)
      try:
        self.db.ping()
      except redis.exceptions.ConnectionError as e:
        print(f"Error connecting to Redis server @ {redis_server}.  Is it started?")
        print(e)
  
  def acuteUpdate(self, message, *args, **kwargs):
    pass
  def getModelInfo(self):
    pass
  def syncRedis(self):
    pass
  def updateWorkerList(self, message=None):
    pass
  def updateModelList(self, message=None):
    pass
  
  def updateModelPlacement(self, message):
    pass
  
  def requestModelPlacement(self, inference_request):
    pass
