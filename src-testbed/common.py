#!env python


from functools import wraps
import time
import collections
import threading
import enum
import json

from threading import Event
from threading import Lock
from itertools import count

import numpy as np

import logging

import queue
from threading import RLock as Lock

import argparse

import exceptions



#############
## Globals ##
#############
TIMEOUT_IN_SECONDS = None #600
POLL_SLEEP_MS = 100
REPORT_INTERVAL = 1
PLACEMENT_PERIOD = 2

DUMMY_EXEC_LATENCY = 0.000
DUMMY_LOAD_LATENCY = 0.000
DUMMY_UNLOAD_LATENCY = 0

KEEP_ALIVE_IN_SECONDS = 2
PLACEMENT_POLL_INTERVAL = 0.1

MODEL_INFO_FRESHNESS = 5.
#############
#############


##############
## Prefixes ##
##############
KEYSPACE_PREFIX = "__keyspace@0__:"
DB_FIELD_SEPARATOR = ":"

UPDATING_FLAG_NAME = "updating_metrics"
PLACEMENT_REQUEST_KEY = "requested-placements"

WORKER_URL_KEY = "workers"
MODEL_NAME_KEY = "models"

MODEL_PLACEMENT_PREFIX = f"model-placement{DB_FIELD_SEPARATOR}"

WORKER_PREFIX = f"worker{DB_FIELD_SEPARATOR}"
MODEL_PREFIX = f"model{DB_FIELD_SEPARATOR}"

STAT_PREFIX = f"statistic{DB_FIELD_SEPARATOR}"
WORKER_STAT_PREFIX = f"{STAT_PREFIX}worker{DB_FIELD_SEPARATOR}"
MODEL_STAT_PREFIX = f"{STAT_PREFIX}model{DB_FIELD_SEPARATOR}"
##############
##############


def getLogger(identifier=None, hide_debug=False):
  
  #return logging.getLogger() 
  
  logFormatter = logging.Formatter(f"%(threadName)s %(asctime)s [%(levelname)-5.5s]  %(message)s")

  consoleHandler = logging.StreamHandler()
  consoleHandler.setFormatter(logFormatter)
  
  logger = logging.getLogger()
  logger.addHandler(consoleHandler)
  
  if hide_debug:
    logger.setLevel(logging.INFO)
  else:
    logger.setLevel(logging.DEBUG)
  logger.setLevel(logging.WARNING)
  
  return logger


def getParser(add_help=True, *args, **kwargs):
  parser = argparse.ArgumentParser(add_help=add_help)
  parser.add_argument('--redis_server', default="redis-server",
            help='Redis Server name')
  parser.add_argument('--redis_port', default=6379, type=int,
            help='Redis Server port')
  
  parser.add_argument('--workload_file', default="workload/workload.txt")
  parser.add_argument('--model_description_file', default="workload/models.json")
  
  parser.add_argument('--rng_seed', default=None, type=int,
                      help="RNG Seed.  Default is no seed.")
  
  ## Options that may affect multiple containers
  parser.add_argument('--max_concurrent_models', default=2, type=int,
                      help="Maximum number of models that can be loaded at once")
  
  parser.add_argument('--num_workers_to_add', default=1, type=int,
                      help="Number of workers to add to the system")
  
  
  parser.add_argument('--worker_memory', default=100, type=float,
                      help="Amount of memory on each worker. (In reality would be per-worked, but this is for the simulation.)")
  parser.add_argument('--dummy_load', action="store_true",
              help="Set to true to use a dummy load, with no loading or unloading of models or inferences.")
  
  return parser


def fixWorkerURL(worker_url):
  if not ":" in worker_url:
    worker_url += ":50051"
  return worker_url
def stripWorkerURL(worker_url):
  if ":" in worker_url:
    worker_url = worker_url[:worker_url.index(':')]
  return worker_url


def timing(f):
  @wraps(f)
  def wrap(*args, **kw):
    ts = time.time()
    result = f(*args, **kw)
    te = time.time()
    logging.info(f"TIMING: {f.__name__}({[str(i)[:100] for i in args], kw}): {te-ts}s")
    return result
  return wrap

def gather_info(f):
  @wraps(f)
  def wrap(*args, **kw):
    ts = time.time()
    result = f(*args, **kw)
    te = time.time()
    logging.info(f"STATISTICS: {f.__name__}({[str(i)[:100] for i in args], kw}): {te-ts}s")
    return result
  return wrap

def getData(img=None, format="FORMAT_NCHW", dtype=np.float32, c=3, h=224, w=224, scaling="INCEPTION", model="resnet"):
  """
  Pre-process an image to meet the size, type and format
  requirements specified by the parameters.
  """
  #np.set_printoptions(threshold='nan')
  
  from PIL import Image
  if img is None:
    img = Image.open("mug.jpg")
  
  if c == 1:
    sample_img = img.convert('L')
  else:
    sample_img = img.convert('RGB')

  resized_img = sample_img.resize((w, h), Image.BILINEAR)
  resized = np.array(resized_img)
  if resized.ndim == 2:
    resized = resized[:,:,np.newaxis]

  typed = resized.astype(dtype)

  if scaling == 'INCEPTION':
    scaled = (typed / 128) - 1
  elif scaling == 'VGG':
    if c == 1:
      scaled = typed - np.asarray((128,), dtype=dtype)
    else:
      scaled = typed - np.asarray((123, 117, 104), dtype=dtype)
  else:
    scaled = typed

  # Swap to CHW if necessary
  #if format == model_config.ModelInput.FORMAT_NCHW:
  ordered = np.transpose(scaled, (2, 0, 1))
  #else:
  #    ordered = scaled
  if model == "inception":
    ordered = np.transpose(ordered, (2, 1, 0))

  # Channels are in RGB order. Currently model configuration data
  # doesn't provide any information as to other channel orderings
  # (like BGR) so we just assume RGB.
  return json.dumps([[ordered.tolist()]])
    
def getModelInfo(db=None, json_file="workload/models.json", ):
  with open(json_file) as json_fid:
    model_stats = json.load(json_fid)
    #logging.debug(f"model_stats: {model_stats}")
    
  model_descriptions = {}
  for model_dict in model_stats:
    logging.debug(f"model_dict: {model_dict}")
    model_name = model_dict["name"]
    avg_exec_latency = model_dict["avg_exec_latency"]
    avg_load_latency = model_dict["avg_load_latency"]
    avg_unload_latency = model_dict["avg_unload_latency"]
    loaded_size = model_dict["loaded_size"]
    
    model_descriptions[model_name] = {
      "load_latency"  : avg_load_latency,
      "exec_latency"  : avg_exec_latency,
      "unload_latency"  : avg_unload_latency,
      "loaded_size"  : loaded_size,
    }
    
    if db is not None:
      logging.debug("Setting model info")
      db.set(f"{MODEL_STAT_PREFIX}{model_name}{DB_FIELD_SEPARATOR}avg_exec_latency", avg_exec_latency)
      db.set(f"{MODEL_STAT_PREFIX}{model_name}{DB_FIELD_SEPARATOR}avg_load_latency", avg_load_latency)
      db.set(f"{MODEL_STAT_PREFIX}{model_name}{DB_FIELD_SEPARATOR}avg_unload_latency", avg_unload_latency)
      db.set(f"{MODEL_STAT_PREFIX}{model_name}{DB_FIELD_SEPARATOR}loaded_size", loaded_size)
    else:
      logging.debug("Not setting any model info")
  return model_descriptions


class InferenceRequest(object):
  _ids = count(1)
  def __init__(self, model_name, data, id_num=None, allow_timeout=True):
    if id_num is None:
      self.id = next(self._ids)
    else:
      self.id = id_num
    self.model_name = model_name
    self.data = data
    
    self.allow_timeout = allow_timeout
    
    self.complete = Event() 
    #self.entry_time = time.time()
    self.times = {
      "entry_time" : time.time(),
      "assignment_time" : 0.,
      "execution_time" : 0.,
      "completion_time" : 0.,
    }
    self.model_miss = False
    self.response = None
    
  #def __getattribute__(self, name):
  #  if object.__getattribute__(self, "allow_timeout"):
  #    if object.__getattribute__(self, "times")["entry_time"] + TIMEOUT_IN_SECONDS < time.time():
  #      raise exceptions.RequestTimeoutException("InferenceRequest invalid (timeout)")
  #  return object.__getattribute__(self, name)
  
  def __repr__(self):
    return f"<{self.__class__.__name__}: {self.id}, {self.model_name}, \"{self.response}\">"
  def __str__(self):
    return repr(self)
    
  def isTimedOut(self):
    return time.time() >= self.times["entry_time"] + TIMEOUT_IN_SECONDS
    
  
  def toJSONString(self):
    attr_dict = {
      "id" : self.id,
      "allow_timeout": self.allow_timeout,
      "model_name" : self.model_name,
      "data" : self.data,
      "times" : self.times,
      "response" : self.response,
      "model_miss" : self.model_miss
    }
    return json.dumps(attr_dict)
  
  @classmethod
  def fromJSONString(cls, json_str):
    attr_dict = json.loads(json_str)
    new_request = cls(
              model_name=attr_dict["model_name"],
              data=attr_dict["data"],
              id_num=attr_dict["id"],
              allow_timeout=attr_dict["allow_timeout"],
            )
    new_request.times = attr_dict["times"]
    new_request.response = attr_dict["response"]
    new_request.model_miss = attr_dict["model_miss"]
    return new_request
  
  def markAssignment(self):
    self.times["assignment_time"] = time.time()
  def markExecution(self):
    self.times["execution_time"] = time.time()
  def markCompletion(self):
    self.times["completion_time"] = time.time()
  
  def markModelMiss(self):
    self.model_miss = True
  
  def mergeRequests(self, other):
    self.times = other.times
    self.response = other.response
  
  def getResponse(self):
    response_dict = {
      "model" : self.model_name,
      "response" : self.response,
      "placement_delay" : self.times["assignment_time"] - self.times["entry_time"],
      "queue_delay" : self.times["execution_time"] - self.times["assignment_time"],
      "execution_delay" : self.times["completion_time"] - self.times["execution_time"],
      "overall_latency" : time.time() - self.times["entry_time"],
      "model_miss" : self.model_miss,
    }
    return json.dumps(response_dict)

class Metrics(object):
  # TODO: make async
  
  def __init__(self, db, prefix, names_of_metrics=[], last_used_metrics=[], report_interval=REPORT_INTERVAL):
    self.db = db
    self.prefix = prefix
    self.report_interval = report_interval
    
    self.names_of_metrics = names_of_metrics
    self.metrics = {
      metric_name : collections.defaultdict(int)
      for metric_name in self.names_of_metrics
    }
    
    self.last_used_metrics = last_used_metrics
    
    self.lock = Lock()
    
    self.metrics_thread = threading.Timer(self.report_interval, self.pushMetrics)
    self.metrics_thread.start()
  
  def incrementMetricBy(self, metric_name, field_name, delta_value=1):
    with self.lock:
      self.metrics[metric_name][field_name] += delta_value
    
  
  def pushMetrics(self):
    #logging.info(f"pushMetrics")
    
    # Restart for next metrics
    self.metrics_thread = threading.Timer(self.report_interval, self.pushMetrics)
    self.metrics_thread.start()
    
    with self.lock:
      metrics_to_report = self.metrics
      self.metrics = {
        metric_name : collections.defaultdict(int)
        for metric_name in self.names_of_metrics
      }
      
    report_time = time.time()
    pipe = self.db.pipeline()
    for metric_name, metrics in metrics_to_report.items():
      for field_name in metrics:
        pipe.incrby( f"{self.prefix}{field_name}{DB_FIELD_SEPARATOR}{metric_name}", metrics[field_name] )
    for metrics in self.last_used_metrics:
      for field_name in metrics_to_report[metrics]:
        pipe.set( f"{self.prefix}{field_name}{DB_FIELD_SEPARATOR}last_used" , report_time )
    #pipe.execute()
      
    results = pipe.execute()

class RedisInterface(object):
  def __init__(self, redis_server="redis-server", redis_port=6379, *args, **kwargs):
    import redis
    self.db = redis.Redis(host=redis_server, port=redis_port)
    try:
      self.db.ping()
    except redis.exceptions.ConnectionError as e:
      print(f"Error connecting to Redis server @ {redis_server}.  Is it started?")
      print(e)
    
class ModelPlacements(object):
  class Model(object):
    def __init__(self, model_name, model_info=None):
      self.name = model_name
      if model_info is not None:
        self.load_latency = model_info["load_latency"]
        self.exec_latency = model_info["exec_latency"]
        self.unload_latency = model_info["unload_latency"]
        self.loaded_size = model_info["loaded_size"]
      else:
        self.load_latency = 0
        self.exec_latency = 0
        self.unload_latency = 0
        self.loaded_size = 0

      self.last_used = 0.0
      
    def __str__(self):
      return self.name
    def __hash__(self):
      return hash(self.name)
    def __lt__(self, other):
      return self.name < other.name
    def __eq__(self, other):
      if isinstance(other, self.__class__):
        return self.name == other.name
      else:
        return self.name == other
      
    def getLoadLatency(self):
      return self.load_latency
    def getExecLatency(self):
      return self.exec_latency
    def getUnloadLatency(self):
      return self.unload_latency
    def getSize(self):
      return self.loaded_size
    def getName(self):
      return self.name

  def __init__(self, *args, **kwargs):
    self.models = set()
    self.workers = set()
    
    self.__workers_by_model = collections.defaultdict(set)
    self.__models_by_worker = collections.defaultdict(set)
    
    self.additions = queue.Queue()
    self.removals = queue.Queue()
    
    self.lock = Lock()
    self.is_synced = True
  def __str__(self):
    return '|'.join([f"{m}:{self.__workers_by_model[m]}" for m in self.models])
  
  def addModel(self, model):
    self.models.add(model)
  def addWorker(self, worker):
    self.workers.add(worker)
  def getModels(self):
    return self.models
  def getWorkers(self):
    return self.workers
  
  def sync(self):
    with self.lock:
      self.__models_by_worker = collections.defaultdict(set) #{ worker : set([]) for worker in self.workers }
      for (model, workers) in self.__workers_by_model.items():
        for worker in workers:
          self.__models_by_worker[worker].add(model)
      self.is_synced = True
  
  def getModelsFromWorker(self, worker):
    logging.debug("getModelsFromWorker()")
    with self.lock:
      if not self.is_synced:
        self.sync()
      return list(self.__models_by_worker[worker])
  def getWorkersFromModel(self, model):
    return list(self.__workers_by_model[model])
  
  def getModelsByWorker(self):
    with self.lock:
      if not self.is_synced:
        self.sync()
      return self.__models_by_worker
  def getWorkersByModel(self):
    return self.__workers_by_model
  
  def addModelToWorker(self, worker, model):
    with self.lock:
      self.__workers_by_model[model].add(worker)
      self.is_synced = False
      self.additions.put( (worker, model) )
  def removeModelFromWorker(self, worker, model):
    logging.debug(f"removeModelFromWorker(self, {worker}, {model})")
    with self.lock:
      self.__workers_by_model[model].remove(worker)
      self.is_synced = False
      self.removals.put( (worker, model) )
  
  def getEmptyWorkers(self):
    logging.debug(f"getEmptyWorkers")
    return [w for w in self.getWorkers() if len(self.getModelsFromWorker(w)) == 0]
  def getModelsInCache(self):
    return [m for m in self.getModels() if len(self.getWorkersFromModel(m)) > 0]
    

def get_subsets_over_size(list_of_models, size_limit):
  list_of_subsets = []
  if len(list_of_models) == 0:
    return list_of_subsets
  for i, base_model in enumerate(list_of_models):
    if base_model.getSize() > size_limit:
      list_of_subsets.append([base_model])
    else:
      for subset in get_subsets_over_size(list_of_models[i+1:], size_limit-base_model.getSize()):
        list_of_subsets.append([base_model] + subset)
  return list_of_subsets

  

def main():
  models_info = getModelInfo(json_file="workload/models.azure.json")
  models = [ModelPlacements.Model(m_name, m_info) for m_name, m_info in models_info.items()]
  
  input_models = sorted(models[:100], key=(lambda m: m.getSize()), reverse=True)

  for m_set in get_subsets_over_size(input_models, 0.6):
    print([str(m) for m in m_set])

  pass

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()
  
  
