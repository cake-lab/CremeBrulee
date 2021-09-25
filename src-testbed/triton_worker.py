import numpy as np
import tensorrtserver.api
import tensorrtserver.api.model_config_pb2 as model_config
import json
import requests
import time

#import asyncio
import threading

from tritonhttpclient import InferenceServerClient
import enum

import os
import common
#logger = common.getLogger() #(f"{os.path.basename(__file__).replace('.py', '')}")
import logging as logger

HTTP = 0
CONNECTION_ATTEMPTS_THRESHOLD = 10


class ModelNotInRepoException(Exception):
  pass
class ModelLoadException(Exception):
  pass
class ModelNoLongerLoadedException(Exception):
  pass

class ModelStatus(enum.Enum):
  # Using defitinos from triton-inference-server/src/core/model_repository_manager.h
  ## This makes life easier when setting status
  UNKNOWN = 0
  READY = 1
  UNAVAILABLE = 2
  LOADING = 3
  UNLOADING = 4


class Model(object):
  def __init__(self, worker, model_name, *args, **kwargs):
    self.worker = worker
    self.model_name = model_name
    self.status = ModelStatus.UNKNOWN
    self.status_obj = None
    self.model_version = 1
    self.inputs = []
    self.outputs = []
    
  def updateInfoFromStatus(self, status_obj):
    
    # This uses a nasty hack
    # TODO: fix the str(status_obj) bit
    if status_obj is None or str(status_obj) == "":
      self.status = ModelStatus(ModelStatus.UNKNOWN)
      self.inputs = []
    else:
      #logger.debug(f"status_obj: '{status_obj}'")
      self.status = ModelStatus(status_obj.version_status[self.model_version].ready_state)
      self.inputs = status_obj.config.input
      self.outputs = status_obj.config.output
  
  def runInference(self, data):
    logger.info(f"runInference {self.model_name} @ {self.worker.worker_url}")
    with self.worker.context_lock:
      inferContext = tensorrtserver.api.InferContext(self.worker.worker_url, HTTP, self.model_name)
    
    @common.timing
    def prepInference():
      inputs = {}
      for i, input in enumerate(self.inputs):
        inputs[input.name] =  list(map(lambda x: np.array(x).astype(model_dtype_to_np(input.data_type)), data[i]))
      outputs = {}
      for output in self.outputs:
        outputs[output.name] = (tensorrtserver.api.InferContext.ResultFormat.CLASS, 1)
      return inputs, outputs
      
    @common.timing
    def runInference():
      return json.dumps(inferContext.run(inputs, outputs))
    
    (inputs, outputs), prep_time = prepInference()
    (response), exec_time = runInference()
    
    logger.debug(f"prep_time: {prep_time}")
    logger.debug(f"exec_time: {exec_time}")
    return response
      
      

#A worker that is supported by a triton infrence server
class TritonWorker:
  
  def __init__(self, worker_url, *args, **kwargs):
    self.worker_url = worker_url
    self.outstanding_requests = 0
    self.models_in_repo = {}
    self.model_status = {}
    
    # Maybe a queue, but everything should be in here I think
    ## Leaving this as a list would allow us to pull out specific items
    ## items would need to be of either specific classes, or a tuple with what things are
    ## e.g. - ("infer", request), ("model_change", request)
    #self.action_queue = asyncio.Queue()
    
    # This thread (may become IO process) will be responsible for processing things in the queue
    self.processing_thread = None
    
    # Set up the client itself.
    self.inference_server_client = InferenceServerClient(url=self.worker_url)
    
    self.ready = False
    
    # might be helpful: https://stackoverflow.com/questions/52582685/using-asyncio-queue-for-producer-consumer-flow
    #self.loop = asyncio.get_event_loop()
    #self.access_lock = threading.RLock()
    
    self.model_lock = threading.Lock()
    self.context_lock = threading.Lock()
  
  async def processQueue():
    while True:
      action = await self.action_queue.get()
      # Do work
      self.action_queue.task_done()
    
    
    
    
  def isLive(self):
    try:
      return self.inference_server_client.is_server_live()
    except ConnectionRefusedError:
      return False
  def isReady(self):
    try:
      with self.context_lock:
        healthContext = tensorrtserver.api.ServerHealthContext(self.worker_url, HTTP)
      return healthContext.is_ready()
    except tensorrtserver.api.InferenceServerException as e:
      logger.warning(f"Readiness check failed on {self.worker_url}")
      logger.warning(str(e))
      return False
    #try:
    #    return self.inference_server_client.is_server_ready()
    #except ConnectionRefusedError:
    #    return False
  
  @classmethod
  def addWorker(cls, worker_url, *args, **kwargs):
    logger.info(f"TritonWorker: addWorker(...) start")
    logger.debug(f"worker_url: {worker_url}")
    # Make new worker
    new_worker = cls(worker_url, *args, **kwargs)
    
    
    is_ready = new_worker.isReady()
    for attempt_num in range(CONNECTION_ATTEMPTS_THRESHOLD):
      logger.info(f"Ready check for '{worker_url}'' {attempt_num+1}/{CONNECTION_ATTEMPTS_THRESHOLD}...")
      if is_ready:
        break
      time.sleep(1)
      is_ready = new_worker.isReady()
    
    if not is_ready:
      logger.info(f"Worker @ {worker_url} failed to become ready.")
      return None
    else:
      logger.info(f"Worker @ {worker_url} is ready")
    
    # I don't know what this does right now, and will need to go through it shortly.
    new_worker.updateModelList()
    new_worker.updateModelStatuses()
    logger.info(f"Worker @ {worker_url} set up and ready to use.")
    
    logger.info(f"TritonWorker: addWorker(...) end")
    #loop.run_until_complete(main())
    return new_worker
  
  def updateModelList(self):
    #Get a list of models that the worker has
    with self.context_lock:
      repoContext = tensorrtserver.api.ModelRepositoryContext(self.worker_url, HTTP)
      self.models_in_repo = {entry.name : Model(self, entry.name) 
                  for entry in repoContext.get_model_repository_index().models
                }
    logger.debug(f"Models in repo: {self.models_in_repo.keys()}")
  
  def updateModelStatuses(self):
    logger.info("updateModelStatuses(...) start")
    #Get the status of each of those models
    with self.context_lock:
      statusContext = tensorrtserver.api.ServerStatusContext(self.worker_url, HTTP)
      status = statusContext.get_server_status().model_status
    
    for model_name in self.models_in_repo.keys():
      logger.debug(f"Updating status for {model_name}")
      model_status = status[model_name]
      self.models_in_repo[model_name].updateInfoFromStatus(model_status)
      #logger.debug(f"{model_name} @ {self.worker_url} : {self.models_in_repo[model_name].status}")
      
      self.model_status[model_name] = self.models_in_repo[model_name].status
    
    logger.info("updateModelStatuses(...) end")
      
  
    
  def hasModelLoaded(self, model_name):
    return self.models_in_repo[model_name].status == ModelStatus.READY
  
  def requestInference(self, model_name, data):
    logger.info(f"requestInference(...): {model_name} @ {self.worker_url}")
    
    # Check if model is even in repo
    if model_name not in self.models_in_repo.keys():
      logger.error("Requested model not available in repo")
      raise ModelNotInRepoException
    
    with self.model_lock:
      # Grab instance of model
      model = self.models_in_repo[model_name]
      
      # Loop until model is ready.  Note, this might never occur and if it doesn't then yolo
      ## Double note: this will never actually occur as the status is never checked again
      ## TODO: Make actually reasonable
      while model.status != ModelStatus.READY:
        logger.warning(f"{model_name} @ {self.worker_url} not ready for inference.  Sleeping....")
        time.sleep(1)
        if model.status not in [ModelStatus.READY, ModelStatus.LOADING]:
          raise ModelNoLongerLoadedException
      
      response = model.runInference(data)
      logger.info(f"Response: {response}")
      return response
    
    
  def infer(self, data, model_name):
    return self.requestInference(model_name, data)
    
  
  
  #Attempts to load the model
  def loadModel(self, model_name):
    logger.info("loadModel(..) start")
    logger.info(f"loading: {model_name} @ {self.worker_url}")
    
    with self.model_lock:
      
      if self.models_in_repo[model_name].status in [ModelStatus.READY, ModelStatus.LOADING]:
        logger.warning("Model already loaded/loading")
        return True
      
      if not self.isThereFreespaceToLoad(model_name):
        raise ModelLoadException("Not enough free space to load model")
      
      try:
        with self.context_lock:
          modelControlContext = tensorrtserver.api.ModelControlContext(self.worker_url, HTTP)
          logger.info(f"modelControlContext: {modelControlContext}")
          
          # TODO: Make load happen in an async fashion
          modelControlContext.load(model_name)
        self.updateModelStatuses()
      except tensorrtserver.api.InferenceServerException:
        logger.critical(f"Model Server @ {self.worker_url} failed to respond to model load event.  Likely died.")
        return False
      
      return True
  
  
  def unloadModel(self, model_name):
    logger.info("unloadModel(..) start")
    logger.info(f"unloading: {model_name} @ {self.worker_url}")
    
    with self.model_lock:
      if self.models_in_repo[model_name].status in [ModelStatus.UNKNOWN, ModelStatus.UNAVAILABLE, ModelStatus.UNLOADING]:
        logger.warning("Model already unloaded/unloading")
        return True
      with self.context_lock:
        modelControlContext = tensorrtserver.api.ModelControlContext(self.worker_url, HTTP)
        logger.info(f"modelControlContext: {modelControlContext}")
        
        # TODO: Make load happen in an async fashion
        modelControlContext.unload(model_name)
      self.updateModelStatuses()
      
      return True
  
  
  
  def isThereFreespaceToLoad(self, model_name):
    """
    Checks whether there is enough free space to load model.
    Currently simply limits number of models that can be loaded to a single model since otherwise things die.
    """
    return True
    return 0 == len([model for model in self.models_in_repo.values() if model.status in [ModelStatus.READY, ModelStatus.LOADING]])
  
  
  
  




#Taken from https://github.com/NVIDIA/triton-inference-server (trition-inference-server/src.clients/ptyhon/api_v1/examples/image_client.py)
def model_dtype_to_np(model_dtype):
  if model_dtype == model_config.TYPE_BOOL:
    return np.bool
  elif model_dtype == model_config.TYPE_INT8:
    return np.int8
  elif model_dtype == model_config.TYPE_INT16:
    return np.int16
  elif model_dtype == model_config.TYPE_INT32:
    return np.int32
  elif model_dtype == model_config.TYPE_INT64:
    return np.int64
  elif model_dtype == model_config.TYPE_UINT8:
    return np.uint8
  elif model_dtype == model_config.TYPE_UINT16:
    return np.uint16
  elif model_dtype == model_config.TYPE_FP16:
    return np.float16
  elif model_dtype == model_config.TYPE_FP32:
    return np.float32
  elif model_dtype == model_config.TYPE_FP64:
    return np.float64
  elif model_dtype == model_config.TYPE_STRING:
    return np.dtype(object)
  return None
