#!env python



import time
import requests
import enum
import json
import docker

import numpy as np
from itertools import count
import random
import base64
import os
import common
#logging = common.getlogging() #common.getlogging() #f"{os.path.basename(__file__).replace('.py', '')}")
import logging as logging

import exceptions

import sys
import queue
import functools
#sys.path.append("/tmp/serving/bazel-bin/tensorflow_serving/config")
#import model_server_config_pb2
import threading

import wrapt


def logInfo(event_type):
  @wrapt.decorator
  def wrapper(wrapped, instance, args, kwargs):
    #logging.info(instance.server_container.stats(stream=False))
    if instance.record:
      start_mem = instance.server_container.stats(stream=False)["memory_stats"]["usage"]
    t_start = time.time()
    result = wrapped(*args, **kwargs)
    t_end = time.time()
    if instance.record:
      end_mem = instance.server_container.stats(stream=False)["memory_stats"]["usage"]
    
    if instance.record:
      instance.writeToLog(instance.model_name, event_type, "MEMORY", start_mem, {"state" : "prior"})
      instance.writeToLog(instance.model_name, event_type, "MEMORY", end_mem, {"state" : "post"})
      instance.writeToLog(instance.model_name, event_type, "MEMORY", end_mem-start_mem, {"state" : "delta"})
      instance.writeToLog(instance.model_name, event_type, "TIME", (t_end - t_start))
    
    return result
  return wrapper



class ModelStatus(enum.Enum):
  # Using defitinos from triton-inference-server/src/core/model_repository_manager.h
  ## This makes life easier when setting status
  UNKNOWN = 0
  UNLOADED = 1
  LOADING = 2
  READY = 3
  UNLOADING = 4
  QUEUED_TO_LOAD = 5
  QUEUED_TO_UNLOAD = 6

class Port():
  _port = count(random.randrange(49152, 65535))
  @classmethod
  def getNextPort(cls):
    return next(cls._port)
  
class Model(object):
  
  _ids = count(1)
  
  def __init__(self, model_name, model_path, *args, **kwargs):
    self.id = next(self._ids)
    self.model_name = model_name
    self.model_path = model_path
    
    self.execution_latencies = []
    self.execution_latencies_cold = []
    self.first_exec = True
    
    self.load_times = []
    self.exec_times = []
    self.unload_times = []
    self.model_sizes = []
    
    self.expected_exec_latency = 0
    self.expected_load_latency = 0
    self.expected_unload_latency = 0
    self.expected_model_size = 0
    
    self.status = ModelStatus.UNLOADED
    
    self.usages_since_last_start = 0
  
  def __repr__(self):
    return f"<{self.__class__}: {self.id}, {self.model_name}, {self.model_path}, {self.status}>"
  def __str__(self):
    return repr(self)
  
  ######################
  ## Public Functions ##
  ######################
  def isAvailable(self):
    return self.status in [ModelStatus.READY, ModelStatus.LOADING]
  def isLoaded(self):
    return self.status in [ModelStatus.READY]
  
  @common.timing
  def startModel(self, model_stats={"loaded_size":0}, callback_func=None, *args, **kwargs):
    """Starts model container.  Calls _startModel after doing checking."""
    logging.info(f"startModel() called for {self.model_name}")
    
    # Check if model already running
    if self.isAvailable():
      raise exceptions.ModelAlreadyLoadedException(f"Model ({self.model_name}) already loaded, cannot load again")
    
    # These are for when we're using dummy models
    #self.expected_exec_latency = model_stats["expected_exec_latency"]
    #self.expected_load_latency = model_stats["expected_load_latency"]
    #self.expected_unload_latency = model_stats["expected_unload_latency"]
    self.expected_model_size = model_stats["loaded_size"]
    
    # Start model
    logging.debug("Starting models")
    self.status = ModelStatus.LOADING
    self._startModel(*args, **kwargs)
    
    self.usages_since_last_start = 0
    self.first_exec = True
    
    logging.info(f"Requested '{self.model_name}'' start.  callback_func: {callback_func}")
    if callback_func is None:
      self.waitUntilLive()
    else:
      threading.Thread(target=self.waitUntilLive, kwargs={'callback_func':callback_func}).start()
  
  def waitUntilLive(self, callback_func=None, *args, **kwargs):
    logging.debug("Checking if model is live")
    # Wait until model is live
    model_liveness = False
    while not model_liveness:
      try:
        model_liveness = self._isLive(*args, **kwargs)
      except exceptions.ServerDeathException as e:
        logging.error(f"ServerDeathException: {e}")
        #self.restart()
      time.sleep(common.POLL_SLEEP_MS / 1000.)
      logging.debug("Still not live")
      
    logging.info(f"Model ({self.model_name}) now live!")
    
    self.status = ModelStatus.READY
    if callback_func is not None:
      callback_func()
    
  def stopModel(self, *args, **kwargs):
    """Stops model container. Calls _stopModel after doing checking."""
    logging.debug(f"stopModel() called for {self.model_name} {self.status}")
    if not self.isAvailable():
      raise exceptions.ModelAlreadyStoppedException("Model already unloaded, cannot unload again")
    self._stopModel(*args, **kwargs)
    self.status = ModelStatus.UNLOADED
  
  def shutdown(self, *args, **kwargs):
    self._stopModel(*args, **kwargs)
  
  def getStatistics(self, *args, **kwargs):
    #logging.debug(f"getStatistics({self.model_name})")
    stats = {
      "avg_exec_latency" : self.getExecTime(),
      "avg_load_latency" : self.getLoadTime(),
      "loaded_size" : self.getModelSize(),
    }
    #logging.debug(f"stats: {stats}")
    return stats
    
  
  @common.timing
  def runInference(self, inference_request, *args, **kwargs):
    """Requests inference on model.  Generalized"""
    logging.debug(f"runInference() called on {self.model_name}")
    
    if not self.isLoaded():
      logging.debug(self)
      raise exceptions.ModelNotReadyException("Model not ready to run inference")
    inference_request.markExecution()
    
    start_time = time.time()
    inference_request.response = self._infer(inference_request.data, *args, **kwargs)
    exec_latency = time.time() - start_time
    logging.debug(f"exec_latency: {exec_latency}")
    #self.execution_latencies.append(exec_latency)
    #if self.first_exec:
    #  self.execution_latencies_cold.append(exec_latency)
    #  self.first_exec = False
    
    self.usages_since_last_start += 1
    inference_request.markCompletion()
    return inference_request

  def getLoadTime(self):
    if len(self.execution_latencies_cold) == 0:
      return 0
    return np.average(self.execution_latencies_cold) - self.getExecTime()
  def getExecTime(self):
    if len(self.execution_latencies) == 0:
      return 0
    return np.average(self.execution_latencies)
  def getModelSize(self):
    return self._getMemoryUsage()
    
  ######################
  ######################
  
  
  #######################
  ## Private Functions ##
  #######################
  def _startModel(self, *args, **kwargs):
    raise NotImplementedError("_startModel not yet implemented")
  def _stopModel(self, *args, **kwargs):
    raise NotImplementedError("_stopModel not yet implemented")
  def _isLive(self, *args, **kwargs):
    raise NotImplementedError("_isLive not yet implemented")
  def _infer(self, *args, **kwargs):
    raise NotImplementedError("_infer not yet implemented")
  
  def _getMemoryUsage(self, *args, **kwargs):
    raise NotImplementedError("_getMemoryUsage not yet implemented")
  
  
    
  #######################
  #######################

class DummyModel(Model):
  
  def _startModel(self, *args, **kwargs):
    logging.info(f"DummyModel._startModel: ({self.model_name})")
    time.sleep(self.expected_load_latency)
  def _stopModel(self, *args, **kwargs):
    logging.info(f"DummyModel._stopModel: ({self.model_name})")
    time.sleep(self.expected_unload_latency)
  def _isLive(self, *args, **kwargs):
    return True
  def _infer(self, *args, **kwargs):
    logging.info(f"DummyModel._infer: ({self.model_name})")
    time.sleep(self.expected_exec_latency)
    return f"Dummy ({self.model_name}) FTW"
  def _getMemoryUsage(self, *args, **kwargs):
    return self.expected_model_size

class DockerBasedModel(Model):
  """
  Base model for docker-based models inference.
  These will be equivalent to one container running in docker.
  Can be extended to be for either TFS or Triton
  """
  
  image_name = "alpine"
  
  def __init__(self, model_name, model_path=None, docker_client=None, network_name=None, name_prefix=None, *args, **kwargs):
    """
    __init__ function sets up worker, including making (but not running) the container.
    """
    super().__init__(model_name, model_path)
    
    logging.info(f"model_name: {model_name}")
    logging.info(f"self.model_name: {self.model_name}")
    
    self.docker_client = docker_client
    self.network_name = network_name
    self.container_name = f"{name_prefix}-model-{self.model_name}"
    self.hostname = self.container_name
    
    
    self.docker_client.images.pull(self.image_name)
    self._initContainer(*args, **kwargs)
    
  
  
  def _initContainer(self, cmd="echo hello", container_kwargs={}, *args, **kwargs):
    """Inits the container that is being used.  Should be updated depending on needed extra flags, if more needed."""
    try:
      self.docker_client.containers.get(self.container_name).remove(force=True, v=True)
    except docker.errors.NotFound:
      pass
    logging.debug(f"cmd: \"{cmd}\"")
    logging.debug(f"volumes: \"{self.model_path}\" : /models/{self.model_name}")
    self.container = self.docker_client.containers.create(self.image_name,
                              command=cmd,
                              name=f"{self.container_name}",
                              hostname=f"{self.hostname}",
                              network=self.network_name,
                              mem_limit="2g",
                              detach=True,
                              volumes={
                                f"{self.model_path}" : {"bind" : f"/tmp/models/{self.model_name}", "mode" : "rw"},
                              },
                              publish_all_ports=True,
                              auto_remove=True,
                              **container_kwargs,
                            )
  
  
  
  def _startModel(self, *args, cmd="echo hello", container_kwargs={}, **kwargs):
    """Starts model container."""
    logging.info(f"DockerBasedModel._startModel() called")
    self.container.start()
    self.container.reload()
    logging.info(f"ports: {self.container.ports}")
    #time.sleep(1)
    logging.debug(f"logs: {self.container.logs()}")
    logging.info(self.container.logs())
  
  
  def _stopModel(self, *args, **kwargs):
    """Stops model container and init's a new one for future use"""
    logging.info(f"_stopModel(self, {args}, {kwargs})")
    self.container.stop()
    self.container.remove()
    if not ("shutdown" in kwargs and kwargs["shutdown"]):
      self._initContainer()
  
  def _getMemoryUsage(self, *args, **kwargs):
    return {}
    #stats = self.container.stats(stream=False)
    #logging.debug(f"stats: {stats}")
  
  def printLogs(self):
    stdout = self.container.logs(stdout=True, stderr=False).decode()
    stderr = self.container.logs(stdout=False, stderr=True).decode()
    logging.debug(f"STDOUT: ")
    for line in stdout.split('\n')[:-10]:
      logging.debug(f"--->{line}")
    logging.debug(f"STDERR: ")
    for line in stderr.split('\n')[:-10]:
      logging.debug(f"--->{line}")

class TritonModel(DockerBasedModel):
  #image_name = "nvcr.io/nvidia/tritonserver:20.07-v1-py3"
  image_name = "nvcr.io/nvidia/tritonserver:20.07-py3"
  
  def _initContainer(self, cmd=None, container_kwargs={}, *args, **kwargs):
    """Inits the container that is being used.  Should be updated depending on needed extra flags, if more needed."""
    
    if cmd is None:
      cmd = "tritonserver --model-repository=/models"
    
    container_kwargs["shm_size"] = "1g"
    super()._initContainer(cmd=cmd, container_kwargs=container_kwargs, *args, **kwargs)
    
  
  def _isLive(self, *args, **kwargs):
    try:
      response = requests.get(f"http://{self.hostname}:8000/v2/health/ready")
      #logging.debug(f"response: {response}")
      if response.status_code == 200:
        return True
    except requests.exceptions.ConnectionError as e:
      logging.error(f"Connection not live: {e}")
      pass
    return False
    
  
  def _infer(self, data, *args, **kwargs):
    logging.info("called _infer")
    
    request_data = {
      "id" : "1",
      "inputs" : [
        {
          "name" : "gpu_0/data",
          "shape" : [1, 3, 224, 224],
          "datatype":"FP32",
          "data" : data
        }  
      ],
      "outputs" : [
        {
          "name":"gpu_0/softmax"
        }
      ]
    }
    
    try:
      response = requests.post(f"http://{self.hostname}:8000/v2/models/{self.model_name}/infer", json=request_data)
      if response.status_code == 200:
        probs = response.json()["outputs"][0]["data"]
        return f"({self.model_name}) index in list: {np.argmax(probs)}"
    except requests.exceptions.ConnectionError as e:
      logging.error("Model container may have died as shown by connection error.  Restarting model")
      logging.error(f"error: {e}")
      loging.error(f"logs: {self.container.logs()}")
      self.restartModel()
    
    raise exceptions.ModelInferenceException("Inference Failed")
  
  def restartModel(self):
    self.stopModel()
    self.startModel()
     
class TFSModel(DockerBasedModel):
  image_name = "tensorflow/serving"
  
  def _initContainer(self, cmd=None, container_kwargs={}, *args, **kwargs):
    """Inits the container that is being used.  Should be updated depending on needed extra flags, if more needed."""
    
    if cmd is None:
      
      #cmd = f"{self.image_name.split(':')[0]} "
      #cmd +=" --port=8500"
      #cmd += " --rest_api_port=8501"
      #cmd += f" --model_name={self.model_name}"
      #cmd += " --model-repository=/models"
      #cmd = "ls -l"
      
      self.rest_api_port = Port.getNextPort()
      
      cmd = ""
      container_kwargs["environment"] = {
        "MODEL_NAME" : f"{self.model_name}",
      }
      container_kwargs["ports"] = {
        "8501/tcp" : self.rest_api_port,
      }
    
    super()._initContainer(cmd=cmd, container_kwargs=container_kwargs, *args, **kwargs)
    self.input_shape = (224, 224, 3)
    
  
  def _isLive(self, *args, **kwargs):
    try:
      response = requests.get(f"http://{self.hostname}:8501/v1/models/{self.model_name}")
      if response.status_code == 200:
        resp_dict = json.loads(response.content)
        logging.debug(resp_dict)
        if any([(version["state"]=="AVAILABLE") for version in resp_dict["model_version_status"]]):
          return True
    except requests.exceptions.ConnectionError as e:
      logging.error(f"Liveness connection failed: {e}")
      raise exceptions.ServerDeathException("Server died during liveness check")
    #self.printLogs()
    return False
  
  
  def _getMetadata(self, *args, **kwargs):
    try:
      response = requests.get(f"http://{self.hostname}:8501/v1/models/{self.model_name}/metadata")
      if response.status_code == 200:
        resp_dict = json.loads(response.content)
        return resp_dict
    except requests.exceptions.ConnectionError as e:
      logging.error(f"Metadata connection failed: {e}")
      raise exceptions.ServerDeathException("Server died during metadata check")
    return None
    
  
  @logInfo("INFERENCE")
  def _infer(self, data, *args, **kwargs):
    logging.info("called _infer")
    #data = np.reshape(np.array(json.loads(data)), self.input_shape)
    
    if self.metadata is None:
      self.metadata = self._getMetadata()
      logging.debug(self.metadata)
      self.input_shape = tuple([int(dim["size"]) for dim in list(self.metadata["metadata"]["signature_def"]["signature_def"]["serving_default"]["inputs"].values())[0]["tensor_shape"]["dim"] if int(dim["size"]) > 0])
      logging.debug(f"inputs: {list(self.metadata['metadata']['signature_def']['signature_def']['serving_default']['inputs'].values())}")
      logging.info(f"input_shape: {self.input_shape}")
      
    data = np.random.rand(*self.input_shape)
    request_data = {"instances": [data.tolist()]}
    try:
      response = requests.post(f"http://{self.hostname}:8501/v1/models/{self.model_name}:predict", json=request_data)
      logging.debug(f"response: {response.content[:200]}")
      if response.status_code == 200:
        probs = response.json()["predictions"][0]
        return f"({self.model_name}) index in list: {np.argmax(probs)}"
      else:
        self.printLogs()
    except requests.exceptions.ConnectionError as e:
      logging.error("Model container may have died as shown by connection error.  Restarting model")
      logging.error(e)
      self.stopModel()
      self.startModel()
    logging.error("Uncaught case it seems")
    raise exceptions.ModelInferenceException("Inference Failed (in _infer)")
  
class TFSModel_Arm(TFSModel):
  #image_name = "tensorflow/serving"
  #image_name = "emacski/tensorflow-serving:latest"
  image_name = "samogden/tfs:latest"
  
  def _initContainer(self, cmd=None, max_memory_in_gb=4, container_kwargs={}, *args, **kwargs):
    """Inits the container that is being used.  Should be updated depending on needed extra flags, if more needed."""
    
    if cmd is None:
      
      self.rest_api_port = Port.getNextPort()
      
      cmd = ""
      container_kwargs["environment"] = {
        "MODEL_NAME" : f"{self.model_name}",
      }
      container_kwargs["ports"] = {
        "8501/tcp" : self.rest_api_port,
      }
      
      cmd = f"tensorflow_model_server \
              --model_base_path=/tmp/models/{self.model_name} \
              --model_name={self.model_name} \
              --total_model_memory_limit_bytes={int(max_memory_in_gb * 1000000000)}"
      
    super()._initContainer(cmd=cmd, container_kwargs=container_kwargs, *args, **kwargs)



class TFSServerModel(TFSModel):
  image_name = None
  image_name_arm = "samogden/tfs:latest"
  image_name_x86 = "tensorflow/serving"
  docker_client = None
  network_name = None
  server_container = None
  hostname = None
  repository_path = None
  records_csv = None
  
  model_names = set()
  
  records_queue = queue.Queue()
  
  def __init__(self, model_name, *args, **kwargs):
    """
    __init__ function sets up worker, including making (but not running) the container.
    """
    
    if self.__class__.server_container is None:
      raise RuntimeError(f"Server container not set up.  Please make sure to call {self.__class__.__name__}.initServer(...) prior to trying to instantiate a model")
      
    super(DockerBasedModel, self).__init__(model_name, self.__class__.repository_path)
    self.model_name = model_name
    logging.debug(f"model_name: {model_name}")
    logging.debug(f"self.model_name: {self.model_name}")
    self.metadata = None
    
  
  def restartModel(self):
    self.initServer(self.docker_client, self.repository_path, self.max_memory_in_gb, self.network_name, self.hostname, self.model_config_file)
  
  def restart(self):
    self.initServer(self.docker_client, self.repository_path, self.max_memory_in_gb, self.network_name, self.hostname, self.model_config_file)
  

  @classmethod
  def initServer(cls, docker_client, model_repository_path, max_memory_in_gb=2, network_name="host", arm=False, host_name="tfs_model_server", model_config_file="/tmp/models/models.config", cmd="echo hello", container_kwargs={}, record=False, *args, **kwargs):
    """Inits the container that is being used.  Should be updated depending on needed extra flags, if more needed."""
    logging.debug(f"Container is arm: {arm}")
    cls.image_name = cls.image_name_arm if arm else cls.image_name_x86
    
    cls.record = record
    #logInfo = functools.partial(logInfo_wrapper, record=record)
    
    cls.docker_client = docker_client
    cls.docker_client.images.pull(cls.image_name)
    cls.hostname = f"{host_name}-{time.time()}"
    cls.max_memory_in_gb = max_memory_in_gb
    cls.network_name = network_name
    cls.model_config_file = model_config_file
    
    
    cls.repository_path = model_repository_path
    # TODO: (sso) this will have to be changed to /tmp/models when it is running in a container itself
    cls.model_config_file = model_config_file
    #cls.model_config_file = os.path.join(model_repository_path, "models.config")
    
    cmd = f"tensorflow_model_server \
            --model_config_file=/tmp/models/models.config \
            --model_config_file_poll_wait_seconds=1 \
            --total_model_memory_limit_bytes={int(1.25 * max_memory_in_gb * (1024*1024*1024))}"
    
    logging.debug("Creating new container...")
    cls.server_container = cls.docker_client.containers.create(
      cls.image_name,
      command=cmd,
      name=f"{cls.hostname}",
      hostname=f"{cls.hostname}",
      network=network_name,
      mem_limit=f"{int( (1024*1024*1024) * (max_memory_in_gb * 1.5) )}m", # Adding a quarter because that seems to be how much memory TFS takes 
      #mem_limit=int( 1024.*1024.*1024.*(max_memory_in_gb * .33)), # Adding a quarter because that seems to be how much memory TFS takes 
      detach=True,
      volumes={f"{model_repository_path}" : {"bind" : f"/tmp/models/", "mode" : "rw"}},
      publish_all_ports=True,
      auto_remove=False,
      **container_kwargs,
    )
    cls.writeConfigFile()
    cls.server_container.start()
    
    if cls.record:  
      cls.records_csv = open("/logs/TFSServerModel-records.csv", 'a+', buffering=1)
      if cls.records_csv.tell() == 0:
        cls.records_csv.write("model,event_type,resource_type,resource_amount,extra_info_json\n")
      cls.records_thread = threading.Thread(target=cls.writeRecords).start()
  
  @classmethod
  def writeRecords(cls):
    while True:
      cls.records_csv.write(cls.records_queue.get())
  
  @classmethod
  def writeConfigFile(cls):
    logging.debug(f"writeConfigFile: {cls.model_config_file}")
    with open(cls.model_config_file, 'w') as fid:
      config_str = ""
      
      config_str += "model_config_list {\n"
      for model in cls.model_names:
        config_str +=  "    config {\n"
        config_str += f"    name: '{model}'\n"
        config_str += f"    base_path: '/tmp/models/{model}'\n"
        config_str += f"    model_platform: 'tensorflow'\n"
        config_str +=  "  }\n"
      config_str +=  "}\n"
      fid.write(config_str)
    
  
  def _startModel(self, *args, **kwargs):
    self.__class__.model_names.add(self.model_name)
    self.writeConfigFile()
  def _stopModel(self, *args, **kwargs):
    self.__class__.model_names.remove(self.model_name)
    self.writeConfigFile()

  
  def printLogs(self):
    stdout = self.__class__.server_container.logs(stdout=True, stderr=False).decode()
    stderr = self.__class__.server_container.logs(stdout=False, stderr=True).decode()
    logging.info(f"STDOUT: ")
    for line in stdout.split('\n')[:-10]:
      logging.info(f"--->{line}")
    logging.info(f"STDERR: ")
    for line in stderr.split('\n')[:-10]:
      logging.info(f"--->{line}")
  
  @logInfo("MODEL_START")
  def startModel(self, *args, **kwargs):
    
    super().startModel(*args, **kwargs)
    
    
  @logInfo("MODEL_STOP")
  def stopModel(self, *args, **kwargs):
    time_start = time.time()
    super().stopModel(*args, **kwargs)
    time_end = time.time()
    
    self.unload_times.append( (time_end - time_start) )
    logging.info(f"stopModel ran in {time_end - time_start}s")

  def getLoadTime(self):
    if len(self.load_times) == 0:
      return 0
    return np.average(self.load_times)
  def getModelSize(self):
    if len(self.model_sizes) == 0:
      return 2
      return self.expected_model_size
    return np.max(self.model_sizes)
  
  def writeToLog(self, model, event_type, resource_type, resource_amount, extra_info_json={}):
    extra_info_json["usages_since_last_start"] = self.usages_since_last_start
    self.__class__.records_queue.put(f"{model},{event_type},{resource_type},{resource_amount},'{json.dumps(extra_info_json)}'\n")
    #self.records_csv.write(f"{model},{event_type},{resource_type},{resource_amount},'{json.dumps(extra_info_json)}'\n")
    
  

if __name__ == "__main__":
  common.getLogger(hide_debug=True)
  
  docker_client = docker.from_env()
  network_name = "host"
  try:
    network = docker_client.networks.get(network_name)
  except docker.errors.NotFound:
    network = docker_client.networks.create(network_name)
  
  try:
    model = TFSServerModel("resnet50_netdef_a")
  except RuntimeError as e:
    logging.error(e)
  logging.info("calling initServer")
  TFSServerModel.initServer(
    docker_client, 
    "/Users/ssogden/research/2020-project-EdgeController/models", 
    model_config_file="/Users/ssogden/research/2020-project-EdgeController/models/models.config", 
    max_memory_in_gb=1, 
    network_name=network_name
  )
  logging.info("creating model")
  model = TFSServerModel("resnet50_netdef_a")
  logging.info("Starting model")
  model.startModel()
  
  model.printLogs()
  
  
  exit()
  import time
  time.sleep(1)
  model.printLogs()
  model._startModel()
  time.sleep(10)
  model.printLogs()
  time.sleep(10)
  model.printLogs()
  logging.info(f"is live: {model._isLive()}")
