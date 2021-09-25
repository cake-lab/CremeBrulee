#!env python

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "protos"))
import grpc
import worker_pb2
import worker_pb2_grpc
from concurrent import futures

import json
import time

# Mine
import exceptions
import common

import logging

class WorkerClient(object):
  def __init__(self, worker_url):
    self.channel = grpc.insecure_channel(f"{worker_url}", options=(('grpc.enable_http_proxy', 0),))
    self.stub = worker_pb2_grpc.WorkerStub(self.channel)
  def __del__(self):
    self.channel.close()
    del self.channel
  def infer(self, inference_request):
    #response_future = self.stub.Infer.future(worker_pb2.InferenceRequest(data=json.dumps(data), model_name=model_name))
    response_future = self.stub.Infer.future(worker_pb2.InferenceRequest(json_obj=inference_request.toJSONString()))
    response = response_future.result()
    if response.infer_status == worker_pb2.InferenceResponse.Status.SUCCESS:
      return common.InferenceRequest.fromJSONString(response.json_obj)
    else:
      raise exceptions.WorkerException(response.error_msg)

  def loadModel(self, model_name):
    response_future = self.stub.Load.future(worker_pb2.ModelManageRequest(model_name=model_name))
    response = response_future.result()
    if response.model_status == worker_pb2.ModelManageResponse.Status.LOADED:
      return True
    elif response.model_status == worker_pb2.ModelManageResponse.Status.UNAVAILABLE:
      raise exceptions.ModelNotInRepoException(response.error_msg)
    elif response.model_status == worker_pb2.ModelManageResponse.Status.NOT_ENOUGH_SPACE:
      raise exceptions.NotEnoughFreeSpace(response.error_msg)

  def unloadModel(self, model_name):
    response_future = self.stub.Unload.future(worker_pb2.ModelManageRequest(model_name=model_name))
    response = response_future.result()
    if response.model_status == worker_pb2.ModelManageResponse.Status.UNLOADED:
      return True
    elif response.model_status == worker_pb2.ModelManageResponse.Status.UNAVAILABLE:
      raise exceptions.ModelNotInRepoException(response.error_msg)
  
  def getMetrics(self):
    logging.info("getMetrics() gRPC call")
    response_future = self.stub.GetMetrics.future(worker_pb2.MetricsRequest())
    response = response_future.result()
    return json.loads(response.metrics_json)
    
class WorkerServer(worker_pb2_grpc.WorkerServicer):
  
  def __init__(self, worker):
    self.worker = worker
    
  def Infer(self, request, context):
    inference_request = common.InferenceRequest.fromJSONString(request.json_obj)
    
    try:
      inference_request = self.worker.requestInference(inference_request)
      response = worker_pb2.InferenceResponse(response=inference_request.response, infer_status=worker_pb2.InferenceResponse.Status.SUCCESS, json_obj=inference_request.toJSONString())
    except exceptions.ModelInferenceException as e:
      logging.error("Error running inference")
      logging.error(e)
      response = worker_pb2.InferenceResponse(response="-1", infer_status=worker_pb2.InferenceResponse.Status.FAILURE, error_msg=str(e))
    except exceptions.WorkerException as e:
      logging.error("Error processing request")
      logging.error(e)
      response = worker_pb2.InferenceResponse(response="-1", infer_status=worker_pb2.InferenceResponse.Status.FAILURE, error_msg=str(e))
    return response
    
  def Load(self, request, context):
    model_name = request.model_name
    try:
      self.worker.loadModel(model_name)
      response = worker_pb2.ModelManageResponse(model_status=worker_pb2.ModelManageResponse.Status.LOADED)
    except exceptions.ModelNotInRepoException as e:
      logging.error("Error processing request")
      logging.error(e)
      response = worker_pb2.ModelManageResponse(model_status=worker_pb2.ModelManageResponse.Status.UNAVAILABLE, error_msg=str(e))
    except exceptions.NotEnoughFreeSpace as e:
      logging.error("Error processing request")
      logging.error(e)
      response = worker_pb2.ModelManageResponse(model_status=worker_pb2.ModelManageResponse.Status.NOT_ENOUGH_SPACE, error_msg=str(e))
    return response
  
  def Unload(self, request, context):
    model_name = request.model_name
    try:
      self.worker.unloadModel(model_name)
      response = worker_pb2.ModelManageResponse(model_status=worker_pb2.ModelManageResponse.Status.UNLOADED)
    except exceptions.ModelNotInRepoException:
      logging.error("Error processing request")
      logging.error(e)
      response = worker_pb2.ModelManageResponse(model_status=worker_pb2.ModelManageResponse.Status.UNAVAILABLE, error_msg=str(e))
    return response
  
  def GetMetrics(self, request, context):
    try:
      # Currently unused as workers report their metrics directly to Redis
      pass
    except exceptions.WorkerException as e:
      logger.error("Error getting metrics")
      logger.error(e)
      metrics_dict = {}
    return worker_pb2.MetricsResponse(metrics_json=json.dumps(metrics_dict))
      
  
  @classmethod
  def serve(cls, worker):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    worker_server = cls(worker)
    worker_pb2_grpc.add_WorkerServicer_to_server(worker_server, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()
        
      
    
    

