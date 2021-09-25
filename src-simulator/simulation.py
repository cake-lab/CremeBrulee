#!env python
import collections
import queue
import logging
import enum
import functools
import json
import time
import os
import gzip
import shutil
import random # ONLY USED FOR RANDOM DELAY AT BEGINNING.

import numpy as np
import argparse

import sys
sys.path.append("../src-testbed")

import events
import common
import placement_controller

class LocalController(object):
  def __init__(self, simulation):
    self.simulation = simulation
    
  def requestInference(self, curr_time, request):
    new_events = []
    if len(self.simulation.model_placements.getWorkersFromModel(request.model)) > 0:
      # There is a placement of the model already
      worker = self.selectWorker(self.simulation.model_placements.getWorkersFromModel(request.model))
      new_events.extend(worker.assignRequest(curr_time, request, model_miss=False))
    elif self.simulation.flags.do_reactive:
      new_events.extend(self.simulation.placement_controller.requestPlacement(curr_time, request))
    else:
      logging.error("No available workers found")
      request.markRejected()
      new_events.append( (curr_time, events.RequestCompletionEvent(self.simulation, request)) )
    
    return new_events
  
  def selectWorker(self, possible_workers):
    return self.simulation.rng.choice(possible_workers)

class PlacementController(object):
  def __init__(self, simulation, flags):
    self.simulation = simulation
    self.flags = flags
    self.model_placements = self.simulation.model_placements
  
    self.placement_controller = placement_controller.PlacementController(flags)
    self.placement_controller.model_placements = self.model_placements # Overwrite with our model_placements
    
  def requestPlacement(self, curr_time, request):
    new_events = []

    model_info = {
            model: {
                    "open_requests"     : (self.simulation.metrics.per_model_requests[model] - self.simulation.metrics.per_model_responses[model]),
                    "last_used"         : model.last_used,
                    "requests_submitted": self.simulation.metrics.per_model_requests[model],
                    "placement_count"   : len(self.model_placements.getWorkersFromModel(model)),
                    "load_latency"      : model.getLoadLatency(),
                    "exec_latency"      : model.getExecLatency(),
                    "loaded_size"       : model.getSize(),
            }
            for model in self.model_placements.getModels()
    }
    
    self.placement_controller.setModelInfo(model_info)
    self.placement_controller.requestToAddModels([request.model], request.id)

    # TODO: Figure out the proper logic on these.  Specifically, this should be negotiated through the local controller
    while not self.placement_controller.model_placements.removals.empty():
      # First we schedule all removals
      worker, model = self.model_placements.removals.get()
      new_events.extend(worker.removeModel(curr_time, model))
      self.simulation.mark_as_saturated()
    
    while not self.placement_controller.model_placements.additions.empty():
      # Next we schedule all additions
      worker, model = self.model_placements.additions.get()
      new_events.extend(worker.addModel(curr_time, model))
    
    # Next we schedule the model on the chosen worker (or see what worker can now take it and assign it)
    if len(self.simulation.model_placements.getWorkersFromModel(request.model)) > 0:
      worker = self.simulation.local_controller.selectWorker(self.simulation.model_placements.getWorkersFromModel(request.model))
      new_events.extend(worker.assignRequest(curr_time, request, model_miss=True))
    else:
      request.markRejected()
      new_events.append( (curr_time, events.RequestCompletionEvent(self.simulation, request)) )
    
    return new_events

@functools.total_ordering
class Worker(object):
  class QueueItem(object):
    def __init__(self, item, latency):
      self.item = item
      self.latency = latency
    
    def getLatency(self):
      return self.latency
  
  def __init__(self, simulation, worker_name, *args, **kwargs):
    self.simulation = simulation
    self.name = worker_name
    self.executing = False
    self.queue = queue.Queue()
    self.models_loaded = set()
    
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
  
  def assignRequest(self, curr_time, request, model_miss):
    new_events = []
    request.assignToWorker(curr_time, model_miss)
    self.queue.put(self.__class__.QueueItem(request, request.model.getExecLatency()))
    if not self.executing:
      new_events.extend(self.startExecuting(curr_time))
    return new_events

  def removeModel(self, curr_time, model):
    new_events = []
    event_to_add = events.ModelRemovalEvent(self.simulation, self, model)
    self.queue.put(self.QueueItem(event_to_add, model.getUnloadLatency()))
    if not self.executing:
      new_events.extend(self.startExecuting(curr_time))
    return new_events

  def addModel(self, curr_time, model):
    new_events = []
    self.queue.put(self.QueueItem(events.ModelAdditionEvent(self.simulation, self, model), model.getLoadLatency()))
    if not self.executing:
      new_events.extend(self.startExecuting(curr_time))
    return new_events
  
  def _removeModel(self, curr_time, model):
    new_events = []
    print(f"({curr_time:0.3f}) Removing {model} from {self}")
    self.models_loaded.remove(model)
    return new_events
  def _addModel(self, curr_time, model):
    new_events = []
    print(f"({curr_time:0.3f}) Adding {model} to {self}")
    self.models_loaded.add(model)
    return new_events
  
  def startExecuting(self, curr_time):
    new_events = []
    if self.executing:
      return new_events
    if self.queue.empty():
      return new_events
    
    self.executing = True
    next_queue_item = self.queue.get()
    if isinstance(next_queue_item.item, self.simulation.Request):
      new_events.extend(next_queue_item.item.model.executeRequest(curr_time))
      next_queue_item.item.startExecution(curr_time)
    completion_event = events.WorkerQueueCompletionEvent(self.simulation, self, next_queue_item)
    new_events.append((curr_time + next_queue_item.getLatency(), completion_event))
    
    return new_events

@functools.total_ordering
class Model(common.ModelPlacements.Model):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
  def executeRequest(self, curr_time):
    new_events = []
    self.last_used = curr_time
    return new_events

class Simulation(object):

  class Metrics(object):
    def __init__(self, simulation):
      self.simulation = simulation
      self.general_metrics = {
              "requests_in" : 0,
              "requests_out" : 0,
      }
      self.per_model_requests = collections.defaultdict(int)
      self.per_model_responses = collections.defaultdict(int)
      self.per_model_latency = collections.defaultdict(list)
    def markRequestIn(self, model_name):
      self.general_metrics["requests_in"] += 1
      self.per_model_requests[model_name] += 1
    def markRequestOut(self, model_name, latency):
      self.general_metrics["requests_out"] += 1
      self.per_model_responses[model_name] += 1
      self.per_model_latency[model_name].append(latency)
    
    def reportMetrics(self):
      print(f"Requests: {self.general_metrics['requests_out']} / {self.general_metrics['requests_in']} completed")
      for model in sorted(self.per_model_latency.keys()):
        print(f"{model} : {np.average(self.per_model_latency[model]):0.3f} : {np.average(self.per_model_latency[model]) / self.simulation.models_by_name[model].load_latency:%}")

  class Request(object):
    class Status(enum.Enum):
      INIT = 1
      ACCEPTED = 2
      REJECTED = 3
      EXECUTING = 4
      COMPLETED = 5
    def __init__(self, simulation, request_id, arrival_time, model_requested, *args, **kwargs):
      self.simulation = simulation
      self.status = self.__class__.Status.INIT
      self.id = int(request_id)
      self.model_requested = model_requested
      self.model = self.simulation.models_by_name[model_requested]

      self.arrival_time = float(arrival_time)
      self.assignment_time = float('inf')
      self.execution_time = float('inf')
      self.completion_time = float('inf')
      
      self.model_miss = False
      self.is_saturated = False
      
    def __str__(self):
      return f"R({self.id}, {self.arrival_time}, {self.model_requested}, {self.status})"
    def markRejected(self):
      self.status = self.__class__.Status.REJECTED
    def markComplete(self, curr_time):
      self.completion_time = curr_time
      self.status = self.__class__.Status.COMPLETED
      self.simulation.metrics.markRequestOut(self.model_requested, (curr_time-self.arrival_time))
    
    def assignToWorker(self, curr_time, model_miss):
      self.assignment_time = curr_time
      self.model_miss = model_miss
    def startExecution(self, curr_time):
      self.execution_time = curr_time
    def getResponse(self):
      response_dict = {
        "request_id" : self.id,
        "model" : self.model_requested,
        "response" : f"{self.status}",
        "placement_delay" : self.assignment_time - self.arrival_time,
        "queue_delay" : self.execution_time - self.assignment_time,
        "execution_delay" : self.completion_time - self.execution_time,
        "overall_latency" : self.completion_time - self.arrival_time,
        "model_miss" : self.model_miss,
        "saturated" : self.is_saturated,
      }
      return json.dumps(response_dict)
      
      
    @classmethod
    def fromLine(cls, simulation, line):
      return cls(simulation, *(line.split()))
  
  def __init__(self, flags, models_to_be_requested, rng_seed=None, *args, **kwargs):
    self.flags = flags
    self.rng = np.random.default_rng(rng_seed)
    

    self.results_fid = gzip.open(os.path.join(flags.results_dir, f"{flags.run_identifier}.log.gz"), 'wt')
    self.cache_size = flags.max_concurrent_models

    self.is_saturated = False
    
    model_descriptions = common.getModelInfo(json_file=flags.model_description_file)
    time.sleep(10*random.random())
    if not os.path.exists(os.path.join(flags.results_dir, os.path.basename(flags.model_description_file))):
      shutil.copy(flags.model_description_file, flags.results_dir)
      shutil.copy(flags.workload_file, flags.results_dir)
    # Internally important data
    self.models_by_name = {
            model_name : Model(model_name, model_descriptions[model_name])
            for model_name in models_to_be_requested
    }
    self.workers_by_name = {
            worker_name : Worker(self, worker_name)
            for worker_name in [f"worker_{i:02d}" for i in range(flags.num_workers_to_add)]
    }
    self.model_placements = common.ModelPlacements()
    for model in self.models_by_name.values():
      self.model_placements.addModel(model)
    for worker in self.workers_by_name.values():
      self.model_placements.addWorker(worker)
    
    self.metrics = self.Metrics(self)
    
    # Components
    self.local_controller = LocalController(self)
    self.placement_controller = PlacementController(self, self.flags)
    
    # Event Queue
    self.event_queue = queue.PriorityQueue()
    
    
    # Setup some models in cache, because why not
    #for worker in sorted(self.workers_by_name.values()):
    #  for model in self.rng.choice(sorted(self.models_by_name.values()), size=self.cache_size, replace=False):
    #    self.model_placements.addModelToWorker(worker, model)
    #self.model_placements.sync()
    
  def run(self):
    logging.info("Starting simulation")
    while not self.event_queue.empty():
      curr_time, next_event = self.event_queue.get()
      logging.debug(f"NextEvent -> ({curr_time} : {next_event}")
      events_to_add = next_event.run(curr_time)
      for event_tuple in events_to_add:
        self.event_queue.put(event_tuple)
    logging.info("Simulation complete")
    self.metrics.reportMetrics()
    self.results_fid.close()

  def mark_as_saturated(self):
    self.is_saturated = True
  
  def recordExit(self, request):
    self.results_fid.write(f"{request.getResponse()}\n")

def getFlags():
  parser = argparse.ArgumentParser(
          parents=[
                  common.getParser(add_help=False),
                  placement_controller.getParser(add_help=False, include_parents=False)
          ],
          conflict_handler='resolve'
  )
  parser.add_argument("--cache_size", default=3)
  parser.add_argument('--workload_file', default="../workload/workload.txt")
  parser.add_argument('--model_description_file', default="../workload/models.json")
  
  parser.add_argument('--stop_after', default=float('inf'), type=float)
  

  parser.add_argument('--run_identifier', default=None,
                      help="Identifier for saving data logs")
  parser.add_argument('--results_dir', default="results/")

  parser.add_argument('--show_debug', action='store_true')
  
  parser.add_argument('--base_logging_dir', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs/simulation')) )
  parser.add_argument('--run_series', default=None)
  
  flags = parser.parse_args()
  if flags.run_identifier is None:
    flags.run_identifier = flags.model_eviction_algorithm
  flags.run_identifier = f"{flags.run_identifier}.{int(time.time())}"
  
  if flags.run_series is not None:
    flags.base_logging_dir = os.path.join(flags.base_logging_dir, flags.run_series)
  else:
    flags.base_logging_dir = os.path.join(flags.base_logging_dir, flags.run_identifier)
  flags.results_dir = flags.base_logging_dir
  
  if not os.path.exists(flags.results_dir):
    os.makedirs(flags.results_dir)
  
  return flags

def main():
  
  flags = getFlags()
  
  common.getLogger(hide_debug=(not flags.show_debug))

  with open(flags.workload_file) as workload_fid:
    models_to_be_requested = set([l.split(' ')[2].strip() for l in workload_fid.readlines()])
  

  simulation = Simulation(flags, models_to_be_requested, cache_size=flags.cache_size, rng_seed=flags.rng_seed)

  workload_fid = open(flags.workload_file)
  line = workload_fid.readline()
  first_request = simulation.Request.fromLine(simulation, line)
  
  simulation.event_queue.put( (first_request.arrival_time, events.RequestArrival(simulation, first_request, workload_fid)) )
  simulation.run()
  
  workload_fid.close()

if __name__ == '__main__':
  main()