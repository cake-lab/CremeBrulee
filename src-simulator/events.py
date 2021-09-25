#!env python

import simulation
import logging
import itertools
import functools

@functools.total_ordering
class Event(object):
  """
  Taken fairly directly from https://github.com/epfl-labos/eagle/
  """
  _ids = itertools.count(0)
  def __init__(self):
    self.id = next(self._ids)
    
  def run(self, current_time):
    raise NotImplementedError("The run() method must be implemented by each class subclassing Event")
  
  def __lt__(self, other):
    return self.id < other.id
  def __eq__(self, other):
    return self.id == other.id


class RequestArrival(Event):
  def __init__(self, simulation, request, workload_fid):
    super().__init__()
    self.simulation = simulation
    self.request = request
    self.workload_fid = workload_fid
  
  def run(self, curr_time):
    new_events = []
    
    line = self.workload_fid.readline()
    if line != '':
      new_request = self.simulation.Request.fromLine(self.simulation, line)
      if new_request.arrival_time < self.simulation.flags.stop_after:
        new_arrival = RequestArrival(self.simulation, new_request, self.workload_fid)
        new_events.append( (new_request.arrival_time, new_arrival) )
    

      self.simulation.metrics.markRequestIn(self.request.model_requested)
      self.request.is_saturated = self.simulation.is_saturated
    
    # 1. Ask local_controller where this request should be assigned, which may result in an event to decide what happens next
      # Does it, actually?  Are we doing realistic in-sim delays, or just delays?
      # Whatever, let's just get a flow going by asking the local controller and go from there
    
    new_events.extend(self.simulation.local_controller.requestInference(curr_time, self.request))
    
    return new_events

class WorkerQueueCompletionEvent(Event):
  def __init__(self, simulation, worker, queue_item):
    super().__init__()
    self.simulation = simulation
    self.worker = worker
    self.queue_item = queue_item
  def run(self, curr_time):
    new_events = []
    logging.debug(f"Event type: {self.queue_item.item.__class__}")
    if isinstance(self.queue_item.item, self.simulation.Request):
      request = self.queue_item.item
      new_events.append( (curr_time, RequestCompletionEvent(self.simulation, request)) )
    elif isinstance(self.queue_item.item, ModelChangeEvent):
      new_events.append( (curr_time, self.queue_item.item) )
    else:
      logging.warn(f"Cannot recognize event type: {self.queue_item.item.__class__}")
    
    if not self.worker.queue.empty():
      next_queue_item = self.worker.queue.get()
      if isinstance(next_queue_item.item, self.simulation.Request):
        new_events.extend(next_queue_item.item.model.executeRequest(curr_time))
        next_queue_item.item.startExecution(curr_time)
      new_events.append( (curr_time + next_queue_item.getLatency(), WorkerQueueCompletionEvent(self.simulation, self.worker, next_queue_item)) )
    else:
      self.worker.executing = False
    logging.debug(f"new events: {new_events}")
    return new_events

class RequestCompletionEvent(Event):
  def __init__(self, simulation, request):
    super().__init__()
    self.simulation = simulation
    self.request = request
    logging.debug(f"Request {request} will be complete")
  def run(self, curr_time):
    new_events = []

    self.request.markComplete(curr_time)
    print(f"({curr_time:0.3f}) Request {self.request.id} for {self.request.model_requested} completed in {curr_time - self.request.arrival_time:0.3f} with status {self.request.status}")
    #print(self.request.getResponse())
    self.simulation.recordExit(self.request)
    return new_events

class ModelChangeEvent(Event):
  def __init__(self, simulation, worker, model):
    super().__init__()
    self.simulation = simulation
    self.worker = worker
    self.model = model
class ModelRemovalEvent(ModelChangeEvent):
  def run(self, curr_time):
    new_events = []
    new_events.extend(self.worker._removeModel(curr_time, self.model))
    return new_events
class ModelAdditionEvent(ModelChangeEvent):
  def run(self, curr_time):
    new_events = []
    new_events.extend(self.worker._addModel(curr_time, self.model))
    return new_events
  