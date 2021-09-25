#!env python

import logging
import itertools
import numpy as np
import collections

class State(object):
  model_info = None
  cache_size = 0
  def __init__(self, state):
    self.state = sorted(state)
  
  def getPossibleEvictions(self, model_requested):
    if model_requested in self.state:
      return []
    
    size_of_new_model = self.model_info[model_requested]["loaded_size"]
    size_of_models_in_cache = sum([self.model_info[m]["loaded_size"] for m in self.state])
    space_to_free = abs((self.cache_size - size_of_models_in_cache) - size_of_new_model) 
    # abs((cache_size - memory_used) - size_to_add)
    #abs(self.cache_size - (size_of_new_model + size_of_models_in_cache))
    
    logging.debug(f"self.cache_size: {self.cache_size}")
    logging.debug(f"size_of_new_model: {size_of_new_model}")
    logging.debug(f"size_of_models_in_cache: {size_of_models_in_cache}")
    logging.debug(f"models_in_cache: {self.state}")
    logging.debug(f"space_to_free: {space_to_free}")
    
    
    # Start with every model that is, by itself, big enough to produce enough space
    complete_lists = [m for m in self.state if self.model_info[m]["loaded_size"] <= space_to_free]
    
    models_to_consider = [m for m in self.state if self.model_info[m]["loaded_size"] <= space_to_free]
    logging.info("Calling itertools...")
    all_combos = [list(t) for t in itertools.chain.from_iterable([itertools.combinations(models_to_consider, i) for i in range(1+len(models_to_consider))])]
    logging.debug(f"all_combos: {all_combos}")
    
    possible_evictions = list(filter(
        (lambda m_list: sum([self.model_info[m]["loaded_size"] for m in m_list]) >= space_to_free ),
        all_combos
      ))
    logging.debug(f"possible_evictions: {possible_evictions}")
    logging.info("Through itertools")
    ## We should remove sets that are superfluous, but that is a lot of work to figure out right now
    
    return list(filter((lambda e: len(e)>0), possible_evictions))
  
  @classmethod
  def setModelInfo(cls, model_info):
    cls.model_info = model_info
  @classmethod
  def setCacheSize(cls, cache_size):
    logging.debug(f"Setting cache size to {cache_size}")
    cls.cache_size = cache_size



    
def getBeladyBoundary(model, future_requests):
  logging.debug(f"getBeladyBoundary({model}, {future_requests[:10]})")
  if model not in future_requests:
    return float('inf')  
  return future_requests.index(model)

def evictionOracle(state, future_requests, value_func):
  possible_evictions = state.getPossibleEvictions(future_requests[0])
  
  return min(possible_evictions, key=(lambda m_list: value_belady(m_list, future_requests)))

def value_belady(eviction_list, future_requests):
  return min([getBeladyBoundary([m], future_requests) for m in eviction_list])







def getIdealCaches(cache_size, requests, model_info, value_func=(lambda vals, *args, **kwargs: np.sum(vals))):
  logging.info(f"getIdealCaches({cache_size}, {requests[:10]}, {model_info})")
  initial_cache = list(collections.OrderedDict.fromkeys(requests))[:cache_size]
  
  costs = { i : collections.defaultdict(lambda: float('+inf')) for i in range(len(requests)+1) }
  costs[0] = { tuple(sorted(initial_cache)) : 0}
  
  # Walk out through the requests, calculating the cost at the next step
  for i in range(len(requests)):
    potential_caches = costs[i]
    for step_cache, step_cost in potential_caches.items():
      requested_model = requests[i]
      
      potential_steps = []
      # Check if model is already in cache
      if requested_model in step_cache:
        # If it is then our cost is nothing added
        key = tuple(sorted(step_cache))
        cost = step_cost
        potential_steps.append( (key, cost) )
      else:
        
        # If it isn't then we need to generate alternative options
        for j, model_to_remove in enumerate(step_cache):
          potential_cache = tuple(sorted([ step_cache[k] if k!=j else requested_model for k in range(len(step_cache)) ]))
          if model_to_remove in requests[i:]:
            potential_cache_cost = step_cost + model_info[model_to_remove]["load_latency"]
          else:
            potential_cache_cost = step_cost
          potential_steps.append( (potential_cache, potential_cache_cost) )
          
      if (i+1) in costs:
        for (key, cost) in potential_steps:
          # Because we're using a default dict with infinite, we can just build down from there
            costs[i+1][key] = min( [costs[i+1][key], cost] )
    logging.debug(f"potential caches ({i}): {list(zip(costs[i].items()))}")
  
  costs = dict(costs)
  
  end_states = costs[len(requests)]
  
  curr_cache = min( end_states, key=end_states.get)
  logging.debug(f"expected cost: {end_states[curr_cache]}")
  
  cache_order = [curr_cache]
  for i in range(len(requests))[::-1]:
    logging.debug(f"{i} : {curr_cache}")
    
    # Get all potential caches that change by at most one model
    potential_caches = [c for c in costs[i] if (len(set(curr_cache) - set(c)) <= 1) and (costs[i+1][curr_cache] >= costs[i][c])] 
    
    # Pick the cache with the lowest step cost
    next_cache = max(potential_caches, key=(lambda c: costs[i][c]))
    logging.debug(f"next_cache: {next_cache} ({costs[i][next_cache]})")
    curr_cache = next_cache
    cache_order.append(tuple(list(curr_cache)))
    logging.debug(f"cache_order: {cache_order}")
  cache_order = cache_order[::-1]
  
  for i, cache in enumerate(cache_order):
    logging.debug(f"{i} : {cache} : {costs[i][cache]}")
  
  return cache_order
  






if __name__ == '__main__':
  
  
  import common
  common.getLogger()
  
  model_info = {
    "model_0" : {"loaded_size" : 1, "load_latency" : 1.0},
    "model_1" : {"loaded_size" : 2, "load_latency" : 2.0},
    "model_2" : {"loaded_size" : 3, "load_latency" : 3.0},
    "model_3" : {"loaded_size" : 4, "load_latency" : 4.0},
    
    "model_5" : {"loaded_size" : 4, "load_latency" : 4.0},
  }
  
  names_of_future_requested_models = [
    "model_0",
    "model_1",
    "model_0",
    "model_3",
    "model_1",
    "model_2",
    "model_3",
    "model_2",
    "model_1",
    "model_2",
    "model_3",
    "model_0",
    "model_3",
    "model_1",
    "model_2",
  ]
  start_state = ["model_2", "model_3"]
  
  State.setCacheSize(5)
  State.setModelInfo(model_info)
  
  possible_states = collections.defaultdict(list)
  
  start_state = State(start_state)
  
  pickModelForEviction(start_state, names_of_future_requested_models)
  
  exit()
  
  possible_states[0] = [start_state]
  for i in range(0, len(names_of_future_requested_models)):
    possible_states[i+1].extend( itertools.chain.from_iterable([s.getPossibleNextStates() for s in possible_states[i]]) )
  
  #print([s.getValue() for s in possible_states[i+1]])
  
  for s in possible_states[i+1]:
    logging.info(f"{s}")
    #logging.info(f"{[str(s) for s in s.getPath()]}\n")
  