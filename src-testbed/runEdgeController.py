#!env python

import os
import docker
import logging
import argparse
import sys
import tarfile
import shutil
import time
import json

import pathlib

import numpy as np

import common


import worker
import client
import placement_controller
import tornado_server


#######################
## General Functions ##
#######################
def buildDockerImage(docker_client, dockerfile, target, tag=None, *args, **kwargs):
  logging.info(f"buildDockerImage(docker_client, {dockerfile}, {target}, {tag})")
  if tag is None:
    docker_img, logs = docker_client.images.build(path=".", dockerfile=dockerfile, target=target)
  else:
    docker_img, logs = docker_client.images.build(path=".", dockerfile=dockerfile, target=target, tag=tag)
  return docker_img
  
def startDockerContainer(docker_client, flags, image, name, command=None, volumes={}, env_vars={}, detach=True, *args, **kwargs):
  logging.info(f"startDockerContainer(docker_client, {image}, {name}, {command}")
  
  volumes[os.path.abspath(flags.base_logging_dir)] = {'bind': '/logs', 'mode': 'rw'}
  
  # Remove any existing containers with the same name
  logging.debug("Checking for an existing container with same name")
  logging.debug(f"Containers with same name: {docker_client.containers.list(all=True, filters={'name': name})}")
  for container in docker_client.containers.list(all=True, filters={'name': name}):
    stopDockerContainer(container)
  
  # Create a container with the same name
  logging.debug("Starting a container with image")
  docker_container = docker_client.containers.create(image, name=name, command=command, volumes=volumes, environment=env_vars, detach=detach)
  
  # Attach networks specified, creating if needed
  for network_name in flags.network_names:
    try:
      network = docker_client.networks.get(network_name)
    except docker.errors.NotFound:
      network = docker_client.networks.create(network_name)
    network.connect(docker_container, aliases=[name])
    
  docker_container.start()
  return docker_container

def buildAndRunContainer(docker_client, flags, target, tag, command=None, volumes={}, env_vars={}, skip_build=False):
  if not skip_build:
    docker_img = buildDockerImage(docker_client, flags.dockerfile, target=target, tag=tag)
  else:
    docker_img = docker_client.images.get(tag)
  container = startDockerContainer(docker_client, flags, docker_img, tag, command=command, volumes=volumes, env_vars=env_vars)
  logging.info(f"logs: {container.logs().decode()}")
  return container

def stopDockerContainer(docker_container, remove=True, *args, **kwargs):
  docker_container.stop()
  if remove:
    docker_container.remove()

#######################



#########################
## Container Functions ##
#########################

def startRedisContainer(docker_client, flags, *args, **kwargs):
  logging.info("Starting Redis")
  target = "localcontroller-redis"
  tag = "redis-server"
  
  return buildAndRunContainer(docker_client, flags, target, tag, *args, **kwargs)

def startPlacementController(docker_client, flags, *args, **kwargs):
  logging.info("Starting PlacementController")
  
  target = "localcontroller-placementcontroller"
  tag = "placementcontroller"
  
  command = "/bin/bash -c \""
  command += f"python placement_controller.py "
  if flags.do_proactive:
    command += f"--do_proactive "
  command += f"--model_eviction_algorithm {flags.model_eviction_algorithm} "
  command += f"--max_concurrent_models {flags.max_concurrent_models} "
  command += f"--scale_func {flags.scale_func} "
  command += f"--weight_func {flags.weight_func} "
  command += f"--cost_function {flags.cost_function} "
  command += f"--worker_memory {flags.worker_memory} "
  command += f"--workload_file /etc/workload/workload.txt "
  if flags.rng_seed is not None:
    command += f"--rng_seed {flags.rng_seed} "
  command += "2>&1 | tee /logs/placement_controller.log\""
  
  logging.info(f"command: {command}")
  
  volumes = {
    os.path.dirname(os.path.abspath(f"{flags.workload_file}")) : {'bind': '/etc/workload', 'mode': 'rw'},
  }

  return buildAndRunContainer(docker_client, flags, target, tag, command, volumes)
  
def startFrontEnd(docker_client, flags, *args, **kwargs):
  logging.info("Starting FrontEnd")
  
  target = "localcontroller-frontend"
  tag = "frontend"
  
  command = "/bin/bash -c \""
  command += "python tornado_server.py "
  command += "2>&1 | tee /logs/local_controller.log \""
  
  volumes = {}
  
  return buildAndRunContainer(docker_client, flags, target, tag, command, volumes)

def startWorker(docker_client, worker_name, flags, *args, **kwargs):
  logging.info(f"Starting Worker ({worker_name})")
  
  target = "localcontroller-worker"
  tag = f"{worker_name}"
  
  
  command = "/bin/bash -c \""
  command += "python worker.py --running_in_docker "
  command += f" --worker_memory {flags.worker_memory} "
  command += f" --real_model_repo_path {flags.real_model_repo_path} "
  command += f" --worker_name {worker_name} "
  command += f" --max_concurrent_execs {flags.max_concurrent_execs} "
  command += f" --model_description_file /etc/workload/models.json "
  if (not flags.update_redis) or flags.get_model_stats:
    command += " --update_redis "
  if flags.use_arm:
    command += " --use_arm "
  if flags.dummy_load:
    command += " --dummy_load "
  if flags.record_measurements:
    command += " --record_measurements "
  if flags.load_in_background:
    command += " --load_in_background "
    
  command += f" 2>&1 | tee /logs/{worker_name}.log\""
  
  volumes = {
    #os.path.abspath("../triton-inference-server/docs/examples/model_repository.limited/") : {'bind': '/models', 'mode': 'rw'},
    os.path.abspath(f"{flags.real_model_repo_path}") : {'bind': '/tmp/models', 'mode': 'rw'},
    os.path.abspath("/var/run/docker.sock") : {'bind': '/var/run/docker.sock', 'mode': 'rw'},
    os.path.dirname(os.path.abspath(f"{flags.model_description_file}")) : {'bind': '/etc/workload', 'mode': 'rw'},
  }
  
  env_vars = {
    'WORKER_NAME' : f"{worker_name}",
  }
  
  return buildAndRunContainer(docker_client, flags, target, tag, command, volumes, env_vars)

#########################


######################
## Action Functions ##
######################
def makeRedisCall(docker_client, flags, redis_cmd, *args, **kwargs):
  redis_img = docker_client.images.get("redis")
  command = f"redis-cli -h redis-server {redis_cmd}"
  
  target = "localcontroller-redis"
  tag = "redis-client"
  
  return buildAndRunContainer(docker_client, flags, target, tag, command, skip_build=False)

def examine_workload(docker_client, flags, workload_file):
  with open(f"{workload_file}") as fid:
    models_requested = list(set([l.strip().split(' ')[-1] for l in fid.readlines()]))
  return models_requested

def runTests(docker_client, flags, workload_file):
  logging.info(f"runTests(docker_client, {flags}, {workload_file})")
  target = "localcontroller-client"
  tag = "client"
  
  command = "/bin/bash -c \""
  command += f"python -u client.py "
  command += f"--workload_file /etc/workload/workload.txt "
  command += f"--identifier {flags.run_identifier} "
  if flags.restrict_requests:
    command += f"--restrict_requests "
  command += "\""
  
  results_dir = os.path.join(flags.base_logging_dir, f"results.{flags.run_identifier}")
  os.makedirs(results_dir, exist_ok=True)
  
  volumes = {
    os.path.dirname(os.path.abspath(f"{workload_file}")) : {'bind': '/etc/workload', 'mode': 'rw'},
    results_dir : {'bind': '/etc/results', 'mode': 'rw'},
  }
  
  logging.info("Testing running....")
  container = buildAndRunContainer(docker_client, flags, target, tag, command, volumes)
  for output in container.attach(stdout=True, stream=True):
    sys.stdout.write(f"{str(output.decode())}")
  logging.info("Tests complete.")


def runModelMeasurementContainer(docker_client, flags, models_to_measure, num_cycles):
  logging.info(f"runModelMeasurementContainer(docker_client, {flags}, {models_to_measure})")
  target = "localcontroller-client"
  tag = "client"
  
  command = "/bin/bash -c \""
  command += f"python -u measure_models.py "
  command += f"--models_to_measure {' '.join(models_to_measure)} "
  command += f"--num_cycles {num_cycles} "
  command += "2>&1 | tee /logs/measure_models.log\""
  
  volumes = {}
  
  logging.info("Testing running....")
  container = buildAndRunContainer(docker_client, flags, target, tag, command, volumes)
  for output in container.attach(stdout=True, stream=True):
    sys.stdout.write(f"{str(output.decode())}")
  logging.info("Tests complete.")

def preserveLogs(flags, base_logging_dir): #logs_dir="logs/", records_dir="records/"):
  records_dir = os.path.join(base_logging_dir, "records")
  #if not os.path.exists(records_dir):
  #  os.makedirs(records_dir)
  #with tarfile.open(os.path.join(records_dir, f"logs.{flags.run_identifier}.tgz"), "w:gz") as tar:
  #  tar.add(base_logging_dir, arcname=f"{flags.run_identifier}")
  
######################


#######################
## Central Functions ##
#######################

def setupEdgeController(docker_client, flags, models_to_add, workers_to_add, *args, **kwargs):
  
  running_containers = []
  # Start redis container
  running_containers.append(startRedisContainer(docker_client, flags))
  
  # Start PlacementController
  running_containers.append(startPlacementController(docker_client, flags))
  
  # Start Front End
  running_containers.append(startFrontEnd(docker_client, flags))
  
  
  # Start Workers
  for worker_name in workers_to_add:
    running_containers.append(startWorker(docker_client, worker_name, flags))
    makeRedisCall(docker_client, flags, f"sadd workers {worker_name}:50051")
  
  makeRedisCall(docker_client, flags, f"sadd models {' '.join(models_to_add)}")
  
  time.sleep(5)
  
  # Add all models, so the cache is already full
  for model in np.random.permutation((models_to_add)):
    makeRedisCall(docker_client, flags, f"sadd {common.PLACEMENT_REQUEST_KEY} {model}")
    time.sleep(7)
    #self.db.sadd(f"{common.PLACEMENT_REQUEST_KEY}", f"{inference_request.model_name}")
  #time.sleep(180) ## all models have a total load time of ~187s
  logging.info("all models have been cycled!")
  return running_containers

def stopContainers(running_containers, *args, **kwargs):
  for container in running_containers[::-1]:
    stopDockerContainer(container)


def runModelMeasurements(docker_client, flags, models_to_test, num_cycles):
  
  
  running_containers = []
  # Start redis container
  #running_containers.append(startRedisContainer(docker_client, flags))
  redis_container = startRedisContainer(docker_client, flags)
  
  overall_stats = []
  for model_name in models_to_test:
    for worker_name in ["worker_0"]:
      worker_container = startWorker(docker_client, worker_name, flags)
      makeRedisCall(docker_client, flags, f"sadd workers {worker_name}:50051")
    
    makeRedisCall(docker_client, flags, f"sadd models {' '.join([model_name])}")
    time.sleep(1)
    
    runModelMeasurementContainer(docker_client, flags, [model_name], num_cycles)
    stopContainers([worker_container])
    
    makeRedisCall(docker_client, flags, f"srem models {' '.join([model_name])}")
    overall_stats.extend(reportModelStats(docker_client, flags, [model_name]))
    
    for container in  docker_client.containers.list(all=True):
      if container != redis_container:
        stopDockerContainer(container, remove=True)
    
    api_client = docker.APIClient("unix:///var/run/docker.sock")
    api_client.prune_containers()
    api_client.prune_images()
    api_client.prune_networks()
  
  stopContainers([redis_container])
  print(json.dumps(overall_stats, indent=2, sort_keys=True))
  
  return []


def reportModelStats(docker_client, flags, models_to_report):
  
  fields_to_report = ["avg_exec_latency", "avg_load_latency", "avg_unload_latency", "loaded_size"]
  keys = []
  for model_name in models_to_report:
    for stat_name in fields_to_report:
      #keys.append( (model_name, stat_name) )
      keys.append(f"{common.MODEL_STAT_PREFIX}{model_name}{common.DB_FIELD_SEPARATOR}{stat_name}")
  results = makeRedisCall(docker_client, flags, f"mget {' '.join(keys)}")
  values = results.logs().decode().split('\n')
  
  stats = []
  for i, model_name in enumerate(models_to_report):
    model_stats = {
      field : float(values[ i*len(fields_to_report) + j])
      for j, field in enumerate(fields_to_report)
    }
    model_stats["name"] = model_name
    stats.append(model_stats)
  return stats
#######################



def parseFlags():
  parser = argparse.ArgumentParser(
      parents=[
        common.getParser(add_help=False),
        worker.getParser(add_help=False, include_parents=False),
        client.getParser(add_help=False, include_parents=False),
        placement_controller.getParser(add_help=False, include_parents=False),
        tornado_server.getParser(add_help=False, include_parents=False),
      ], 
      conflict_handler='resolve'
    )
  
  # General settings
  parser.add_argument('--path_to_dockerfile', default="../docker/Dockerfile", dest="dockerfile",
                      help='Path to directory containing dockerfile to use for builds')
  parser.add_argument('--network_names', nargs='+', default=["local_controller_network"],
                      help="Name of networks to add all devices to")
  parser.add_argument('--skip_cleanup', action="store_true",
                      help="Skip removing docker containers")
  
  parser.add_argument("--start_only", nargs="+", choices=["worker", "placement_controller", "local_controller", "redis"], default=None,
                      help="Option to start only a subset of docker containers in order to increase testing rapidity.")
  
  parser.add_argument("--prune", action="store_true")
  parser.add_argument("--show_debug", action="store_true")
  
  parser.add_argument('--get_model_stats', action='store_true')
  
  
  parser.add_argument('--base_logging_dir', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs/test_bed')) )
  parser.add_argument('--run_series', default=None)
  parser.add_argument('--run_identifier', default=None, help="Identifier for saving data logs")
  
  parser.add_argument('--logs_dir', default="logs/", help="Directory for storing log files")
  parser.add_argument('--results_dir', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '../results')))
  
  parser.add_argument('--model_dir', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/models')))
  
  parser.add_argument('--use_azure', action="store_true", help="If true then parse Aure traces")
  parser.add_argument('--workload_file', default="../workload/workload.txt")
  
  flags = parser.parse_args()
  
  if flags.run_identifier is None:
    flags.run_identifier = flags.model_eviction_algorithm
  flags.run_identifier = f"{flags.run_identifier}.{int(time.time())}"
  
  if flags.run_series is not None:
    flags.base_logging_dir = os.path.join(flags.base_logging_dir, flags.run_series)
  else:
    flags.base_logging_dir = os.path.join(flags.base_logging_dir, flags.run_identifier)
  if not os.path.exists(flags.base_logging_dir):
    os.makedirs(flags.base_logging_dir, exist_ok=True)
  
  if not os.path.isabs(flags.real_model_repo_path):
    flags.real_model_repo_path = os.path.abspath(flags.real_model_repo_path)
  
  return flags


def main():
  flags = parseFlags()
  common.getLogger(f"{os.path.basename(__file__).replace('.py', '')}", hide_debug=(not flags.show_debug))
  
  logging.debug(f"flags: {flags}")
  
  np.random.seed(flags.rng_seed)
  
  try:
    docker_client = docker.from_env()
  except FileNotFoundError as e:
    logging.critical("Cannot find docker file.  Is it running?")
    logging.critical(e)
  
  if not flags.get_model_stats: # (sso) TODO
    models_to_add = examine_workload(docker_client, flags, flags.workload_file)
    print(flags.workload_file)
    print(models_to_add)
    running_containers = setupEdgeController(docker_client, flags, models_to_add, [f"worker_{i}" for i in range(flags.num_workers_to_add)])
    
    runTests(docker_client, flags, flags.workload_file)
    
    #reportModelStats(docker_client, flags, models_to_add)
  
    if not flags.skip_cleanup:
      stopContainers(running_containers)
    
  else:
    models_to_add = [
      d for d in os.listdir(flags.model_dir) if os.path.isdir(os.path.join(flags.model_dir, d))
    ]
    logging.info(f"models_to_add: {models_to_add}")
    runModelMeasurements(docker_client, flags, models_to_test=sorted(models_to_add), num_cycles=50)
  
  
  if flags.prune:
    api_client = docker.APIClient("unix:///var/run/docker.sock")
    api_client.prune_containers()
    api_client.prune_images()
    api_client.prune_networks()
  
  preserveLogs(flags, flags.base_logging_dir)


if __name__ == '__main__':
  main()
