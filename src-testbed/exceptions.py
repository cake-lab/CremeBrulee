#!env python



#####################################
## Exceptions for InferenceRequest ##
#####################################
class RequestTimeoutException(Exception):
  pass
#####################################


###########################
## Exceptions for Worker ##
###########################
class WorkerException(Exception):
  pass
  
class NotEnoughFreeSpace(WorkerException):
  pass
  
class ModelLoadException(WorkerException):
  pass
class ModelNotReadyException(ModelLoadException):
  pass
class ModelAlreadyLoadedException(ModelLoadException):
  pass
class ModelAlreadyStoppedException(ModelLoadException):
  pass
  
class ModelRepoException(WorkerException):
  pass
class ModelNotInRepoException(ModelRepoException):
  pass
  
class ModelInferenceException(WorkerException):
  pass

class ServerDeathException(WorkerException):
  pass
###########################


####################################
## Exceptions for LocalController ##
####################################
class LocalControllerException(Exception):
  pass

class InferenceFailedException(LocalControllerException):
  pass



class ModelNotFoundExeception(LocalControllerException):
  pass
class ModelFindingTimeoutException(LocalControllerException):
  pass
####################################

########################################
## Exceptions for PlacementController ##
########################################
class PlacementControllerException(Exception):
  pass
  
class ModelInformationMissingException(PlacementControllerException):
  pass

########################################