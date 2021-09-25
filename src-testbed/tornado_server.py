#!env python

import tornado.httpclient
import tornado.ioloop
import tornado.web
import tornado.ioloop
import tornado.web
from tornado import options
import asyncio

import json
import argparse
import functools

import local_controller as LocalController

import os
import common
import tornado.log
import logging 
import signal


import exceptions

#import logging as logger
#logger = common.getLogger(f"{os.path.basename(__file__).replace('.py', '')}")


class MainHandler(tornado.web.RequestHandler):
  def get(self):
    self.write("Hello, world")
  
class InferHandler(tornado.web.RequestHandler):
  def get(self, model_name):
    self.write(f"Cannot complete request using GET.  Send something to infer, durr.. Something besides {model_name}")
    
  async def post(self, model_name):
    try:
      logger.info(f"self: {self}")
      self.set_header("Content-Type", "text/plain")
      
      id_num = self.get_argument('id', None)
      #data = json.loads(self.get_argument('data'))
      
      inference_request = common.InferenceRequest(model_name, data, id_num)
      response = await tornado.ioloop.IOLoop.current().run_in_executor(None, local_controller.infer, inference_request)
      print(f"Response: {response}")
      logger.info(f"response (tornado): {response}")
      self.finish(f"{response}")
    except exceptions.RequestTimeoutException as e:
      logger.error(f"InferenceRequest timed out: {e}")
      self.set_status(504)
      self.finish(f"InferenceRequest timed out: {e}")
    except exceptions.ModelNotFoundExeception as e:
      logger.error("Model not found on server")
      self.set_status(503)
      self.finish(f"Model not available: {e}")
    except exceptions.ModelFindingTimeoutException as e:
      logger.error("Timed out when finding model")
      self.set_status(504)
      self.finish(f"Model took too long to be placed: {e}")
    except exceptions.InferenceFailedException as e:
      logger.error("Inference Failed")
      self.set_status(500)
      self.finish(f"Inference failed: {e}")
    finally:
      logger.info("")


class POSTHandler(tornado.web.RequestHandler):
  def post(self,):
    for field_name, files in self.request.files.items():
      for info in files:
        filename, content_type = info["filename"], info["content_type"]
        body = info["body"]
        logging.info(
          'POST "%s" "%s" %d bytes', filename, content_type, len(body)
        )
        

    self.write("OK")

def make_app():
  return tornado.web.Application([
    (r"/", MainHandler),
    (r"/infer/([^/]+)", InferHandler)
  ])


def getParser(add_help=True, include_parents=True):
  parser = argparse.ArgumentParser(add_help=add_help,
    parents=([common.getParser(add_help=False)] if include_parents else [])  
  )
  
  return parser



if __name__ == "__main__":
  
  logger = logging.getLogger("tornado.application")
  args = getParser().parse_args()
  
  signal.signal(signal.SIGTERM, (lambda *_: sys.exit(0)))
  
  data = common.getData()
  
  logger.info("Setting up server")
  tornado.httpclient.AsyncHTTPClient.configure('tornado.curl_httpclient.CurlAsyncHTTPClient')
  local_controller = LocalController.LocalController(args.redis_server, args.redis_port)
  
  # Tornado configures logging.
  options.parse_command_line()
  app = make_app()
  app.listen(5000)
  #logger.setLevel(logger.DEBUG)
  tornado.ioloop.IOLoop.current().start()