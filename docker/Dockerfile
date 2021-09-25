
FROM python:3.8.5-buster AS localcontroller-base
WORKDIR /usr/src
RUN apt-get update
RUN apt-get install -y redis
COPY requirements.txt /usr/src/requirements.txt
RUN pip install -r requirements.txt

FROM localcontroller-base AS localcontroller-client
COPY ./ /usr/src/

FROM redis AS localcontroller-redis
#COPY redis.conf /usr/local/etc/redis/redis.conf
RUN echo "notify-keyspace-events KA" > /tmp/redis.conf
EXPOSE 6379/tcp
CMD [ "redis-server", "/tmp/redis.conf" ]

FROM localcontroller-base AS localcontroller-frontend
EXPOSE 5000
COPY ./ /usr/src/
CMD python tornado_server.py 2>&1 | tee /logs/local_controller.log 



FROM localcontroller-base AS localcontroller-metricsmanager
COPY ./ /usr/src/
CMD python metrics_manager.py 2>&1 | tee /logs/metrics_manager.log 


FROM localcontroller-base AS localcontroller-placementcontroller
COPY ./ /usr/src/
CMD python placement_controller.py --keep_alive --model_cooling --model_warming --disable_reactive --force_remove_periodic --model_eviction_algorithm least-outstanding 2>&1 | tee /logs/placement_controller.log 


FROM localcontroller-base AS localcontroller-worker
RUN curl -fsSL https://get.docker.com -o get-docker.sh
RUN sh get-docker.sh
#RUN apt-get install -y docker.io
#CMD python3.8 -m pip /opt/triton_assets/python/*.whl
COPY ./ /usr/src/
#CMD python worker.py --running_in_docker --dummy_load 2>&1 | tee /logs/$HOSTNAME.log

#CMD locust -f locustfile.py --headless -t 900s -u 10 --host http://frontend:5000  2>&1 | tee /logs/test.log 
