#!env bash


/Users/ssogden/research/2020-project-EdgeController/dev-env/bin/python -m grpc_tools.protoc -I. --python_out=.. --grpc_python_out=.. $1
