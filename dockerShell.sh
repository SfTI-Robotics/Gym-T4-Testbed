#!/bin/bash
docker build --tag=p4p .

docker run -v $(pwd):/home -it p4p bash
