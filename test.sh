#!/bin/bash

for FILE1 in "$@"
do
wc $FILE1
done