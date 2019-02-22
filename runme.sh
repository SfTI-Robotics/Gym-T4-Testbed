#!/bin/bash

# Read the textfile which holds the arguments and then parse those arguments 
# into the run file for an algorithm
file="testbed.txt"

# it reads the file per line, IFS leaves in the whitespacing
# [[ -n "$Line" ]] -> this prevents the last line from being ignored"
# I set the IFS to tab spaces that splits based on tabs per line
while IFS=$'\t' read -r line ;
	do 
		# get input arguments
		words=($line)

		# print out the environment
		echo " "
		echo 'Inputting argument parameters'
		echo "Now Running ${words[0]} environment"
		
		# Running the python script
		python3 run_main.py -env ${words[0]} -alg ${words[1]} -eps ${words[2]}

		# close all programs to avoid memory leak
		kill -9 $(jobs -p)
	
# checking if the entire file has been read
done < "$file"

#Run the bash file ./NAME_OF_BASH_FILE
#if you cant run this just go chmod +x the_file_name