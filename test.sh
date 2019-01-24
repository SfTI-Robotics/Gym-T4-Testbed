#!/bin/bash

# Read the textfile which holds the arguments and then parse those arguments 
# into the run file for an algorithm


#it reads the file per line, IFS leaves in the whitespacing
# [[ -n "$Line" ]] -> this prevents the last line from being ignored"


#I set the IFS to tab spaces that splits based on tabs per line
while IFS=$'\t' read -r line || [[ -n "$line" ]];

#loop
do 
	#Echo is used to print so , printing the line
	words=($line)

	#print the array of words
	
	echo "Now Running ${words[0]} environment"
	
	sleep 1s

	#echo "${words[0]}"
	#echo "${words[1]}"
	#echo "${words[2]}"

	echo "Inputting argument parameters"

	sleep 1s


	#Running the python script
	#hard code the link 
	python3 copy_v0_easy.py ${words[0]} ${words[1]} ${words[2]}


# checking if the entire file has been readb
done < "$1"

#Run the bash file, Name of the Bash file with the run command
#if you cant run this just go chmod +x the_file_name
