#!/bin/bash
file="list_of_tests.txt"
# Read the textfile which holds the arguments and then parse those arguments 
# into the run file for an algorithm


#it reads the file per line, IFS leaves in the whitespacing
# [[ -n "$Line" ]] -> this prevents the last line from being ignored"


#I set the IFS to tab spaces that splits based on tabs per line
while IFS=$'\t' read -r line ;
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

		python3 run_main.py -env ${words[0]} -alg ${words[1]} -eps ${words[2]}
		kill -9 $(jobs -p)
		#i+=1
		#echo "i = $i"


# checking if the entire file has been read
done < "$file"



#Run the bash file ./NAME_OF_BASH_FILE
#if you cant run this just go chmod +x the_file_name
