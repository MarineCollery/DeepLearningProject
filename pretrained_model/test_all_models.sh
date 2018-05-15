#!/bin/sh

red=`tput setaf 1`
green=`tput setaf 2`
blue=`tput setaf 4`
reset=`tput sgr0`


if [ $# -ne 2 ] 
   then echo "${red}Enter the path of the directory with the TEST dataset folder ${reset}"
   		read path_dataset 
   		echo "${red}Enter the path of the folder with the TMP directories ${reset}"
   		read path_tmp
   		# echo "${red}Enter the path of the directory for the output temporary .wav files${reset}"
   		# read path_output 
   		# echo "${red}Enter the path of the directory for the output .csv files with the extracted features${reset}"
   		# read path_output_csv 
else
    path_dataset=$1 
    echo ${path_dataset};
    path_tmp=$2 
    echo ${path_tmp};
    # path_output=$3 
    # echo ${path_output};
    # path_output_csv=$4 
    # echo ${path_output_csv};
fi

# ls ${path_tmp}/ > dir.txt

while read tmp ; do
  bash test_model.sh ${path_dataset} ${path_tmp}/${tmp}
done < dir.txt

