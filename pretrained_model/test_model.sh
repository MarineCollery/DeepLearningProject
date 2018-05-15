#!/bin/sh

red=`tput setaf 1`
green=`tput setaf 2`
blue=`tput setaf 4`
reset=`tput sgr0`


if [ $# -ne 2 ] 
   then echo "${red}Enter the path of the directory with the TEST dataset folder ${reset}"
   		read path_dataset 
   		echo "${red}Enter the path of the TMP directory for of the retrained model${reset}"
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


find ${path_dataset}/dataset_smile_test/smiling -type f > temp_smile.txt
# find {path_dataset}/dataset_smile_test/not_smiling -type f > temp_not_smile.txt

# echo "/Volumes/Marine_WD_Elements/Dataset_DL_testing/dataset_smile_test/smiling/0001-image20056.jpg" > temp_smile.txt

limit=0.5
add=1

true_smiling=0
false_smiling=0 #one smiling was predicted as not smiling
true_not_smiling=0
false_not_smiling=0 #one not smiling was predicted as smiling

while read name_file ; do

	result=`/Users/Marine/anaconda2/envs/tensorflow/bin/python label_image.py \
	--graph=${path_tmp}/output_graph.pb \
	--labels=${path_tmp}/output_labels.txt \
	--input_layer=Placeholder \
	--output_layer=final_result \
	--input_height=331 --input_width=331 \
	--image=${name_file}`


	not_smiling_prob=$(echo ${result} | grep 'not smiling' | cut -d ' ' -f3 | bc)

	if (( $(echo "($not_smiling_prob) < $limit" | bc -l) )); then 
		true_smiling=$(echo "$true_smiling+$add" | bc )
	else
		false_smiling=$(echo "$false_smiling+$add" | bc )
	fi

	echo "true_smiling : ${true_smiling}"
	echo "false_smiling : ${false_smiling}"

done < temp_smile.txt


while read name_file ; do

	result=`/Users/Marine/anaconda2/envs/tensorflow/bin/python label_image.py \
	--graph=${path_tmp}/output_graph.pb \
	--labels=${path_tmp}/output_labels.txt \
	--input_layer=Placeholder \
	--output_layer=final_result \
	--input_height=331 --input_width=331 \
	--image=${name_file}`


	not_smiling_prob=$(echo ${result} | grep 'not smiling' | cut -d ' ' -f3 | bc)

	if (( $(echo "($not_smiling_prob) < $limit" | bc -l) )); then 
		false_not_smiling=$(echo "$false_not_smiling+$add" | bc )
	else
		true_not_smiling=$(echo "$true_not_smiling+$add" | bc )
	fi

	echo "true_not_smiling : ${true_not_smiling}"
	echo "false_not_smiling : ${false_not_smiling}"

done < temp_not_smile.txt

echo "${red}FINAL RESULT${reset}"
echo "${red}true_smiling : ${true_smiling}${reset}"
echo "${red}false_smiling : ${false_smiling}${reset}"
echo "${red}true_not_smiling : ${true_not_smiling}${reset}"
echo "${red}false_not_smiling : ${false_not_smiling}${reset}"

echo "${red}FINAL RESULT${reset}" > ${path_tmp}/testing_results.txt
echo "${red}true_smiling : ${true_smiling}${reset}">> ${path_tmp}/testing_results.txt
echo "${red}false_smiling : ${false_smiling}${reset}">> ${path_tmp}/testing_results.txt
echo "${red}true_not_smiling : ${true_not_smiling}${reset}">> ${path_tmp}/testing_results.txt
echo "${red}false_not_smiling : ${false_not_smiling}${reset}" >> ${path_tmp}/testing_results.txt
