# SPDX-License-Identifier: CC-BY-4.0
# Code for "Active Continual Learning with Metaplastic Binary Bayesian Neural Networks"
# Kellian Cottart, Théo Ballet, Djohan Bonnet, Damien Querlioz
# Portions of the code are adapted from the Pytorch project (BSD-3-Clause)
# Author: Kellian Cottart <kellian.cottart@gmail.com>
# Date: 2025-30-01

process_folders() {
    local folder=$1
    local extra_args=$2
    mkdir -p "results-$folder"
    for file in configurations/"$folder"/*.json; do
        without_json=${file%.json}
        without_folder_name=${without_json#configurations/}
        python main.py -c "$without_folder_name" -it 5 -fits -ood openloris
        wait
        mv results/* "results-$folder"/
    done
    wait
}

for folder in main-openloris-1024 main-openloris-8192 main-openloris-25088; do
    process_folders "$folder"
done

wait

# save results as a single folder and unpack the big folders
mkdir -p results-main-openloris
for folder in main-openloris-1024 main-openloris-8192 main-openloris-25088; do
    for file in results-"$folder"/*; do
        mv "$file" results-main-openloris/
    done
done

rm -r results-main-openloris-1024
rm -r results-main-openloris-8192
rm -r results-main-openloris-25088
