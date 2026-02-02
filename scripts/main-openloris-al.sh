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
        python main.py -c "$without_folder_name" -it 5 -fits -pca
        wait
        mv results/* "results-$folder"/
    done
    wait
}

for folder in main-openloris-al-bayesbinn-8192 main-openloris-al-bimu-8192 main-openloris-al-mesu-8192 main-openloris-al-mesu-standardized-8192 main-openloris-al-ste-8192; do
    process_folders "$folder"
done

wait
mkdir -p results-main-openloris-al
for folder in main-openloris-al-bayesbinn-8192 main-openloris-al-bimu-8192 main-openloris-al-mesu-8192 main-openloris-al-mesu-standardized-8192 main-openloris-al-ste-8192; do
    for file in results-"$folder"/*; do
        mv "$file" results-main-openloris-al/
    done
done
rm -r results-main-openloris-al-bayesbinn-8192
rm -r results-main-openloris-al-bimu-8192
rm -r results-main-openloris-al-mesu-8192
rm -r results-main-openloris-al-mesu-standardized-8192
rm -r results-main-openloris-al-ste-8192
