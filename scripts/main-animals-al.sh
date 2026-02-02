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

for folder in main-animals-al-bayesbinn main-animals-al-bimu main-animals-al-mesu main-animals-al-mesu-standardized main-animals-al-ste; do
    process_folders "$folder"
done

wait
mkdir -p results-main-animals-al
for folder in main-animals-al-bayesbinn main-animals-al-bimu main-animals-al-mesu main-animals-al-mesu-standardized main-animals-al-ste; do
    for file in results-"$folder"/*; do
        mv "$file" results-main-animals-al/
    done
done
rm -r results-main-animals-al-bayesbinn
rm -r results-main-animals-al-bimu
rm -r results-main-animals-al-mesu
rm -r results-main-animals-al-mesu-standardized
rm -r results-main-animals-al-ste
