#!/bin/bash
#clear

echo -e "\n############################################################"
echo -e   "###### run Simulated Annealing to optimize TTN layout ######"
echo -e   "############################################################"

##########################################################################

echo -e "\n -- Setting the parameters that stay unchanged -- \n"

WORK_PATH="./"
TODAY="2023-10-30"
DATA_PATH="data/"$TODAY"_TTN_SA/"

# Optimization method and options
NRUNS=50
NITERS=1000

# Loop on (L,m,s) values.
declare -a ARRAY_OF_PARS=(16,3,77777 24,3,17489 16,10,38564 24,10,81217)

##########################################################################

echo -e "\n -- Create the output folder if not present -- \n"

if [ ! -d $DATA_PATH ]; then
    mkdir $DATA_PATH
fi

##########################################################################

for i in "${ARRAY_OF_PARS[@]}"
do
    IFS=","; set -- $i; echo L=$1 , m=$2 , seed=$3
    rootname="job_ttn_L"$1"_D2_m"$2"_s"$3
    log_filename=$DATA_PATH$rootname".log"
    job_file=$DATA_PATH$rootname".slurm"
    job_content=\
"#!/bin/bash"$'\n\n'\
"#SBATCH -o "$DATA_PATH$rootname".o%j"$'\n'\
"#SBATCH -e "$DATA_PATH$rootname".err%j"$'\n'\
"#SBATCH -D ./"$'\n'\
"#SBATCH --get-user-env"$'\n'\
"#SBATCH --partition=spr"$'\n'\
"#SBATCH --time=23:59:00"$'\n'\
"#SBATCH --ntasks=1"$'\n'\
"#SBATCH --ntasks-per-node=1"$'\n'\
"#SBATCH -J "$rootname$'\n'\
"#SBATCH --mail-user=gian.giacomo.guerreschi@intel.com"$'\n'\
"#SBATCH --mail-type=begin --mail-type=end"$'\n\n'\
"source ~/anaconda3/bin/activate pando-env"$'\n'\
"python "$WORK_PATH"ttn_perform_SA.py -L "$1" -m "$2" -s "$3" -r "$NRUNS" -i "$NITERS" --today "$TODAY
    echo "$job_content"  > $job_file
    sbatch $job_file
    echo
done

exit 1

##########################################################################
