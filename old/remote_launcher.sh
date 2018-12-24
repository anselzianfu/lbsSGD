#!/bin/bash
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -p|--plot)
    PLOT=YES
    shift
    ;;
    -c|--config)
    REMOTECONFIG="$2"
    shift # past argument
    shift # past value
    ;;
	-r|--remote)
    REMOTE=YES
    shift # past argument
    ;;
    -e|--env)
    REMOTEENV="$2"
    shift
    shift
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

IFS=: read -r IP LOC <<< "$1"

if [ ${REMOTE} == "YES" ]; then
   echo "Using remote config file: ${REMOTECONFIG}"
else
  echo "Using local config file: ${REMOTECONFIG}"
  scp ${REMOTECONFIG} $1  
fi
if [ -z "${REMOTEENV}" ]; then
   ssh ${IP} "cd ${LOC} && ./run.sh ${REMOTECONFIG}"
else
   ssh ${IP} "source ${REMOTEENV}/bin/activate && cd ${LOC} && ./run.sh ${REMOTECONFIG}"
fi

if [ ${PLOT} == "YES" ]; then
    echo "Copying over results"
    scp -r $1/plots .
fi
