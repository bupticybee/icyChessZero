PY_ENV="python3"
GPU_CORE=0
THREADS=20
SERVER="http://10.109.247.219:10088"
parsearg () {
  while getopts "p:g:t:s:" optname
    do
      case "$optname" in
        "p")
          PY_ENV=${OPTARG} 
          ;;
        "g")
          GPU_CORE=${OPTARG} 
          ;;
        "t")
          THREADS=${OPTARG}
          ;;
        "s")
          SERVER=${OPTARG}
          ;;
        "?")
          echo "Unknown option $OPTARG"
          ;;
        ":")
          echo "No argument value for option $OPTARG"
          ;;
        *)
        # Should not occur
          echo "Unknown error while processing options"
          ;;
      esac
    done
  return $OPTIND
}
 
parsearg "$@"
echo "-----------------------------------"
echo "python env: "${PY_ENV}
echo "gpu core: "${GPU_CORE}
echo "threads number:"${THREADS}
echo "server :"${SERVER}
echo "-----------------------------------"
echo "Is the setting ok? [y/N]"

#read is_okay
#if [ "${is_okay}" = "y" ]
#then 
#	echo "okay"
#else
#	echo "exiting..."
#    exit
#fi

echo "executing..."

date 
for i in `seq 1 ${THREADS}`
do
{
	#/usr/local/bin/python3 self_play.py	
    echo "${PY_ENV} -g=${GPU_CORE}"
	${PY_ENV} validate.py -g ${GPU_CORE} -s ${SERVER}
} &
done
wait

