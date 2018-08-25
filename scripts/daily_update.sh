cd /work/icybee/icyChessZero/scripts

# if 10,000 gameplays has been generated
count=`ls ../data/distributed | wc -w`
echo ${count}
if [ $count -le 10000 ];then
    echo 'exiting'
    exit
fi 

sh kill_all.sh
sleep 10s
sh multithread_start.sh -p /usr/local/bin/python3  -g 1 -t 20 -n no
sleep 10s
/usr/local/bin/python3 model_update.py -g 0 > update_log.txt
sleep 10s
sh validate.sh -p /usr/local/bin/python3 -g 0 -t 20 > validate_log.txt
sleep 10s
python3 check_ifup.py > uplog.tt

sleep 10s
sh multithread_start.sh -p /usr/local/bin/python3  -g 0 -t 20
