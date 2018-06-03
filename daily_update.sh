cd /work/icybee/icyElephant
sh kill_all.sh
sleep 10s
sh multithread_start.sh -p /usr/local/bin/python3  -g 1 -t 20 -n no
sleep 10s
/usr/local/bin/python3 model_update.py
sleep 10s
sh validate.sh -p /usr/local/bin/python3
sleep 10s
/usr/local/bin/python3 cal_elo.py
sleep 10s
sh multithread_start.sh -p /usr/local/bin/python3  -g 0 
