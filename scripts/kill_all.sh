ps -aux | grep selfplay | grep -v grep | cut -c 9-15 | xargs kill -s 9
ps -aux | grep validate | grep -v grep | cut -c 9-15 | xargs kill -s 9
