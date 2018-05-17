ps -aux | grep self_play | grep -v grep | cut -c 9-15 | xargs kill -s 9
