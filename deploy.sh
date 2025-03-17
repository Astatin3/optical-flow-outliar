sshpass -p Bookshelf scp -r src/ username@10.42.0.1:~/

echo "########## Starting #########"

sshpass -p Bookshelf ssh username@10.42.0.1 "cd src && python3 main.py"
