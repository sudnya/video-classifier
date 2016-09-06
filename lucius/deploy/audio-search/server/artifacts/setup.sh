#! /bin/bash

PROJECT_NAME="artifacts"

apt-get update
apt-get install --assume-yes python-pip python-dev nginx libav-tools
pip install virtualenv

virtualenv $PROJECT_NAME/env

source $PROJECT_NAME/env/bin/activate

pip install uwsgi flask flask-restful pydub requests

IP=$(dig +short myip.opendns.com @resolver1.opendns.com)
HOSTNAME=($curl -s http://169.254.169.254/latest/meta-data/public-hostname)

cat > /etc/nginx/sites-available/$PROJECT_NAME << EOF
server {
    listen 80;
    server_name $HOSTNAME;

    location / {
        include uwsgi_params;
        uwsgi_pass unix:/home/ubuntu/$PROJECT_NAME/$PROJECT_NAME.sock;
    }
}
EOF

cat > $PROJECT_NAME/$PROJECT_NAME.ini << EOF
[uwsgi]
module = wsgi

master = true
processes = 5

socket = $PROJECT_NAME.sock
chmod-socket = 660
vacuum = true

die-on-term = true
EOF


cat > /etc/init/$PROJECT_NAME.conf << EOF
description "uWSGI server instance configured to serve $PROJECT_NAME"

start on runlevel [2345]
stop on runlevel [!2345]

setuid ubuntu
setgid www-data

env PATH=/home/ubuntu/$PROJECT_NAME/env/bin:/usr/bin
chdir /home/ubuntu/$PROJECT_NAME
exec uwsgi --ini $PROJECT_NAME.ini
EOF

start $PROJECT_NAME

ln -s /etc/nginx/sites-available/$PROJECT_NAME /etc/nginx/sites-enabled

service nginx restart

