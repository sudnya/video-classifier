#! /bin/bash

PROJECT_NAME="artifacts"

wget www.gdiamos.net/files/$PROJECT_NAME.tar.gz
tar -xzf $PROJECT_NAME.tar.gz

chmod a+x ./$PROJECT_NAME/setup.sh
chmod -R a+w ./$PROJECT_NAME/
./$PROJECT_NAME/setup.sh

