#! /bin/bash

PROJECT_NAME="artifacts"

wget www.gdiamos.net/files/$PROJECT_NAME.tar.gz
tar -xzf www.gdiamos.net/files/$PROJECT_NAME.tar.gz

chmod a+x ./$PROJECT_NAME/setup.sh
./$PROJECT_NAME/setup.sh

