# user-defined variables
PYTHON_VERSION=8 # python version (3.x)

# default variables 
PROJECT_DIR=$(pwd)
ENV_DIR=$PROJECT_DIR/.venv
QSOPTDIR=$PROJECT_DIR/qsopt
CONCORDEDIR=$PROJECT_DIR/concorde
PYTHON=python3.$PYTHON_VERSION

# color definitions
ORANGE='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m' # No Color

# prepare virtual environment
echo -e "${ORANGE}1) Preparing environment${NC}"
mkdir -p $ENV_DIR
sudo apt update
sudo apt install $PYTHON-dev $PYTHON-venv $PYTHON-tk wget -y
$PYTHON -m venv $ENV_DIR

# install Concorde
echo -e "${ORANGE}2) Installing Concorde${NC}"

echo -e "${BLUE}=> download${NC}"
mkdir -p $QSOPTDIR
wget -O $QSOPTDIR/qsopt.a http://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/PIC/qsopt.PIC.a
wget -O $QSOPTDIR/qsopt.h http://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/PIC/qsopt.h
wget -O $PROJECT_DIR/concorde.tgz http://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/src/co031219.tgz
tar -xzf $PROJECT_DIR/concorde.tgz -C $PROJECT_DIR
rm -rf $PROJECT_DIR/concorde.tgz

echo -e "${BLUE}=> configuration${NC}"
cd $CONCORDEDIR
./configure --with-qsopt=$QSOPTDIR

echo -e "${BLUE}=> compilation${NC}"
make

# adjust PATH
echo -e "${BLUE}=> adjusting PATH${NC}"
if ! grep -q "$CONCORDEDIR/TSP" $ENV_DIR/bin/activate ; then
   echo export PATH=\$PATH:$CONCORDEDIR/TSP >> $ENV_DIR/bin/activate
fi

# install Python requirements
echo -e "${ORANGE} 3) Installing Python requirements${NC}"
source $ENV_DIR/bin/activate
cd $PROJECT_DIR
pip install -r requirements.txt