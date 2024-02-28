. scripts/common.sh

for deps in eigen3 emp-ot emp-tool hexl SEAL-3.7
do
  if [ ! -d $BUILD_DIR/include/$deps ] 
  then
	echo -e "${RED}$deps${NC} seems absent in ${BUILD_DIR}/include/, please re-run scripts/build-deps.sh"
	exit 1
  fi
done

for deps in zstd.h 
do
  if [ ! -f $BUILD_DIR/include/$deps ] 
  then
	echo -e "${RED}$deps${NC} seems absent in ${BUILD_DIR}/include/, please re-run scripts/build-deps.sh"
	exit 1
  fi
done

cd $BUILD_DIR/
cmake .. -DCMAKE_BUILD_TYPE=Debug -DSCI_BUILD_NETWORKS=ON -DOPENSSL_ROOT_DIR=/usr/local/opt/openssl -DCMAKE_PREFIX_PATH=$BUILD_DIR -DUSE_APPROX_RESHARE=ON
for net in resnet50 sqnet densenet121
do
     make ${net}-cheetah -j4 
     make ${net}-SCI_HE -j4 
     make ${net}-hare -j4
done
make resnet50_seperate-hare -j4
make sqnet_seperate-hare -j4
make densenet121_seperate-hare -j4
make resnet50_seperate-cheetah -j4
make sqnet_seperate-cheetah -j4
make densenet121_seperate-cheetah -j4