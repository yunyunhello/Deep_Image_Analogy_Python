cd src/

protoc caffe/proto/caffe.proto --python_out=../python/

cd ..

wget -c http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel -P deep_image_analogy/models/vgg19


