TF_CFLAGS=( $(/usr/local/bin/python2.7 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(/usr/local/bin/python2.7 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++11 -shared zero_out.cc -o zero_out.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
g++ -std=c++11 -shared inner_product.cc -o inner_product.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
g++ -std=c++11 -shared inner_product_grad.cc -o inner_product_grad.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2