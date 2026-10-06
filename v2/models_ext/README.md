# Official EfficientNet-Lite models (downloaded, not built by us)

Google's EfficientNet-Lite family, published as ready-made TFLite files
(fp32 and full-integer int8). Source:
https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite{0..4}.tar.gz

Input: RGB at the model's native size (224/240/260/280/300 px),
float = (pixel - 127) / 128; the int8 files take uint8 input quantized from
that float. Output: 1000 ImageNet classes (softmax).

Kinds (registry.json): efl0q..efl4q (int8), efl0f..efl4f (fp32).
Variant names use the native size: efl0q_224, efl1q_240, efl2q_260, efl3q_280, efl4q_300.

The .tflite files are not in git (large); download on the Pi with:
    cd ~/stone/v2/models_ext
    for i in 0 1 2 3 4; do wget -q https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite$i.tar.gz && tar -xzf efficientnet-lite$i.tar.gz; done
    mv efficientnet-lite*/*.tflite .
