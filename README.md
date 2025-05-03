# Transformer

Just trying to implement decoder-only gpt-style transformer with libtorch.

## Clone

```bash
git clone --recursive --depth=1 https://github.com/1Kuso4ek1/Transformer.git
```

## Build

```bash
mkdir build
cd build
cmake ..
make
```

## Run

```bash
# Try out the pretrained model!
./nn --config ../config/config.json --mode inference
```
