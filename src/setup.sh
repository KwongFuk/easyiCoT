# Install the packages in R1-V .
cd R1-V/src/r1-v 
pip install -e ".[dev]"
pip install wandb==0.18.3
pip install tensorboardx
pip install qwen_vl_utils 
pip install flash-attn --no-build-isolation

# Install transformers from source
cd ../../..
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout v4.49.0
pip install -e .
cd ..

SRC_DIR="./src/transformers"
TARGET_BASE="./transformers/src/transformers"

cp "$SRC_DIR/utils.py" "$TARGET_BASE/generation/utils.py"
# replace models/qwen2_vl/image_processing_qwen2_vl.py
cp "$SRC_DIR/image_processing_qwen2_vl.py" "$TARGET_BASE/models/qwen2_vl/image_processing_qwen2_vl.py"
# replace models/qwen2_vl/modeling_qwen2_vl.py
cp "$SRC_DIR/modeling_qwen2_vl.py" "$TARGET_BASE/models/qwen2_vl/modeling_qwen2_vl.py"

# install llama-factory
cd LLaMA-Factory
pip install -e ".[metrics]" --no-build-isolation
pip install trl==0.14.0
cd ..