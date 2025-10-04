# test_detailed.py
packages = ['torch', 'transformers', 'flask', 'pandas', 'numpy', 'sentencepiece', 'protobuf']

for package in packages:
    try:
        __import__(package)
        print(f"✅ {package} - OK")
    except ImportError as e:
        print(f"❌ {package} - MISSING: {e}")

print("\nChecking versions:")
try:
    import torch
    print(f"Torch: {torch.__version__}")
except:
    print("Torch: Not available")

try:
    import transformers
    print(f"Transformers: {transformers.__version__}")
except:
    print("Transformers: Not available")