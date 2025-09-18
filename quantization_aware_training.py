import torch
import torch.nn.functional as F
#from torchtune.models.llama3 import llama3
#from torchao.quantization.prototype.qat import Int8DynActInt4WeightQATQuantizer


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


model = "llama3(vocab_size = 4096, num_layers=16, num_heads=16, num_kv_heads=4, embed_dim=2048,  max_seq_len=2048)"

# 3. Prepare model for QAT and move to device
qat_quantizer = "Int8DynActInt4WeightQATQuantizer()"
model = qat_quantizer.prepare(model).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


model.train()
for i in range(500):
    optimizer.zero_grad(set_to_none=True)

    input_tokens = torch.randint(0, 4096, (10, 16)).to(device)
    target_tokens = torch.randint(0, 4096, (10, 16)).to(device)
    B, T = input_tokens.shape

    model_logits = model(input_tokens)


    loss = F.cross_entropy(model_logits.view(B*T, -1), target_tokens.view(-1))

    loss.backward()
    optimizer.step()

    if i % 100 == 0:
      print(f"Step {i}, Loss: {loss.item():.4f}")

print("Training finished.")


model.eval()

print("Moving model to CPU for conversion...")
model_on_cpu = model.to("cpu")
converted_model = qat_quantizer.convert(model_on_cpu)

print("\nModel successfully converted to quantized format on CPU.")
print(converted_model)




