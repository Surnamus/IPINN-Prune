import torch

# Detect if CUDA is available, otherwise use CPU
if __name__=="__main__":
  # one inverse model (trains nu + IPINN) and one forward model (known nu, PINN)
  from modelbase import model, nu, X, T
  from forwardpinn import PINN
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # Generate test inputs from global X and T (assuming they are set by previous cells)
  test_x = torch.tensor(X.ravel(), dtype=torch.float32,device=device).unsqueeze(1)
  test_t = torch.tensor(T.ravel(), dtype=torch.float32,device=device).unsqueeze(1)
  print(f"Using device: {device}")

  print("--- Testing Inverse PINN ---")
  model.eval()
  with torch.no_grad():
    print(f"Discovered nu: {nu.item():.6f}")
    u_pred_inverse = model(test_x, test_t)
    print(f"Inverse PINN prediction shape: {u_pred_inverse.shape}")

  print("\n--- Testing Forward PINN ---")
  forward_pinn_test_model = PINN()
  forward_pinn_test_model.eval()
  with torch.no_grad():
    u_pred_forward = forward_pinn_test_model(test_x, test_t)
    print(f"Forward PINN prediction shape: {u_pred_forward.shape}")
