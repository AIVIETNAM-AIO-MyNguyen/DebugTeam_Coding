import json
from pathlib import Path
import yaml

runs_dir = Path("runs/legacy_logs") if Path("runs/legacy_logs").exists() else Path("runs")

experiments = [
    ("ResMLP Local Injection in Token Mix (RepToken)", "train_res_mlp_local_injection_in_token_mix.json", "train_res_mlp_local_injection_in_token_mix.yaml"),
    ("ResMLP Local Stem", "resmlp_local_stem.json", "resmlp_local_stem.yaml"),
    ("Pyramid ResMLP", "pyramid_res_mlp.json", "pyramid_resmlp.yaml"),
]

for title, json_file, yaml_file in experiments:
    jpath = runs_dir / json_file
    ypath = runs_dir / yaml_file
    print("=" * 75)
    print(f"EXPERIMENT: {title}")
    print("=" * 75)

    if ypath.exists():
        with open(ypath, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            model_cfg = cfg.get("model", {})
            train_cfg = cfg.get("training", {})
            data_cfg = cfg.get("data", {})
            print(f"Model Architecture: {model_cfg.get('name')}")
            print(f"Model Parameters details: {model_cfg}")
            print(f"Training settings: epochs={train_cfg.get('epochs')}, lr={train_cfg.get('lr')}, batch_size={data_cfg.get('batch_size')}")

    if jpath.exists():
        with open(jpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            epochs = len(data)
            best_val_top1 = max(ep.get("val_top1", 0) for ep in data)
            best_val_top5 = max(ep.get("val_top5", 0) for ep in data)
            final_ep = data[-1]
            best_eps = [ep for ep in data if ep.get("val_top1", 0) == best_val_top1]
            best_ep_num = best_eps[0].get("epoch", -1) if best_eps else -1

            print(f"Total Epochs Completed: {epochs}")
            print(f">>> BEST Val Top-1 Accuracy: {best_val_top1:.2f}% (achieved at Epoch {best_ep_num})")
            print(f">>> BEST Val Top-5 Accuracy: {best_val_top5:.2f}%")
            print(f"Final Epoch ({final_ep.get('epoch')}): Train Acc={final_ep.get('train_top1', 0):.2f}% | Val Acc={final_ep.get('val_top1', 0):.2f}% | Val Loss={final_ep.get('val_loss', 0):.4f}")

            print("\nConvergence Progress Across Epochs:")
            for e in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                match = [ep for ep in data if ep.get("epoch") == e]
                if match:
                    m = match[0]
                    print(f"  Epoch {e:3d}: Train Top-1={m.get('train_top1', 0):5.2f}% | Val Top-1={m.get('val_top1', 0):5.2f}% | Train Loss={m.get('train_loss', 0):.4f} | Val Loss={m.get('val_loss', 0):.4f}")
    print()
