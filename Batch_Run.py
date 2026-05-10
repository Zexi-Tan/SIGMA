import warnings

warnings.filterwarnings("ignore")
import os
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PYTHONHASHSEED'] = '2026'

import time
from torch import optim, nn
import math
import torch
import argparse
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

torch.use_deterministic_algorithms(True, warn_only=False)

# Self-defined
from utils import *
from mymodal import SIGMA
from pooling import pooling
from Contrastive_loss import calculate_contrastive

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    g = torch.Generator()
    g.manual_seed(seed)
    return g


def parse_args():
    parser = argparse.ArgumentParser(description="SIGMA Model Training")

    parser.add_argument('--dataname', type=str, default="StandWalkJump", help="Name of the dataset")
    parser.add_argument('--seeds', nargs='+', type=int, default=[2022, 2023, 2024, 2025, 2026],
                        help="List of random seeds")

    parser.add_argument('--use_mask', type=int, default=1, help="1 to use ECSM mask, 0 for Baseline")

    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--epoch_num', type=int, default=200, help="Number of training epochs")

    parser.add_argument('--output_dims', type=int, default=64, help="Output dimensions")
    parser.add_argument('--hidden_dims', type=int, default=32, help="Hidden dimensions")
    parser.add_argument('--num_view', type=int, default=2, help="Number of views")
    parser.add_argument('--quan', type=float, default=1.0, help="mask mix parameter (alpha in model)")
    parser.add_argument('--temperature', type=float, default=0.1, help="Temperature for contrastive loss")

    parser.add_argument('--patch_size', type=int, default=4, help="Size of each patch for the Stem")
    parser.add_argument('--target_keep_rate', type=float, default=0.5, help="Target keep rate for Sparsity Loss")

    parser.add_argument('--noise_std', type=float, default=0.0, help="Noise test")

    parser.add_argument('--gama', type=float, default=1.0, help="Gama parameter for contrastive loss")
    parser.add_argument('--alpha', type=float, default=1.0, help="Weight for intra-view reconstruction loss")
    parser.add_argument('--beta', type=float, default=0.1, help="Weight for inter-view diversity loss")

    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")

    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Using device: {device}")

    acc_list, f1_list, nmi_list, ari_list = [], [], [], []
    drop_list = []

    best_epoch_list = []
    convergence_time_list = []
    total_time_list = []

    global_best_acc = -1
    best_seed_history = None

    all_seeds_drops = []
    all_seeds_accs = []

    # 1. Loading/Processing Data
    data_path = f'./New_Data/{args.dataname}/{args.dataname}.npy'
    data = np.load(data_path, allow_pickle=True).item()
    train_X, train_Y, data_test, label_test = data['train_X'], data['train_Y'], data['test_X'], data['test_Y']

    all_X = np.concatenate((train_X, data_test), axis=0)
    all_Y = np.concatenate((train_Y, label_test), axis=0)

    sample_num, seq_len, feature_dim = all_X.shape

    scaler = StandardScaler()
    all_X_reshaped = all_X.reshape(-1, feature_dim)
    all_X_normalized = scaler.fit_transform(all_X_reshaped)

    if args.noise_std > 0.0:
        noise = np.random.normal(loc=0.0, scale=args.noise_std, size=all_X_normalized.shape)
        all_X_normalized = all_X_normalized + noise

    data_train = all_X_normalized.reshape(sample_num, seq_len, feature_dim)
    label_train = all_Y

    num_cluster = len(np.unique(label_train))
    print(f"Dataset: {args.dataname} | Data shape: {data_train.shape} | Clusters: {num_cluster}")

    train_data_tensor = torch.tensor(data_train, dtype=torch.float32, device=device)

    indices = torch.arange(sample_num, dtype=torch.long, device=device)
    dataset = TensorDataset(train_data_tensor, indices)

    for seed in args.seeds:
        g = setup_seed(seed)

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, generator=g)
        eval_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

        model = SIGMA(input_dims=feature_dim, output_dims=args.output_dims, seq_len=seq_len,
                    target_keep_rate=args.target_keep_rate,
                    hidden_dims=args.hidden_dims, num_views=args.num_view, patch_size=args.patch_size).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch_num, eta_min=1e-4)

        best_acc, best_f1, best_nmi, best_ari = -1, -1, -1, -1
        best_drop_rate = 0.0
        best_epoch = 0
        running_pure_time = 0.0
        best_convergence_time = 0.0

        current_seed_epochs = []
        current_seed_accs = []
        current_seed_drops = []
        seed_pure_train_times = []

        global_soft_labels = torch.zeros((sample_num, num_cluster), device=device, dtype=torch.float32)
        prev_centers = None
        momentum_alpha = 0.8

        model.eval()
        init_features_list = []
        with torch.no_grad():
            for batch_x, _ in eval_dataloader:
                R_batch = model(batch_x, alpha=1.0, use_mask=args.use_mask)
                if isinstance(R_batch, tuple): R_batch = R_batch[0]
                pooled_batch = pooling(R_batch)
                fused_batch = sum(pooled_batch) / args.num_view
                init_features_list.append(fused_batch)
        init_global_np = torch.cat(init_features_list, dim=0).cpu().numpy()

        rng_init = np.random.RandomState(seed)
        kmeans_init = KMeans(n_clusters=num_cluster, n_init=10, random_state=rng_init)
        predict_labels_np = kmeans_init.fit_predict(init_global_np)
        prev_centers = kmeans_init.cluster_centers_

        predict_labels = torch.tensor(predict_labels_np, device=device).long()
        global_soft_labels = F.one_hot(predict_labels, num_classes=num_cluster).float()
        # =========================================================================

        for epoch in range(args.epoch_num + 1):
            epoch_start_time = time.time()
            current_is_warmup = True if epoch < 30 else False

            if epoch % 5 == 0 and epoch > 0:
                model.eval()
                fused_features_list = []

                with torch.no_grad():
                    for batch_x, _ in eval_dataloader:
                        R_batch = model(batch_x, alpha=1.0, use_mask=args.use_mask)
                        if isinstance(R_batch, tuple): R_batch = R_batch[0]
                        pooled_batch = pooling(R_batch)
                        fused_batch = sum(pooled_batch) / args.num_view
                        fused_features_list.append(fused_batch)

                fused_global = torch.cat(fused_features_list, dim=0)
                fused_global_np = fused_global.cpu().numpy()

                rng_update = np.random.RandomState(seed)
                kmeans = KMeans(n_clusters=num_cluster, n_init=10, random_state=rng_update)
                current_labels_np = kmeans.fit_predict(fused_global_np)
                current_centers = kmeans.cluster_centers_

                from sklearn.metrics.pairwise import euclidean_distances
                cost_matrix = euclidean_distances(current_centers, prev_centers)
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                mapping = {row: col for row, col in zip(row_ind, col_ind)}

                current_labels_np = np.array([mapping[l] for l in current_labels_np])

                aligned_centers = np.zeros_like(current_centers)
                for new_idx, old_idx in mapping.items():
                    aligned_centers[old_idx] = current_centers[new_idx]
                prev_centers = aligned_centers

                current_labels_tensor = torch.tensor(current_labels_np, device=device).long()
                current_one_hot = F.one_hot(current_labels_tensor, num_classes=num_cluster).float()

                if current_is_warmup:
                    global_soft_labels = current_one_hot
                else:
                    global_soft_labels = momentum_alpha * global_soft_labels + (1.0 - momentum_alpha) * current_one_hot

                predict_labels = global_soft_labels.argmax(dim=1)

            model.train()
            total_loss = 0.0
            total_drop_rate = 0.0

            total_recon = 0.0
            total_con = 0.0
            total_div = 0.0

            if torch.cuda.is_available(): torch.cuda.synchronize()
            pure_train_start = time.time()

            for batch_x, batch_idx in dataloader:
                optimizer.zero_grad()

                x_whole_list, recon_loss, diversity_loss, sparsity_loss = model(
                    batch_x, alpha=args.quan, use_mask=args.use_mask)

                total_drop_rate += model.drop_rate

                pooled_features = pooling(x_whole_list)
                fused_data = sum(pooled_features) / args.num_view

                batch_pseudo_labels = predict_labels[batch_idx]

                con_loss = calculate_contrastive(fused_data, batch_pseudo_labels, tau=args.temperature)

                if current_is_warmup:
                    loss = args.alpha * recon_loss
                else:
                    ramp_up = min(1.0, (epoch - 30) / 50.0)
                    loss = args.alpha * recon_loss + ramp_up * (args.gama * con_loss + args.beta * diversity_loss)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                # total_recon += recon_loss.item()
                # total_con += con_loss.item()
                # total_div += diversity_loss.item()

            scheduler.step()

            if torch.cuda.is_available(): torch.cuda.synchronize()
            pure_train_time = time.time() - pure_train_start
            seed_pure_train_times.append(pure_train_time)
            running_pure_time += pure_train_time

            n_batches = len(dataloader)
            avg_loss = total_loss / n_batches
            avg_drop = total_drop_rate / n_batches

            print(f"Seed-{seed} Epoch-{epoch:03d} | "
                  f"Loss: {avg_loss:.4f} | "
                  #   f"Recon: {avg_recon:.4f} | "
                  #   f"Con: {avg_con:.4f} | "
                  f"Drop: {avg_drop * 100:.1f}% | "  # 顺便打出来
                  f"Time: {time.time() - epoch_start_time:.2f}s"
                  )

            # Evaluate
            if epoch == 200:
                model.eval()
                fused_features_eval_list = []

                with torch.no_grad():
                    for batch_x, _ in eval_dataloader:
                        R_batch_eval = model(batch_x, alpha=1.0, use_mask=args.use_mask)
                        if isinstance(R_batch_eval, tuple): R_batch_eval = R_batch_eval[0]
                        pooled_batch_eval = pooling(R_batch_eval)
                        fused_batch_eval = sum(pooled_batch_eval) / args.num_view
                        fused_features_eval_list.append(fused_batch_eval)

                fused_data_eval = torch.cat(fused_features_eval_list, dim=0)

                acc, f1, nmi, ari, p = clustering(fused_data_eval.cpu().numpy(), label_train, num_cluster, seed=seed)

                print(f"ACC {float(acc):.4f} | F1 {float(f1):.4f} | ARI {float(ari):.4f} | NMI {float(nmi):.4f}")

                current_seed_epochs.append(epoch)
                current_seed_accs.append(float(acc))
                current_seed_drops.append(float(avg_drop * 100))

                best_acc, best_f1, best_nmi, best_ari = acc, f1, nmi, ari
                best_drop_rate = avg_drop
                best_convergence_time = running_pure_time

        all_seeds_drops.extend(current_seed_drops)
        all_seeds_accs.extend(current_seed_accs)

        print(f"--- Seed {seed} Best ACC: {best_acc:.4f} (Achieved with Drop Rate: {best_drop_rate * 100:.2f}%) ---")
        print(
            f"--- ⚡ Time to Convergence: {best_convergence_time:.2f}s | Total Time (200 Epochs): {running_pure_time:.2f}s ---\n")

        acc_list.append(best_acc)
        f1_list.append(best_f1)
        nmi_list.append(best_nmi)
        ari_list.append(best_ari)
        drop_list.append(best_drop_rate)
        convergence_time_list.append(best_convergence_time)
        total_time_list.append(running_pure_time)

    print("\n" + "=" * 55)
    print(f"🚀 Final Results for '{args.dataname}' (Mask: {'ON' if args.use_mask else 'OFF'})")
    print(f"ACC:  {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
    print(f"F1:   {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
    print(f"NMI:  {np.mean(nmi_list):.4f} ± {np.std(nmi_list):.4f}")
    print(f"ARI:  {np.mean(ari_list):.4f} ± {np.std(ari_list):.4f}")
    print("-" * 55)
    print(f"🎯 Avg Optimal Drop Rate: {np.mean(drop_list) * 100:.2f}%")
    print(f"⏱️ Avg Convergence Epoch: {np.mean(best_epoch_list):.1f}")
    print(f"⚡ Avg Time to Convergence: {np.mean(convergence_time_list):.2f} seconds")
    print(f"⏳ Avg Total Train Time (200 Epochs): {np.mean(total_time_list):.2f} seconds")
    print("=" * 55)


if __name__ == "__main__":
    main()