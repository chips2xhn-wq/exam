import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from sklearn.preprocessing import StandardScaler
import os
import sys
import logging
from datetime import datetime
import optuna
import time
import math
import pickle

# 1. 设置与全局路径
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'LaRC05_Compression_FPF_3000.csv')
split_seed = 42
split_ratio = (0.7, 0.2, 0.1)

# 压缩/拉伸共用的搜索与训练轮次。
OPTUNA_TRIALS = 150
PRUNING_EPOCHS = 60
FINAL_TRAINING_EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 60

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TeeStream:
    """将终端输出同步写入运行日志。"""
    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], 'encoding', 'utf-8')

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
        return len(message)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, 'isatty', lambda: False)() for stream in self.streams)


def start_run_log(task_name):
    """创建带时间戳的日志，并捕获 print、Optuna 日志和异常堆栈。"""
    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_path = os.path.join(log_dir, f'{task_name}-{timestamp}.log')
    log_file = open(log_path, 'a', encoding='utf-8', buffering=1)

    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)

    # Optuna 在导入时可能已经绑定原始 stderr，这里改为 TeeStream。
    optuna_logger = optuna.logging.get_logger('optuna')
    for handler in optuna_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setStream(sys.stderr)

    return log_path, log_file


def plot_nature_results(train_losses, val_losses, trues_real, preds_real, r2, rmse, save_stem):
    """Create a compact, publication-ready training and test figure."""
    nature_style = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 8,
        'axes.labelsize': 8.5,
        'axes.titlesize': 9,
        'axes.linewidth': 0.8,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
        'legend.fontsize': 7.5,
        'lines.linewidth': 1.25,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.facecolor': 'white',
        'figure.facecolor': 'white',
    }
    blue, orange, gray = '#0072B2', '#D55E00', '#4D4D4D'
    floor = np.finfo(np.float64).tiny
    train_curve = np.maximum(np.asarray(train_losses, dtype=float), floor)
    val_curve = np.maximum(np.asarray(val_losses, dtype=float), floor)
    true_force = np.asarray(trues_real, dtype=float)
    predicted_force = np.asarray(preds_real, dtype=float)

    with plt.rc_context(nature_style):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), constrained_layout=True)

        epochs = np.arange(1, len(train_curve) + 1)
        axes[0].semilogy(epochs, train_curve, color=blue, label='Training')
        axes[0].semilogy(epochs, val_curve, color=orange, label='Validation')
        axes[0].set_title('Training history', loc='left', fontweight='semibold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MSE loss (scaled log force)')
        axes[0].legend(frameon=False, loc='upper right')
        axes[0].grid(axis='y', which='major', color='#D9D9D9', linewidth=0.55)

        lower = float(min(true_force.min(), predicted_force.min()))
        upper = float(max(true_force.max(), predicted_force.max()))
        padding = max(1.0, 0.035 * (upper - lower))
        limits = [lower - padding, upper + padding]
        axes[1].scatter(
            true_force, predicted_force,
            s=16, facecolors=blue, edgecolors='white', linewidths=0.35,
            alpha=0.72, rasterized=True, zorder=2,
        )
        axes[1].plot(limits, limits, linestyle='--', color=gray, linewidth=1.0, zorder=1)
        axes[1].set_xlim(limits)
        axes[1].set_ylim(limits)
        axes[1].set_aspect('equal', adjustable='box')
        axes[1].set_title('Held-out test set', loc='left', fontweight='semibold')
        axes[1].set_xlabel('True force (N)')
        axes[1].set_ylabel('Predicted force (N)')
        axes[1].xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        axes[1].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        axes[1].text(
            0.05, 0.95, f'$R^2$ = {r2:.5f}\nRMSE = {rmse:.2f} N',
            transform=axes[1].transAxes, ha='left', va='top',
        )

        for panel, label in zip(axes, ('a', 'b')):
            panel.spines['top'].set_visible(False)
            panel.spines['right'].set_visible(False)
            panel.tick_params(which='both', top=False, right=False, length=3.2)
            panel.tick_params(which='minor', length=1.8)
            panel.minorticks_on()
            panel.text(
                -0.13, 1.05, label, transform=panel.transAxes,
                fontsize=10, fontweight='bold', va='top',
            )

        png_path = f'{save_stem}.png'
        pdf_path = f'{save_stem}.pdf'
        fig.savefig(png_path, dpi=600, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        plt.close(fig)
    return png_path, pdf_path

# 2. 列定义与预处理工具
# 定义列名映射 (12角度 + 1力 + 20材料)
COL_NAMES = [f"Angle_{i}" for i in range(12)] + ["Force"] + [
    '密度', 'E11', 'E22', 'v12', 'v13', 'v23', 'G12', 'G13', 'G23', 
    'XT', 'XC', 'YT', 'YC', 'ST', 'SC', 'Gft', 'Gfe', 'Gmt', 'Gme', 't'
]
# --- 材料参数缩放器 (混合策略: MagScale + Group Z-Score) ---
class MaterialGroupScaler:
    def __init__(self):
        self.groups = {
            'Density': ['密度'],
            'Modulus': ['E11', 'E22', 'G12', 'G13', 'G23'],
            'v': ['v12', 'v13', 'v23'],
            'Strength': ['XT', 'XC', 'YT', 'YC', 'ST', 'SC'],
            'Gf': ['Gft', 'Gfe'],
            'Gm': ['Gmt', 'Gme'],
            't': ['t']
        }
        self.stats = {} 
        
    def fit(self, df):
        """仅在训练集上计算统计量"""
        print("\n[Material Preprocessing] Calculating Stats on Training Data:")
        for group_name, cols in self.groups.items():
            valid_cols = [c for c in cols if c in df.columns]
            if not valid_cols: continue
            
            values = df[valid_cols].values
            
            if group_name in ['Density', 't']:
                # 策略 1: Magnitude Scaling (数量级缩放)
                max_val = np.max(np.abs(values))
                if max_val == 0: scale = 1.0
                else:
                    power = np.ceil(np.log10(max_val))
                    scale = 10 ** (-power)
                self.stats[group_name] = {'type': 'magnitude', 'scale': scale, 'cols': valid_cols}
                
            else:
                # 策略 2: Group Z-Score (组内标准化)
                # 计算该组所有数据的均值和方差 
                group_vals = values.flatten()
                mean_val = np.mean(group_vals)
                std_val = np.std(group_vals)
                if std_val == 0: std_val = 1.0
                
                self.stats[group_name] = {'type': 'zscore', 'mean': mean_val, 'std': std_val, 'cols': valid_cols}

    def transform(self, df):
        """应用统计量进行转换"""
        df_scaled = df.copy()
        for group_name, params in self.stats.items():
            cols = params['cols']
            if params['type'] == 'magnitude':
                for c in cols:
                    df_scaled[c] = df[c] * params['scale']
            elif params['type'] == 'zscore':
                for c in cols:
                    df_scaled[c] = (df[c] - params['mean']) / params['std']
        
        # 返回仅包含材料列的 numpy 数组 (保持顺序)
        mat_cols = [c for c in df.columns if c in COL_NAMES[13:]]
        return df_scaled[mat_cols].values.astype(np.float32)


# 3. 数据集类
class PhysicsDataset(Dataset):
    def __init__(self, angle_features, labels, mat_features, scaler=None, y_scaler=None, fit_scaler=False):
        # 1. 角度: Sin/Cos 变换
        radians = angle_features * (np.pi / 180.0)
        self.features = np.concatenate([np.sin(2 * radians), np.cos(2 * radians)], axis=1).astype(np.float32)
        
        # 2. 材料: 直接使用预处理后的 features
        self.mat_params = mat_features.astype(np.float32)
        
        # 3. 力: Log -- StandardScaler
        labels_log = np.log(labels.reshape(-1, 1))
        if fit_scaler:
            self.y_scaler = StandardScaler()
            self.labels = self.y_scaler.fit_transform(labels_log).flatten()
        else:
            self.y_scaler = y_scaler
            if y_scaler is not None:
                self.labels = self.y_scaler.transform(labels_log).flatten()
            else:
                self.labels = labels_log.flatten()

    def __getitem__(self, index):
        x = torch.tensor(self.features[index], dtype=torch.float32)
        mat = torch.tensor(self.mat_params[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return x, mat, label

    def __len__(self):
        return len(self.features)

# 4. 网络架构 (ResNet + Fusion)
class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))

class PhysicsResNetNN(nn.Module):
    def __init__(self, mat_dim, hidden_dim=32, dropout_rate=0.1):
        super().__init__()
        input_angle_dim = 24  # 12层 * 2 (sin/cos)
        output_angle_dim = math.ceil(input_angle_dim//12)*128
        # 角度分支
        self.angle_encoder = nn.Sequential(
            nn.Linear(input_angle_dim, output_angle_dim), nn.LayerNorm(output_angle_dim), nn.GELU(),
            ResNetBlock(output_angle_dim),
            nn.Linear(output_angle_dim, output_angle_dim),
            nn.LayerNorm(output_angle_dim),
            nn.GELU(),
            nn.Linear(output_angle_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        
        # 融合分支
        fusion_input_dim = hidden_dim + mat_dim
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_input_dim , output_angle_dim), 
            nn.LayerNorm(output_angle_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(output_angle_dim, 32),
            nn.LayerNorm(32), 
            nn.GELU(),
            nn.Linear(32, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x, mat_params): 
        angle_feat = self.angle_encoder(x)
        combined = torch.cat([angle_feat, mat_params], dim=1)
        out = self.fusion_net(combined)
        return out

# 5. 全局数据加载
X_train_ang, y_train, X_train_mat = None, None, None
X_val_ang, y_val, X_val_mat = None, None, None
X_test_ang, y_test, X_test_mat = None, None, None
mat_dim_global = 0
mat_scaler = None

def load_and_preprocess_global_data():
    global X_train_ang, y_train, X_train_mat
    global X_val_ang, y_val, X_val_mat
    global X_test_ang, y_test, X_test_mat, mat_dim_global, mat_scaler
    
    if not os.path.exists(data_path):
        print(f"错误：未找到数据文件 {data_path}")
        return False

    print("正在加载 LaRC05 压缩 FPF 数据...")
    df = pd.read_csv(data_path)
    angle_cols = [f"Ply{i}" for i in range(1, 13)]
    force_col = "CriticalLoad_N"
    required_cols = angle_cols + [force_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV 缺少必要列: {missing_cols}")

    model_df = df[required_cols].apply(pd.to_numeric, errors='coerce')
    invalid_mask = model_df.isna().any(axis=1) | (model_df[force_col] <= 0)
    if invalid_mask.any():
        raise ValueError(f"发现 {int(invalid_mask.sum())} 条无效数据（含空值、非数值或非正载荷）")

    # 固定随机种子打乱后按 7:2:1 划分；3000 条数据对应 2100/600/300。
    rng = np.random.default_rng(split_seed)
    shuffled_indices = rng.permutation(len(model_df))
    n_train = int(len(model_df) * split_ratio[0])
    n_val = int(len(model_df) * split_ratio[1])
    train_df = model_df.iloc[shuffled_indices[:n_train]].reset_index(drop=True)
    val_df = model_df.iloc[shuffled_indices[n_train:n_train + n_val]].reset_index(drop=True)
    test_df = model_df.iloc[shuffled_indices[n_train + n_val:]].reset_index(drop=True)

    X_train_ang = train_df[angle_cols].values.astype(np.float32)
    X_val_ang = val_df[angle_cols].values.astype(np.float32)
    X_test_ang = test_df[angle_cols].values.astype(np.float32)
    y_train = train_df[force_col].values.astype(np.float32)
    y_val = val_df[force_col].values.astype(np.float32)
    y_test = test_df[force_col].values.astype(np.float32)

    # 本 CSV 不含材料参数；保留网络接口并使用 0 维材料特征。
    X_train_mat = np.empty((len(train_df), 0), dtype=np.float32)
    X_val_mat = np.empty((len(val_df), 0), dtype=np.float32)
    X_test_mat = np.empty((len(test_df), 0), dtype=np.float32)
    mat_scaler = None
    mat_dim_global = 0

    print(
        f"数据加载完成（seed={split_seed}）: "
        f"Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)}"
    )
    return True

# 6. Optuna 目标函数
def objective(trial):
    # A. 搜索空间
    # 两个任务使用相同搜索空间，最优超参数允许不同。
    suggested_hidden_dim = trial.suggest_categorical("hidden_dim", [6, 18, 30, 42, 54, 72])
    suggested_dropout = trial.suggest_float("dropout_rate", 0.0, 0.4)
    suggested_lr = trial.suggest_float("lr", 8e-4, 2e-3, log=True)
    suggested_wd = trial.suggest_float("weight_decay", 8e-5, 2e-4, log=True)
    suggested_batch_size = trial.suggest_categorical("batch_size", [ 16, 32, 64])

    # B. 构建数据集
    train_ds = PhysicsDataset(X_train_ang, y_train, X_train_mat, fit_scaler=True)
    val_ds = PhysicsDataset(X_val_ang, y_val, X_val_mat, y_scaler=train_ds.y_scaler, fit_scaler=False)
    
    train_loader = DataLoader(train_ds, batch_size=suggested_batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=suggested_batch_size, shuffle=False, drop_last=False)

    # C. 模型
    model = PhysicsResNetNN(mat_dim=mat_dim_global, 
                            hidden_dim=suggested_hidden_dim, 
                            dropout_rate=suggested_dropout).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=suggested_lr, weight_decay=suggested_wd)
    criterion = nn.MSELoss()

    # D. 快速训练循环 (Pruning)
    pruning_epochs = PRUNING_EPOCHS
    
    for epoch in range(pruning_epochs):
        model.train()
        for ang, mat, lbl in train_loader:
            ang, mat, lbl = ang.to(device), mat.to(device), lbl.to(device)
            optimizer.zero_grad()
            out = model(ang, mat).view(-1)
            loss = criterion(out, lbl)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for ang, mat, lbl in val_loader:
                ang, mat, lbl = ang.to(device), mat.to(device), lbl.to(device)
                out = model(ang, mat).view(-1)
                val_loss += criterion(out, lbl).item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return avg_val_loss

# 7. 主程序
if __name__ == '__main__':
    log_path, _run_log_file = start_run_log('DD-ply12-com')
    print(f"运行日志: {log_path}")
    print(f"启动时间: {datetime.now().isoformat(timespec='seconds')}")
    print(f"脚本路径: {os.path.abspath(__file__)}")
    print(f"数据路径: {data_path}")
    print(f"运行环境: Python {sys.version.split()[0]}, PyTorch {torch.__version__}, device={device}")
    print(
        f"实验配置: split={split_ratio}, seed={split_seed}, trials={OPTUNA_TRIALS}, "
        f"pruning_epochs={PRUNING_EPOCHS}, final_epochs={FINAL_TRAINING_EPOCHS}, "
        f"early_stopping_patience={EARLY_STOPPING_PATIENCE}"
    )

    # 记录总开始时间
    total_start_time = time.time()
    np.random.seed(split_seed)
    torch.manual_seed(split_seed)
    
    if load_and_preprocess_global_data():
        
        print("开始 Optuna 贝叶斯超参数优化...")
        optuna_start_time = time.time()
        
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=split_seed),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=0, interval_steps=1
            ),
        )
        #初始参数
        study.enqueue_trial({
            "hidden_dim": 18,
            "batch_size": 32,          
            "lr": 1e-3,               
            "weight_decay": 1e-4,      
            "dropout_rate": 0.1
        })
        study.optimize(objective, n_trials=OPTUNA_TRIALS)
        
        # Optuna优化结束，记录时间
        optuna_end_time = time.time()
        optuna_duration = optuna_end_time - optuna_start_time
        
        print("\n最佳参数:")
        best_params = study.best_params
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        print(f"\nOptuna超参数优化耗时: {optuna_duration:.2f} 秒 ({optuna_duration/60:.2f} 分钟)")
            
        # 2. 用最佳参数进行训练
        print("\n用最佳参数训练...")
        train_start_time = time.time()
        
        train_ds = PhysicsDataset(X_train_ang, y_train, X_train_mat, fit_scaler=True)
        y_scaler_final = train_ds.y_scaler
        val_ds = PhysicsDataset(X_val_ang, y_val, X_val_mat, y_scaler=y_scaler_final, fit_scaler=False)
        test_ds = PhysicsDataset(X_test_ang, y_test, X_test_mat, y_scaler=y_scaler_final, fit_scaler=False)
        
        best_bs = best_params['batch_size']
        train_loader = DataLoader(train_ds, batch_size=best_bs, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=best_bs, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=best_bs, shuffle=False)
        
        model = PhysicsResNetNN(mat_dim=mat_dim_global, 
                                hidden_dim=best_params['hidden_dim'], 
                                dropout_rate=best_params['dropout_rate']).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
        criterion = nn.MSELoss()
        
        epochs = FINAL_TRAINING_EPOCHS
        best_val_loss = float('inf')
        patience = EARLY_STOPPING_PATIENCE
        counter = 0
        train_losses, val_losses = [], []
        best_model_state = None
        
        for epoch in range(epochs):
            model.train()
            train_loss_ep = 0
            for ang, mat, lbl in train_loader:
                ang, mat, lbl = ang.to(device), mat.to(device), lbl.to(device)
                optimizer.zero_grad()
                out = model(ang, mat).view(-1)
                loss = criterion(out, lbl)
                loss.backward()
                optimizer.step()
                train_loss_ep += loss.item()
            
            avg_train_loss = train_loss_ep / len(train_loader)
            train_losses.append(avg_train_loss)
            
            model.eval()
            val_loss_ep = 0
            with torch.no_grad():
                for ang, mat, lbl in val_loader:
                    ang, mat, lbl = ang.to(device), mat.to(device), lbl.to(device)
                    out = model(ang, mat).view(-1)
                    val_loss_ep += criterion(out, lbl).item()
            
            avg_val_loss = val_loss_ep / len(val_loader)
            val_losses.append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                best_model_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            else:
                counter += 1
                
            if counter >= patience:
                print(f"早停触发于 Epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break
                
            if (epoch+1) % 50 == 0:
                print(f"Epoch {epoch+1} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 训练结束，记录时间
        train_end_time = time.time()
        train_duration = train_end_time - train_start_time
        print(f"\n最终模型训练耗时: {train_duration:.2f} 秒 ({train_duration/60:.2f} 分钟)")

        # 3. 最终评估
        print("\n最终测试集评估...")
        eval_start_time = time.time()
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        model.eval()
        
        preds, trues = [], []
        with torch.no_grad():
            for ang, mat, lbl in test_loader:
                ang, mat, lbl = ang.to(device), mat.to(device), lbl.to(device)
                out = model(ang, mat).view(-1)
                preds.extend(out.cpu().numpy())
                trues.extend(lbl.cpu().numpy())
                
        preds_real = np.exp(y_scaler_final.inverse_transform(np.array(preds).reshape(-1,1)).flatten())
        trues_real = np.exp(y_scaler_final.inverse_transform(np.array(trues).reshape(-1,1)).flatten())
        
        # 评估结束，记录时间
        eval_end_time = time.time()
        eval_duration = eval_end_time - eval_start_time
        
        mse = np.mean((preds_real - trues_real)**2)
        rmse = np.sqrt(mse) 
        r2 = 1 - (np.sum((trues_real - preds_real)**2) / np.sum((trues_real - np.mean(trues_real))**2))
        
        print(f"测试结果评估耗时: {eval_duration:.2f} 秒")
        print(f"最终结果: RMSE={rmse:.4f}, R²={r2:.4f}")
        
        print("【异常分析】相对误差 > 10% 的样本")
        rel_err = np.abs(preds_real - trues_real) / (trues_real + 1e-8)
        err_idx = np.where(rel_err > 0.10)[0]
        
        if len(err_idx) > 0:
            print(f"{'Index':^6} | {'Features (Angles)':^40} | {'True':^10} | {'Pred':^10} | {'Error %':^10}")
            print("-" * 80)
            for i in err_idx:
                raw_feats = X_test_ang[i]
                feat_str = ", ".join([f"{f:.0f}" for f in raw_feats])
                print(f"{i:^6d} | [{feat_str:^38}] | {trues_real[i]:^10.2f} | {-preds_real[i]:^10.2f} | {rel_err[i]*100:^9.2f}%")
        else:
            print("未发现异常样本。")
        
        save_stem = os.path.join(base_dir, 'DD-ply12-com-result')
        save_path, pdf_path = plot_nature_results(
            train_losses, val_losses, trues_real, preds_real, r2, rmse, save_stem
        )
        print(f"结果图已保存至: {save_path}")
        print(f"矢量图已保存至: {pdf_path}")
        
        # 保存模型
        model_save_dir = os.path.join(base_dir, 'models')
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir, exist_ok=True)
        
        model_save_path = os.path.join(model_save_dir, 'DD-ply12-com.pth')
        torch.save({
            'model_state_dict': best_model_state,
            'best_params': best_params,
            'mat_dim': mat_dim_global,
            'rmse': rmse,
            'r2': r2,
            'y_scaler': y_scaler_final,
            'mat_scaler': mat_scaler,
            'data_file': os.path.basename(data_path),
            'angle_columns': [f"Ply{i}" for i in range(1, 13)],
            'target_column': 'CriticalLoad_N',
            'split_ratio': split_ratio,
            'split_seed': split_seed,
        }, model_save_path)
        print(f"模型已保存至: {model_save_path}")
        
        # 保存 scalers 到单独的文件
        scalers_save_path = os.path.join(model_save_dir, 'DD-ply12-com.pkl')
        with open(scalers_save_path, 'wb') as f:
            pickle.dump({
                'y_scaler': y_scaler_final,
                'mat_scaler': mat_scaler,
                'data_file': os.path.basename(data_path),
                'angle_columns': [f"Ply{i}" for i in range(1, 13)],
                'target_column': 'CriticalLoad_N',
                'split_ratio': split_ratio,
                'split_seed': split_seed,
            }, f)
        print(f"Scalers已保存至: {scalers_save_path}")
        
        # 输出总耗时
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        print("\n" + "="*60)
        print("计算时长统计:")
        print(f"  Optuna超参数优化: {optuna_duration:.2f} 秒 ({optuna_duration/60:.2f} 分钟)")
        print(f"  最终模型训练:     {train_duration:.2f} 秒 ({train_duration/60:.2f} 分钟)")
        print(f"  测试集评估:       {eval_duration:.2f} 秒")
        print(f"  总耗时:           {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟, {total_duration/3600:.2f} 小时)")
        print("="*60)
    print(f"运行日志已保存至: {log_path}")
# 最佳参数:
#   hidden_dim: 42
#   dropout_rate: 0.06413503347893501
#   lr: 0.0015257751356602778
#   weight_decay: 0.00010529868804094298
#   batch_size: 32
# ============================================================
# 计算时长统计:
#   Optuna超参数优化: 491.14 秒 (8.19 分钟)
#   最终模型训练:     105.50 秒 (1.76 分钟)
#   测试集评估:       0.02 秒
#   总耗时:           596.64 秒 (9.94 分钟, 0.166 小时)
# ============================================================
