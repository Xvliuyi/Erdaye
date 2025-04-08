import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import mean_squared_error, mean_absolute_error
# ================== LSTM核心模块 ==================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2
class LSTMParameter:
    def __init__(self, data):
        self.data = data
        self.diff = np.zeros_like(data)
class LstmParam:
    def __init__(self, mem_cell_ct, x_dim):
        self.mem_cell_ct = mem_cell_ct
        self.x_dim = x_dim
        concat_len = x_dim + mem_cell_ct
        self.wg = LSTMParameter(np.random.randn(mem_cell_ct, concat_len) * 0.1)
        self.wi = LSTMParameter(np.random.randn(mem_cell_ct, concat_len) * 0.1)
        self.wf = LSTMParameter(np.random.randn(mem_cell_ct, concat_len) * 0.1)
        self.wo = LSTMParameter(np.random.randn(mem_cell_ct, concat_len) * 0.1)
        self.bg = LSTMParameter(np.random.randn(mem_cell_ct) * 0.1)
        self.bi = LSTMParameter(np.random.randn(mem_cell_ct) * 0.1)
        self.bf = LSTMParameter(np.random.randn(mem_cell_ct) * 0.1)
        self.bo = LSTMParameter(np.random.randn(mem_cell_ct) * 0.1)
        self.params = [self.wg, self.wi, self.wf, self.wo,
                       self.bg, self.bi, self.bf, self.bo]

    def apply_diff(self, lr=1, grad_clip=1.0):
        for param in self.params:
            np.clip(param.diff, -grad_clip, grad_clip, out=param.diff)
            param.data -= lr * param.diff
            param.diff.fill(0)
class LstmState:
    def __init__(self, mem_cell_ct, x_dim):
        self.g = np.zeros(mem_cell_ct)
        self.i = np.zeros(mem_cell_ct)
        self.f = np.zeros(mem_cell_ct)
        self.o = np.zeros(mem_cell_ct)
        self.s = np.zeros(mem_cell_ct)
        self.h = np.zeros(mem_cell_ct)
        self.bottom_diff_h = np.zeros_like(self.h)
        self.bottom_diff_s = np.zeros_like(self.s)
class LstmNode:
    def __init__(self, lstm_param, lstm_state):
        self.state = lstm_state
        self.param = lstm_param
        self.xc = None

    def bottom_data_is(self, x, s_prev=None, h_prev=None):
        if s_prev is None: s_prev = np.zeros_like(self.state.s)
        if h_prev is None: h_prev = np.zeros_like(self.state.h)
        x = x.flatten()
        xc = np.hstack((x, h_prev))
        self.s_prev = s_prev
        self.h_prev = h_prev
        self.state.g = np.tanh(np.dot(self.param.wg.data, xc) + self.param.bg.data)
        self.state.i = sigmoid(np.dot(self.param.wi.data, xc) + self.param.bi.data)
        self.state.f = sigmoid(np.dot(self.param.wf.data, xc) + self.param.bf.data)
        self.state.o = sigmoid(np.dot(self.param.wo.data, xc) + self.param.bo.data)
        self.state.s = self.state.g * self.state.i + s_prev * self.state.f
        self.state.h = self.state.s * self.state.o
        self.xc = xc

    def top_diff_is(self, top_diff_h, top_diff_s):
        ds = self.state.o * top_diff_h + top_diff_s
        do = self.state.s * top_diff_h
        di = self.state.g * ds
        dg = self.state.i * ds
        df = self.s_prev * ds
        di_input = sigmoid_derivative(self.state.i) * di
        df_input = sigmoid_derivative(self.state.f) * df
        do_input = sigmoid_derivative(self.state.o) * do
        dg_input = tanh_derivative(self.state.g) * dg

        self.param.wi.diff += np.outer(di_input, self.xc)
        self.param.wf.diff += np.outer(df_input, self.xc)
        self.param.wo.diff += np.outer(do_input, self.xc)
        self.param.wg.diff += np.outer(dg_input, self.xc)
        self.param.bi.diff += di_input
        self.param.bf.diff += df_input
        self.param.bo.diff += do_input
        self.param.bg.diff += dg_input

        dxc = np.zeros_like(self.xc)
        dxc += np.dot(self.param.wi.data.T, di_input)
        dxc += np.dot(self.param.wf.data.T, df_input)
        dxc += np.dot(self.param.wo.data.T, do_input)
        dxc += np.dot(self.param.wg.data.T, dg_input)
        self.state.bottom_diff_s = ds * self.state.f
        self.state.bottom_diff_h = dxc[self.param.x_dim:]
class LstmNetwork:
    def __init__(self, lstm_param):
        self.lstm_param = lstm_param
        self.lstm_node_list = []
        self.x_list = []

    def reset_states(self):
        for node in self.lstm_node_list:
            node.state = LstmState(self.lstm_param.mem_cell_ct, self.lstm_param.x_dim)

    def x_list_clear(self): self.x_list = []

    def x_list_add(self, x):
        self.x_list.append(x)
        if len(self.x_list) > len(self.lstm_node_list):
            lstm_state = LstmState(self.lstm_param.mem_cell_ct, self.lstm_param.x_dim)
            self.lstm_node_list.append(LstmNode(self.lstm_param, lstm_state))
        idx = len(self.x_list) - 1
        if idx == 0:
            self.lstm_node_list[idx].bottom_data_is(x)
        else:
            s_prev = self.lstm_node_list[idx - 1].state.s
            h_prev = self.lstm_node_list[idx - 1].state.h
            self.lstm_node_list[idx].bottom_data_is(x, s_prev, h_prev)

    def backward(self):
        for t in reversed(range(len(self.lstm_node_list))):
            node = self.lstm_node_list[t]
            top_diff_s = node.state.bottom_diff_s
            if t < len(self.lstm_node_list) - 1:
                top_diff_s += self.lstm_node_list[t + 1].state.bottom_diff_s
            node.top_diff_is(node.state.bottom_diff_h, top_diff_s)
class EnhancedLSTM:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.lstm_param = LstmParam(hidden_dim, input_dim)
        self.lstm_net = LstmNetwork(self.lstm_param)
        self.output_weights = np.random.randn(hidden_dim, output_dim) * 0.01
        self.output_bias = np.zeros(output_dim)
        self.output_weights_diff = np.zeros_like(self.output_weights)
        self.output_bias_diff = np.zeros_like(self.output_bias)
        self.loss_diff_func = None

    def set_loss_func(self, loss_diff_func): self.loss_diff_func = loss_diff_func

    def forward(self, x_seq):
        self.lstm_net.x_list_clear()
        for x in x_seq:
            x = x.flatten()
            self.lstm_net.x_list_add(x)
        if self.lstm_net.lstm_node_list:
            last_h = self.lstm_net.lstm_node_list[-1].state.h
            last_h = last_h.reshape(1, -1)
            output = np.dot(last_h, self.output_weights) + self.output_bias
            return output.flatten()
        else:
            return np.zeros(self.output_bias.shape)

    def backward(self, pred, y_true):
        for node in self.lstm_net.lstm_node_list:
            node.state.bottom_diff_h.fill(0)
            node.state.bottom_diff_s.fill(0)
        d_output = self.loss_diff_func(pred, y_true)
        h = self.lstm_net.lstm_node_list[-1].state.h.reshape(1, -1)
        self.output_weights_diff += np.dot(h.T, d_output.reshape(1, -1))
        self.output_bias_diff += d_output
        d_hidden = np.dot(d_output.reshape(1, -1), self.output_weights.T)
        self.lstm_net.lstm_node_list[-1].state.bottom_diff_h += d_hidden.flatten()
        self.lstm_net.backward()

    def apply_gradients(self, lr=0.01, grad_clip=1.0):
        self.lstm_param.apply_diff(lr, grad_clip)
        np.clip(self.output_weights_diff, -grad_clip, grad_clip, out=self.output_weights_diff)
        np.clip(self.output_bias_diff, -grad_clip, grad_clip, out=self.output_bias_diff)
        self.output_weights -= lr * self.output_weights_diff
        self.output_bias -= lr * self.output_bias_diff
        self.output_weights_diff.fill(0)
        self.output_bias_diff.fill(0)
# ================== 损失函数模块 ==================
class VectorLossLayer:
    @staticmethod
    def loss(pred, target): return 0.5 * np.sum((pred - target) ** 2)
    @staticmethod
    def bottom_diff(pred, target): return (pred - target).flatten()
# ================== 数据预处理模块（修复try-except结构）==================
class CSVProcessor:
    @staticmethod
    def process_folder(folder_path, window_size=20, predict_start=20, predict_end=40):
        valid_datasets = []
        csv_files = list(Path(folder_path).glob("*.csv"))
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(CSVProcessor._safe_process_file, f, window_size, predict_start, predict_end)
                       for f in csv_files]
            valid_datasets = [f.result() for f in futures if f.result() is not None]
        if not valid_datasets: raise ValueError("无有效CSV文件")
        return valid_datasets

    @staticmethod
    def _safe_process_file(file_path, window_size, predict_start, predict_end):
        """修复try-except结构"""
        try:
            # 自动检测文件编码
            with open(file_path, 'rb') as f:
                rawdata = f.read(10000)
                encoding_detect = chardet.detect(rawdata)['encoding']

            df = pd.read_csv(file_path,
                             usecols=['tx', 'ty', 'tz'],
                             encoding=encoding_detect,
                             engine='python',
                             on_bad_lines='skip')

            if df.empty or any(col not in df.columns for col in ['tx', 'ty', 'tz']):
                raise ValueError(f"文件 {file_path.name} 缺失必需列")

            return CSVProcessor._process_valid_data(df, file_path, window_size, predict_start, predict_end)

        except Exception as e:  # 添加缺失的except块
            print(f"文件处理错误: {file_path.name} | 原因: {str(e)}")
            return None

    @staticmethod
    def _process_valid_data(df, file_path, window_size, predict_start, predict_end):
        """修复缩进和异常处理"""
        try:  # 添加完整的try-except结构
            for axis in ['x', 'y', 'z']:
                df[f'delta_t{axis}'] = df[f't{axis}'].diff().fillna(0)
                df[f'acc_t{axis}'] = df[f'delta_t{axis}'].diff().fillna(0)
                df[f'angle_t{axis}'] = np.arctan(df[f'delta_t{axis}'] / (df[f't{axis}'] + 1e-6))

            features = df[[c for c in df if c.startswith(('tx', 'ty', 'tz', 'delta', 'acc', 'angle'))]].values.astype(
                np.float32)
            targets = df[['tx', 'ty', 'tz']].values.astype(np.float32)

            feature_scaler = StandardScaler().fit(features)
            target_scaler = StandardScaler().fit(targets)

            X, y = [], []
            start_idx = max(predict_start - window_size, 0)

            for i in range(start_idx, predict_end - window_size + 1):
                if (i + window_size) < len(features):
                    X.append(feature_scaler.transform(features[i:i + window_size]))
                    y.append(target_scaler.transform(targets[i + window_size].reshape(1, -1)))

            return (np.array(X), np.array(y), target_scaler, str(file_path))

        except IndexError as e:
            raise ValueError(f"数据长度不足: {str(e)}")
        except Exception as e:
            print(f"数据处理异常: {str(e)}")
            return None
# ================== 预测记录器模块（需放在训练类之前）==================
class PredictionRecorder:
    def __init__(self, output_dir="predictions"):
        self.output_dir = Path(output_dir)
        self.results = []
        self._init_output_dir()

    def _init_output_dir(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def record(self, epoch, dataset_name, predictions, truths, scaler):
        """带编码自动检测的预测记录"""
        try:
            pred_denorm = scaler.inverse_transform(predictions)
            truth_denorm = scaler.inverse_transform(truths)
        except ValueError as e:
            print(f"反归一化错误：{str(e)}")
            return

        result_df = pd.DataFrame({
            'epoch': epoch,
            'time_step': range(len(pred_denorm)),
            'pred_tx': pred_denorm[:, 0],
            'pred_ty': pred_denorm[:, 1],
            'pred_tz': pred_denorm[:, 2],
            'true_tx': truth_denorm[:, 0],
            'true_ty': truth_denorm[:, 1],
            'true_tz': truth_denorm[:, 2],
            'dataset': dataset_name
        })

        file_path = self.output_dir / f"{dataset_name}_epoch{epoch}.csv"
        result_df.to_csv(file_path, index=False, encoding='utf_8_sig')  # 使用兼容编码
        self.results.append(result_df)
# ================== 训练父类定义（必须先于子类）==================
class RobustTrainingPipeline:
    def __init__(self, input_dim=12, hidden_dim=64, output_dim=3):
        self.model = EnhancedLSTM(input_dim, hidden_dim, output_dim)
        self.model.set_loss_func(VectorLossLayer.bottom_diff)
        self.results = []

    def run(self, datasets, epochs=500):
        """统一版训练流程"""
        if not datasets:
            raise ValueError("无有效训练数据")

        try:
            for epoch in range(epochs):
                total_loss = 0
                current_lr = 0.001 * (0.98 ** (epoch // 50))

                for dataset in datasets:
                    X, y, scaler, path = dataset
                    file_loss = self._train_file(X, y, current_lr)
                    total_loss += file_loss

                    if epoch % 100 == 0:
                        self._save_predictions(epoch, X, y, scaler, path)

                avg_loss = total_loss / len(datasets)
                print(f"Epoch {epoch} Avg Loss: {avg_loss:.4f}")

            return pd.concat(self.results)
        except KeyboardInterrupt:
            return pd.concat(self.results) if self.results else None

    def _train_file(self, X, y, lr):
        """统一训练方法"""
        if len(X) == 0:
            return 0.0

        file_loss = 0
        for i in range(len(X)):
            self.model.lstm_net.reset_states()
            pred = self.model.forward(X[i])
            loss = VectorLossLayer.loss(pred, y[i])
            self.model.backward(pred, y[i])
            self.model.apply_gradients(lr=lr, grad_clip=3.0)
            file_loss += loss
        return file_loss / len(X)

    def _save_predictions(self, epoch, X, y, scaler, path):
        """统一结果保存"""
        predictions = [self.model.forward(x) for x in X]
        if not predictions:
            return

        pred_denorm = scaler.inverse_transform(np.array(predictions))
        true_denorm = scaler.inverse_transform(y)

        result_df = pd.DataFrame({
            'epoch': [epoch] * len(pred_denorm),
            'pred_tx': pred_denorm[:, 0],
            'pred_ty': pred_denorm[:, 1],
            'pred_tz': pred_denorm[:, 2],
            'true_tx': true_denorm[:, 0],
            'true_ty': true_denorm[:, 1],
            'true_tz': true_denorm[:, 2],
            'source_file': Path(path).name
        })
        result_df.to_excel(f"results/{Path(path).stem}_epoch{epoch}.xlsx", index=False)
        self.results.append(result_df)
# ================== 增强训练子类（继承父类后定义）==================
class EnhancedTrainingPipeline(RobustTrainingPipeline):
    def __init__(self, input_dim=12, hidden_dim=64, output_dim=3):
        super().__init__(input_dim, hidden_dim, output_dim)
        self.recorder = PredictionRecorder()
        self.metrics = []

    def _train_file(self, X, y, lr, scaler, dataset_name):
        """带数据记录的训练"""
        file_loss = 0
        for i in range(len(X)):
            self.model.lstm_net.reset_states()
            pred = self.model.forward(X[i])
            loss = VectorLossLayer.loss(pred, y[i])

            self.recorder.record(
                epoch=self.current_epoch,
                dataset_name=dataset_name,
                predictions=pred.reshape(1, -1),
                truths=y[i].reshape(1, -1),
                scaler=scaler
            )

            self.model.backward(pred, y[i])
            self.model.apply_gradients(lr=lr, grad_clip=3.0)
            file_loss += loss
        return file_loss / (len(X) or 1)

    def run(self, datasets, epochs=500):
        """增强版训练流程"""
        try:
            for epoch in range(epochs):
                self.current_epoch = epoch
                total_loss = 0

                for dataset in datasets:
                    X, y, scaler, path = dataset
                    dataset_name = Path(path).stem
                    file_loss = self._train_file(X, y, 0.001, scaler, dataset_name)
                    total_loss += file_loss

                if epoch % 10 == 0:
                    combined_df = pd.concat(self.recorder.results)
                    rmse = np.sqrt(mean_squared_error(
                        combined_df[['pred_tx', 'pred_ty', 'pred_tz']],
                        combined_df[['true_tx', 'true_ty', 'true_tz']]
                    ))
                    self.metrics.append({'epoch': epoch, 'RMSE': rmse})

                print(f"Epoch {epoch} Avg Loss: {total_loss / len(datasets):.4f}")

            pd.DataFrame(self.metrics).to_csv("training_metrics.csv", index=False)
            return pd.concat(self.recorder.results)

        except KeyboardInterrupt:
            print("Training interrupted, saving partial results...")
            return pd.concat(self.recorder.results) if self.recorder.results else None
# ================== 数据预处理模块（添加编码自动检测）==================
if __name__ == "__main__":
    try:
        import chardet
    except ImportError:
        print("请先安装编码检测库：pip install chardet")
        exit(1)

    DATA_PATH = Path(r"G:\opencvinstall\pythonProject3\data\test")
    OUTPUT_DIR = Path("./csv_results")

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise RuntimeError(f"输出目录创建失败: {e}") from None

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"数据路径不存在: {DATA_PATH}")

    print("=== 开始CSV数据预处理 ===")
    try:
        datasets = CSVProcessor.process_folder(DATA_PATH)
        print(f"成功加载 {len(datasets)} 个有效数据集")
    except Exception as e:
        print(f"预处理失败: {str(e)}")
        exit(1)

    pipeline = EnhancedTrainingPipeline()

    print("\n=== 开始模型训练 ===")
    try:
        final_results = pipeline.run(datasets)
        if final_results is not None:
            final_metrics = final_results.groupby('dataset').apply(
                lambda x: pd.Series({
                    'Final_RMSE': np.sqrt(mean_squared_error(
                        x[['pred_tx', 'pred_ty', 'pred_tz']],
                        x[['true_tx', 'true_ty', 'true_tz']]
                    )),
                    'MAE': mean_absolute_error(
                        x[['pred_tx', 'pred_ty', 'pred_tz']],
                        x[['true_tx', 'true_ty', 'true_tz']]
                    )
                })
            )
            final_metrics.to_excel("final_metrics_summary.xlsx")
        print("=== 处理完成 ===")
    except Exception as e:
        print(f"训练异常终止: {str(e)}")
        exit(1)