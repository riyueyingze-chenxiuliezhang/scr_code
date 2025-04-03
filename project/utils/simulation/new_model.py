"""
# @Time: 2025/3/27 16:51
# @File: new_model.py
"""
import joblib
import numpy as np
import torch
import torch.nn as nn


class EnvNet(nn.Module):
    """ 环境模型架构 """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 512),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


class SimuModelN:
    """ 加载环境模型参数 """
    def __init__(self, model_path, scaler_x_path, scaler_y_path):
        # 初始化原始模型结构
        self.env_model = EnvNet()
        self.env_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.env_model.eval()

        # 加载标准化器
        self.scaler_x = joblib.load(scaler_x_path)
        self.scaler_y = joblib.load(scaler_y_path)

        # 特征配置
        self.feature_keys = [
            '入口NO2浓度（折算）',
            'CEM_脱硝入口烟气流量（工况）',
            'GGH原烟气侧出口温度',
            '焦炉煤气阀门开度',
            '煤气压力1热风炉气动阀1前'
        ]
        self.prev_outlet_c = None  # 前时刻出口浓度真实值

    def predict(self, current_features, predicted_prev=None):
        """预测方法"""
        try:
            # 特征校验
            if missing := [k for k in self.feature_keys if k not in current_features]:
                raise ValueError(f"缺少特征: {missing}")

            # 构建特征向量
            features = [current_features[k] for k in self.feature_keys]
            lag_feature = predicted_prev if predicted_prev is not None else (
                self.prev_outlet_c if self.prev_outlet_c is not None else self.scaler_x.mean_[-1]
            )

            # 标准化预测
            raw_features = np.array(features + [lag_feature], dtype=np.float32)
            scaled = self.scaler_x.transform([raw_features])
            tensor = torch.tensor(scaled, dtype=torch.float32)

            # 执行预测
            with torch.no_grad():
                predict = self.env_model(tensor).numpy()

            # 反标准化并更新状态
            result = self.scaler_y.inverse_transform(predict).item()
            self.prev_outlet_c = predicted_prev if predicted_prev is not None else result
            return result

        except Exception as e:
            print(f"预测失败: {str(e)}")
            self.prev_outlet_c = None
            return None

    def reset_state(self):
        """重置前时刻出口浓度"""
        self.prev_outlet_c = None