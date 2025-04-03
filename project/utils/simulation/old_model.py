"""
# @Time: 2025/3/27 16:51
# @File: old_model.py
"""
import joblib
import pandas as pd


class SimuEnvO:
    """ 加载环境模型参数 """
    def __init__(self, model_path, scaler_path):
        # 加载模型
        self.env_model = joblib.load(model_path)

        # 加载标准化器
        self.scaler = joblib.load(scaler_path)

    @staticmethod
    def _calculate_single_feature(data_point):
        """针对单个数据点计算特征（严格匹配训练时逻辑，导数置0）"""
        df = pd.DataFrame([data_point], columns=[
            '焦炉煤气阀门开度', '煤气压力1热风炉气动阀1前',
            'GGH原烟气侧出口温度', 'CEM_脱硝入口烟气流量（工况）',
            '入口NO2浓度（折算）'
        ])

        # 显式设置所有导数特征为0（无需依赖历史数据）
        for col in df.columns:
            df[f'{col}_一阶导'] = 0
            df[f'{col}_二阶导'] = 0

        # 保留与训练阶段完全相同的特征生成逻辑（包括空值处理）
        # 注意：此处直接返回包含所有必要特征的DataFrame
        return df

    def predict(self, data_point):
        """单个数据点预测函数（严格匹配训练时特征顺序）
        Returns:

        """
        # 明确特征顺序（必须与训练时完全一致）
        base_features = [
            '焦炉煤气阀门开度', '煤气压力1热风炉气动阀1前',
            'GGH原烟气侧出口温度', 'CEM_脱硝入口烟气流量（工况）',
            '入口NO2浓度（折算）'
        ]
        derivative_features = [
                                  f'{col}_一阶导' for col in base_features
                              ] + [f'{col}_二阶导' for col in base_features]

        required_features = base_features + derivative_features

        # 处理单个数据点的特征
        processed_data = self._calculate_single_feature(data_point)

        # 验证特征完整性（必须包含所有训练阶段使用的特征）
        if not all(col in processed_data.columns for col in required_features):
            missing = set(required_features) - set(processed_data.columns)
            raise KeyError(f"缺失必要特征列: {missing}")

        # 标准化并预测
        x_scaled = self.scaler.transform(processed_data[required_features])
        prediction = self.env_model.predict(x_scaled)  # 直接使用 NumPy 数组，移除特征名称

        return prediction[0]
