```text
project             # 项目根目录
│  __init__.py      # 标识文件夹为 python 包
│
├─experience_config         # 参数配置文件夹
│  │  model_config.yaml     # 配置模型参数
│  │
│  └─real_data_config       # 为 real_data 配置参数
│          param_1_config.yaml  # dqn 训练和测试的参数
│
├─real_data
│  │  test.py               # dqn 训练启动文件
│  │  train.py              # dqn 测试启动文件
│  │  __init__.py           # 标识此文件目录为 python 包
│  │
│  ├─config
│  │  __init__.py           # 导入 project.utils 相关包，配置训练以及测试环境
│  │
│  └─utils                  # real_data 工具包
│      process.py           # 将数据转化为五元组（state，action，reward，next_state，done）供 train_env.py 使用
│      test_env.py          # 设置 dqn 测试环境
│      train_env.py         # 设置 dqn 训练环境
│      __init__.py          # 向 train.py 和 test.py 提供环境
│
├─utils                     # 工具包
│  │  __init__.py           # 向外部提供相应的功能
│  │
│  ├─core
│  │  │  __init__.py        # 向外部提供内部类调用
│  │  │
│  │  ├─base                    # 基类继承自 interface 中提供的接口，没有完全实现接口中的抽象方法
│  │  │  action_manager.py      # 继承自 interface 中 action_manager.py 提供的类
│  │  │  data_processor.py      # 继承自 interface 中 data_processor.py 提供的类
│  │  │  reward_calculator.py   # 继承自 interface 中 reward_calculator.py 提供的类
│  │  │  rl_agent.py            # 继承自 interface 中 rl_agent.py 提供的类
│  │  │  rl_algorithm.py        # 继承自 interface 中 rl_algorithm.py 提供的类
│  │  │  state_manager.py       # 继承自 interface 中 state_manager.py 提供的类
│  │  │  __init__.py
│  │  │
│  │  └─interface               # 抽象类接口
│  │     action_manager.py      # 提供动作管理的接口
│  │     data_processor.py      # 提供数据管理的接口
│  │     reward_calculator.py   # 提供奖励管理的接口
│  │     rl_agent.py            # 提供 dqn 智能体的接口
│  │     rl_algorithm.py        # 提供 dqn 算法实现的接口
│  │     state_manager.py       # 提供状态管理的接口
│  │     __init__.py
│  │
│  ├─impl                       # 继承自 base 中的基类，具体实现部分功能
│  │  │  __init__.py            # 向外部提供内部类调用
│  │  │
│  │  ├─action                  # 继承自 base.action_manager 实现动作管理
│  │  │  action_impl.py         # 基础动作类的具体实现
│  │  │  __init__.py
│  │  │
│  │  │
│  │  ├─net                     # 无继承，实现神经网络类
│  │  │  dueling_impl.py        # 实现 dueling dqn 神经网络
│  │  │  mlp_impl.py            # 实现基础的 dqn mlp 网络
│  │  │  __init__.py            # 向外部提供内部类调用
│  │  │
│  │  ├─processor               # 继承自 base.data_processor 实现数据处理
│  │  │  minmax_processor.py    # 最大最小，归一化特征的具体实现
│  │  │  normal_processor.py    # 标准化特征的具体实现
│  │  │  no_processor.py        # 保持原始特征，不做处理
│  │  │  __init__.py            # 向外部提供内部类调用
│  │  │
│  │  ├─reward                  # 继承自 base.reward_calculator 实现奖励管理
│  │  │  reward_impl.py         # 基础奖励计算类的具体实现
│  │  │  __init__.py            # 向外部提供内部类调用
│  │  │
│  │  │
│  │  └─state                   # 继承自 base.state_manager 实现状态
│  │     state_impl.py          # 基础状态类的具体实现
│  │     __init__.py            # 向外部提供内部类调用
│  │
│  ├─rl                         # 强化学习包
│  │  │  dqn.py                 # dqn 的具体实现类
│  │  │  __init__.py            # 向外部提供内部类调用
│  │  │
│  │  ├─dqn_impl                # 继承自 base
│  │  │  dqn_agent.py           # 继承自 base.rl_agent 中的类，不实现任何东西
│  │  │  dqn_algorithm.py       # 继承自 base.rl_algorithm 中的类，使用策略委托的方式实现 dqn 基本算法
│  │  │  __init__.py            # 向外部提供内部类调用
│  │  │
│  │  ├─dqn_strategy            # dqn 委托类
│  │  │  base.py                # dqn 委托类基类，提供接口，不实现任何方法
│  │  │  ddqn.py                # 继承 bease.py 中提供的基类，使用 dqn_algorithm.py 中提供的委托方法装饰器，实现 ddqn 逻辑
│  │  │  pri_buffer.py          # 继承 bease.py 中提供的基类，使用 dqn_algorithm.py 中提供的委托方法装饰器，实现优先经验池逻辑
│  │  │  soft_update.py         # 继承 bease.py 中提供的基类，使用 dqn_algorithm.py 中提供的委托方法装饰器，实现软更新逻辑
│  │  │  strategy.py            # 继承 bease.py 中提供的基类，使用 dqn_algorithm.py 中提供的委托方法装饰器，实现基本的 dqn 逻辑
│  │  │  __init__.py            # 向外部提供内部类调用
│  │  │
│  │  ├─exp_replay              # 经验回放类
│  │  │  base.py                # 基础经验回放池的具体实现
│  │  │  pro_pri.py             # 优先经验回放的具体实现
│  │  │  replay_memory.py       # 经验回放工厂模式，根据参数选择使用基础经验回放或是优先经验回放
│  │  │  __init__.py            # 向外部提供内部类调用
│  │  │
│  │  └─net_impl                # 网络类
│  │     dqn_net.py             # dqn 神经网络工厂模式，根据参数选择使用基础神经网络或是 dueling 神经网络
│  │     __init__.py_           # 向外部提供内部类调用
│  │
└─ └─tool                       # 提供一些工具类
      argument_parser.py
      config_loader.py
      data_record.py
      diff.py
      draw_figure.py
      exception_check.py
       __init__.py
```

### 2025.4.9
**修复了**
- 修改 dqn 神经网络结构，可选神经网络参数 ["mlp", "dueling"]

**新增了**
- 新增配置训练次数选项
- config_loader.py 新增 get 方法
- 新增 dueling dqn 方法
- 新增工厂模式用于构建 dqn 神经网络，通过参数调整使用的神经网络
- 状态中可选择上一次出口浓度
- 新增优先经验回放
- 新增工厂模式用于构建 dqn 经验池，通过参数调整使用的经验池
- 新增策略委托方式。构建 dqn、ddqn、软更新、优先经验回放的策略
- 新增参数配置项

**移除了**


**优化了**
- 优化一些逻辑

### 2025.4.16
**修复了**
- 修复绘图工具函数的缺失

**新增了**
- 增加数据处理的工厂方式


**移除了**
- model_config.yaml 文件

**优化了**
- 优化一些逻辑
