# LAB 3

#### 1. And 2.
```python

for i, config in enumerate(search_spaces):
    mg, _ = quantize_transform_pass(mg, config)
    j = 0
    
    #MODIFIED
    # Measure model size after quantization
    temp_model_path = "temp_model.pth"
    torch.save(mg.model.state_dict(), temp_model_path)
    model_size = os.path.getsize(temp_model_path) / (1024 * 1024)  # Size in MB
    os.remove(temp_model_path)  # Clean up the temporary file

    # this is the inner loop, where we also call it as a runner.
    acc_avg, loss_avg = 0, 0
    accs, losses = [], []
    latencies = []
    for inputs in data_module.train_dataloader():
        xs, ys = inputs
        
        #MODIFIED
        #latency
        start_time = time.time()
        preds = mg.model(xs)
        end_time = time.time()
        latencies.append(end_time - start_time)
        #accuracy; loss
        loss = torch.nn.functional.cross_entropy(preds, ys)
        acc = metric(preds, ys)
        accs.append(acc)
        losses.append(loss)
        if j > num_batchs:
            break
        j += 1

        
    acc_avg = sum(accs) / len(accs)
    loss_avg = sum(losses) / len(losses)
    latency_avg = sum(latencies) / len(latencies)
    recorded_accs.append(acc_avg)
    
    #print(acc_avg)
    #print(loss_avg)
    #print((end_time - start_time))

    print(f"Accuracy: {acc_avg:.4g}, Loss: {loss_avg:.4g}, Latency: {latency_avg} seconds")
```
OUTPUT:
```
Accuracy: 0.3004, Loss: 1.526, Latency: 0.00023865699768066406 seconds
Accuracy: 0.2512, Loss: 1.555, Latency: 0.00022520337785993303 seconds
Accuracy: 0.1976, Loss: 1.626, Latency: 0.00021123886108398438 seconds
Accuracy: 0.07143, Loss: 1.632, Latency: 0.00021542821611676897 seconds
Accuracy: 0.2024, Loss: 1.554, Latency: 0.00021682466779436384 seconds
Accuracy: 0.306, Loss: 1.524, Latency: 0.00022833687918526785 seconds
Accuracy: 0.1143, Loss: 1.633, Latency: 0.00021420206342424666 seconds
Accuracy: 0.2333, Loss: 1.551, Latency: 0.00020994458879743303 seconds
Accuracy: 0.2315, Loss: 1.559, Latency: 0.0002005781446184431 seconds
Accuracy: 0.2738, Loss: 1.531, Latency: 0.00019325528826032366 seconds
Accuracy: 0.09524, Loss: 1.6, Latency: 0.0002029282706124442 seconds
Accuracy: 0.3968, Loss: 1.511, Latency: 0.00020599365234375 seconds
Accuracy: 0.2536, Loss: 1.567, Latency: 0.00019689968654087612 seconds
Accuracy: 0.2119, Loss: 1.59, Latency: 0.0002951622009277344 seconds
Accuracy: 0.2702, Loss: 1.583, Latency: 0.00019540105547223772 seconds
Accuracy: 0.3304, Loss: 1.529, Latency: 0.00020762852260044644 seconds
```
In this classification tasks, accuracy and loss are as key performance indicators. Accuracy reflects the proportion of correct predictions, serving as a straightforward measure of a model's effectiveness. Loss, on the other hand, quantifies the model's prediction errors, offering a insight into its predictive precision. There is a positive correlation in general exists between these metrics: as accuracy improves, loss diminishes. This relationship highlights a fundamental aspect of machine learning model optimization, where enhancing predictive accuracy intrinsically leads to reduced prediction errors which is loss. Therefore, they can be treat as one metrics.

#### 3.
In /search_space/strategies/software/Optuna.py add brute-force sampler
```python
def sampler_map(self, name):
        match name.lower():
            case "brute-force":
                sampler = optuna.samplers.BruteForceSampler()
            case "random":
                sampler = optuna.samplers.RandomSampler()
            case "tpe":
                sampler = optuna.samplers.TPESampler()
            case "nsgaii":
                sampler = optuna.samplers.NSGAIISampler()
            case "nsgaiii":
                sampler = optuna.samplers.NSGAIIISampler()
            case "qmc":
                sampler = optuna.samplers.QMCSampler()
            case _:
                raise ValueError(f"Unknown sampler name: {name}")
        return sampler


```
and change the sampler name in jsc_toy_by_type.toml file

```toml

[search.strategy.setup]
n_jobs = 1
n_trials = 20
timeout = 20000
sampler = "tpe"
#sampler = "brute-force"
# sum_scaled_metrics = true # single objective
# direction = "maximize"
sum_scaled_metrics = false # multi objective

```




#### 4.

The result:

```
Brute-force：

INFO     Building search space...
INFO     Search started...
/home/lch121600/ADLSlab/mase/machop/chop/actions/search/strategies/optuna.py:47: ExperimentalWarning: BruteForceSampler is experimental (supported from v3.1.0). The interface can change in the future.
  sampler = optuna.samplers.BruteForceSampler()
 90%|███████████████████████████████████████████████████████▊      | 
 18/20 [00:13<00:01,  1.31it/s, 13.74/20000 seconds]
INFO     Best trial(s):
Best trial(s):
|    |   number | software_metrics                   | hardware_metrics                                  | scaled_metrics                               |
|----+----------+------------------------------------+---------------------------------------------------+----------------------------------------------|
|  0 |        1 | {'loss': 1.506, 'accuracy': 0.388} | {'average_bitwidth': 2.0, 'memory_density': 16.0} | {'accuracy': 0.388, 'average_bitwidth': 0.4} |
INFO     Searching is completed


-----------------------------------------------------------------
TPE：

INFO     Building search space...
INFO     Search started...
100%|██████████████████████████████████████████████████████████████| 
20/20 [00:15<00:00,  1.30it/s, 15.39/20000 seconds]
INFO     Best trial(s):
Best trial(s):
|    |   number | software_metrics                   | hardware_metrics                                  | scaled_metrics                               |
|----+----------+------------------------------------+---------------------------------------------------+----------------------------------------------|
|  0 |        0 | {'loss': 1.476, 'accuracy': 0.401} | {'average_bitwidth': 8.0, 'memory_density': 4.0}  | {'accuracy': 0.401, 'average_bitwidth': 1.6} |
|  1 |        2 | {'loss': 1.495, 'accuracy': 0.388} | {'average_bitwidth': 4.0, 'memory_density': 8.0}  | {'accuracy': 0.388, 'average_bitwidth': 0.8} |
|  2 |        6 | {'loss': 1.498, 'accuracy': 0.383} | {'average_bitwidth': 2.0, 'memory_density': 16.0} | {'accuracy': 0.383, 'average_bitwidth': 0.4} |

```
In terms of sample effeciency, for this limit 18 possible combination task the sample effeciency of brute-force is better than the TPE is cost less time when finished the search. The brute-force cost 13.74 seconds and TPE costs 15.39 seconds. In this task we defined 20 trials for searching, therefore TPE do all 20 trails but brute-force only do 18 becasue of the limit combinations 2 x 3 x 3.

In terms of performance, the TPE is better than the brute-force. It find the highest accuracy and lowest loss one when tring and optimizting the possibly optimal search which is 'loss': 1.476, 'accuracy': 0.401 




# LAB4

#### 1.

```python
from chop.passes.graph import report_graph_analysis_pass

pass_config = {
"by": "name",
"default": {"config": {"name": None}},

"seq_blocks_2": {
    "config": {
        "name": "output_only",
        "channel_multiplier": 2,
        }
    },
"seq_blocks_4": {
    "config": {
        "name": "both",
        "channel_multiplier": 2,
        }
    },
"seq_blocks_6": {
    "config": {
        "name": "input_only",
        "channel_multiplier": 2,
        }
    },
}
```
By changing the pass_config can enalbe the model double sized. Output as following:

```
GraphModule(
  (seq_blocks): Module(
    (0): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): ReLU()
    (2): Linear(in_features=16, out_features=32, bias=True)
    (3): ReLU()
    (4): Linear(in_features=32, out_features=32, bias=True)
    (5): ReLU()
    (6): Linear(in_features=32, out_features=5, bias=True)
    (7): ReLU(inplace=True)
  )
)
```

#### 2. and 3.

The following code is a modified code. The code is updated by adding a intermedium variable: `last_multi` to record the `channel_multiplier` of the previous block's output feature and then set it to current blocks's input features to make sure the output from the last layers is matched with the input of the next layers.

The code block below is a pass_config setup and a build of search space:

```python
pass_config = {
"by": "name",
"default": {"config": {"name": None}},
"seq_blocks_2": {
    "config": {
        "name": "output_only",
        "channel_multiplier": 2,
        }
    },
"seq_blocks_4": {
    "config": {
        "name": "output_only",
        "channel_multiplier": 2,
        }
    },
"seq_blocks_6": {
    "config": {
        "name": "input_only",
        }
    },
}

import copy
# build a search space
channel_multiplier_2 = [1,2,4]
channel_multiplier_4 = [1,2,4]
channel_multiplier_6 = [1,2,4]
search_spaces = []
for c_config in channel_multiplier_2:
    for b_config in channel_multiplier_4:
           pass_config['seq_blocks_2']['config']['channel_multiplier'] = c_config
           pass_config['seq_blocks_4']['config']['channel_multiplier'] = b_config
           pass_config['seq_blocks_6']['config']['channel_multiplier'] = b_config
        # dict.copy() and dict(dict) only perform shallow copies
        # in fact, only primitive data types in python are doing implicit copy when a = b happens
           search_spaces.append(copy.deepcopy(pass_config))
           print(pass_config)
```

This block shows the modified `redefine_linear_transform_pass` function.

```python

def instantiate_linear(in_features, out_features, bias):
    if bias is not None:
        bias = True
    return nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias)

def redefine_linear_transform_pass(graph, pass_args=None):
    main_config = pass_args.pop('config')
    print(main_config)
    default = main_config.pop('default', None)
    print(default)
    if default is None:
        print(default)
        raise ValueError(f"default value must be provided.")
    i = 0
    last_multi = 1
    for node in graph.fx_graph.nodes:
        i += 1
        # if node name is not matched, it won't be tracked
        config = main_config.get(node.name, default)['config']
        name = config.get("name", None)
        if name is not None:
            ori_module = graph.modules[node.target]
            in_features = ori_module.in_features
            out_features = ori_module.out_features
            bias = ori_module.bias
            if name == "output_only":
                out_features = out_features * config["channel_multiplier"]
                in_features = in_features*last_multi
                last_multi = config["channel_multiplier"]
            elif name == "both":
                in_features = in_features * config["channel_multiplier"]
                out_features = out_features * config["channel_multiplier"]
            elif name == "input_only":
                in_features = in_features * last_multi
                

            new_module = instantiate_linear(in_features, out_features, bias)
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
    _ = report_graph_analysis_pass(mg)
    return graph, {}
```

Do the grid search:

```python

mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)

metric = MulticlassAccuracy(num_classes=5)
num_batchs = 5
# This first loop is basically our search strategy,
# in this case, it is a simple brute force search

recorded_accs = []
for i, config in enumerate(search_spaces):
    mg, _ = redefine_linear_transform_pass(graph=mg, pass_args={"config": config})
    j = 0

    # this is the inner loop, where we also call it as a runner.
    acc_avg, loss_avg = 0, 0
    accs, losses = [], []
    for inputs in data_module.train_dataloader():
        xs, ys = inputs
        preds = mg.model(xs)
        loss = torch.nn.functional.cross_entropy(preds, ys)
        acc = metric(preds, ys)
        accs.append(acc)
        losses.append(loss)
        if j > num_batchs:
            break
        j += 1
    acc_avg = sum(accs) / len(accs)
    loss_avg = sum(losses) / len(losses)
    recorded_accs.append(acc_avg)
    print(acc_avg)
```

The result:

```
graph():
    %x : [num_users=1] = placeholder[target=x]
    %seq_blocks_0 : [num_users=1] = call_module[target=seq_blocks.0](args = (%x,), kwargs = {})
    %seq_blocks_1 : [num_users=1] = call_module[target=seq_blocks.1](args = (%seq_blocks_0,), kwargs = {})
    %seq_blocks_2 : [num_users=1] = call_module[target=seq_blocks.2](args = (%seq_blocks_1,), kwargs = {})
    %seq_blocks_3 : [num_users=1] = call_module[target=seq_blocks.3](args = (%seq_blocks_2,), kwargs = {})
    %seq_blocks_4 : [num_users=1] = call_module[target=seq_blocks.4](args = (%seq_blocks_3,), kwargs = {})
    %seq_blocks_5 : [num_users=1] = call_module[target=seq_blocks.5](args = (%seq_blocks_4,), kwargs = {})
    %seq_blocks_6 : [num_users=1] = call_module[target=seq_blocks.6](args = (%seq_blocks_5,), kwargs = {})
    %seq_blocks_7 : [num_users=1] = call_module[target=seq_blocks.7](args = (%seq_blocks_6,), kwargs = {})
    return seq_blocks_7Network overview:
{'placeholder': 1, 'get_attr': 0, 'call_function': 0, 'call_method': 0, 'call_module': 8, 'output': 1}
Layer types:
[BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(), Linear(in_features=16, out_features=16, bias=True), ReLU(), Linear(in_features=16, out_features=16, bias=True), ReLU(), Linear(in_features=16, out_features=5, bias=True), ReLU(inplace=True)]
accuracy
tensor(0.2024)
graph():
    %x : [num_users=1] = placeholder[target=x]
    %seq_blocks_0 : [num_users=1] = call_module[target=seq_blocks.0](args = (%x,), kwargs = {})
    %seq_blocks_1 : [num_users=1] = call_module[target=seq_blocks.1](args = (%seq_blocks_0,), kwargs = {})
    %seq_blocks_2 : [num_users=1] = call_module[target=seq_blocks.2](args = (%seq_blocks_1,), kwargs = {})
    %seq_blocks_3 : [num_users=1] = call_module[target=seq_blocks.3](args = (%seq_blocks_2,), kwargs = {})
    %seq_blocks_4 : [num_users=1] = call_module[target=seq_blocks.4](args = (%seq_blocks_3,), kwargs = {})
    %seq_blocks_5 : [num_users=1] = call_module[target=seq_blocks.5](args = (%seq_blocks_4,), kwargs = {})
    %seq_blocks_6 : [num_users=1] = call_module[target=seq_blocks.6](args = (%seq_blocks_5,), kwargs = {})
    %seq_blocks_7 : [num_users=1] = call_module[target=seq_blocks.7](args = (%seq_blocks_6,), kwargs = {})
    return seq_blocks_7Network overview:
{'placeholder': 1, 'get_attr': 0, 'call_function': 0, 'call_method': 0, 'call_module': 8, 'output': 1}
Layer types:
[BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(), Linear(in_features=16, out_features=16, bias=True), ReLU(), Linear(in_features=16, out_features=32, bias=True), ReLU(), Linear(in_features=32, out_features=5, bias=True), ReLU(inplace=True)]
accuracy
tensor(0.1214)
graph():
    %x : [num_users=1] = placeholder[target=x]
    %seq_blocks_0 : [num_users=1] = call_module[target=seq_blocks.0](args = (%x,), kwargs = {})
    %seq_blocks_1 : [num_users=1] = call_module[target=seq_blocks.1](args = (%seq_blocks_0,), kwargs = {})
    %seq_blocks_2 : [num_users=1] = call_module[target=seq_blocks.2](args = (%seq_blocks_1,), kwargs = {})
    %seq_blocks_3 : [num_users=1] = call_module[target=seq_blocks.3](args = (%seq_blocks_2,), kwargs = {})
    %seq_blocks_4 : [num_users=1] = call_module[target=seq_blocks.4](args = (%seq_blocks_3,), kwargs = {})
    %seq_blocks_5 : [num_users=1] = call_module[target=seq_blocks.5](args = (%seq_blocks_4,), kwargs = {})
    %seq_blocks_6 : [num_users=1] = call_module[target=seq_blocks.6](args = (%seq_blocks_5,), kwargs = {})
    %seq_blocks_7 : [num_users=1] = call_module[target=seq_blocks.7](args = (%seq_blocks_6,), kwargs = {})
    return seq_blocks_7Network overview:
{'placeholder': 1, 'get_attr': 0, 'call_function': 0, 'call_method': 0, 'call_module': 8, 'output': 1}
Layer types:
[BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(), Linear(in_features=16, out_features=16, bias=True), ReLU(), Linear(in_features=16, out_features=128, bias=True), ReLU(), Linear(in_features=128, out_features=5, bias=True), ReLU(inplace=True)]
accuracy
tensor(0.2726)
graph():
    %x : [num_users=1] = placeholder[target=x]
    %seq_blocks_0 : [num_users=1] = call_module[target=seq_blocks.0](args = (%x,), kwargs = {})
    %seq_blocks_1 : [num_users=1] = call_module[target=seq_blocks.1](args = (%seq_blocks_0,), kwargs = {})
    %seq_blocks_2 : [num_users=1] = call_module[target=seq_blocks.2](args = (%seq_blocks_1,), kwargs = {})
    %seq_blocks_3 : [num_users=1] = call_module[target=seq_blocks.3](args = (%seq_blocks_2,), kwargs = {})
    %seq_blocks_4 : [num_users=1] = call_module[target=seq_blocks.4](args = (%seq_blocks_3,), kwargs = {})
    %seq_blocks_5 : [num_users=1] = call_module[target=seq_blocks.5](args = (%seq_blocks_4,), kwargs = {})
    %seq_blocks_6 : [num_users=1] = call_module[target=seq_blocks.6](args = (%seq_blocks_5,), kwargs = {})
    %seq_blocks_7 : [num_users=1] = call_module[target=seq_blocks.7](args = (%seq_blocks_6,), kwargs = {})
    return seq_blocks_7Network overview:
{'placeholder': 1, 'get_attr': 0, 'call_function': 0, 'call_method': 0, 'call_module': 8, 'output': 1}
Layer types:
[BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(), Linear(in_features=16, out_features=32, bias=True), ReLU(), Linear(in_features=32, out_features=128, bias=True), ReLU(), Linear(in_features=128, out_features=5, bias=True), ReLU(inplace=True)]
accuracy
tensor(0.2119)
graph():
    %x : [num_users=1] = placeholder[target=x]
    %seq_blocks_0 : [num_users=1] = call_module[target=seq_blocks.0](args = (%x,), kwargs = {})
    %seq_blocks_1 : [num_users=1] = call_module[target=seq_blocks.1](args = (%seq_blocks_0,), kwargs = {})
    %seq_blocks_2 : [num_users=1] = call_module[target=seq_blocks.2](args = (%seq_blocks_1,), kwargs = {})
    %seq_blocks_3 : [num_users=1] = call_module[target=seq_blocks.3](args = (%seq_blocks_2,), kwargs = {})
    %seq_blocks_4 : [num_users=1] = call_module[target=seq_blocks.4](args = (%seq_blocks_3,), kwargs = {})
    %seq_blocks_5 : [num_users=1] = call_module[target=seq_blocks.5](args = (%seq_blocks_4,), kwargs = {})
    %seq_blocks_6 : [num_users=1] = call_module[target=seq_blocks.6](args = (%seq_blocks_5,), kwargs = {})
    %seq_blocks_7 : [num_users=1] = call_module[target=seq_blocks.7](args = (%seq_blocks_6,), kwargs = {})
    return seq_blocks_7Network overview:
{'placeholder': 1, 'get_attr': 0, 'call_function': 0, 'call_method': 0, 'call_module': 8, 'output': 1}
Layer types:
[BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(), Linear(in_features=16, out_features=64, bias=True), ReLU(), Linear(in_features=64, out_features=256, bias=True), ReLU(), Linear(in_features=256, out_features=5, bias=True), ReLU(inplace=True)]
accuracy
tensor(0.1061)
graph():
    %x : [num_users=1] = placeholder[target=x]
    %seq_blocks_0 : [num_users=1] = call_module[target=seq_blocks.0](args = (%x,), kwargs = {})
    %seq_blocks_1 : [num_users=1] = call_module[target=seq_blocks.1](args = (%seq_blocks_0,), kwargs = {})
    %seq_blocks_2 : [num_users=1] = call_module[target=seq_blocks.2](args = (%seq_blocks_1,), kwargs = {})
    %seq_blocks_3 : [num_users=1] = call_module[target=seq_blocks.3](args = (%seq_blocks_2,), kwargs = {})
    %seq_blocks_4 : [num_users=1] = call_module[target=seq_blocks.4](args = (%seq_blocks_3,), kwargs = {})
    %seq_blocks_5 : [num_users=1] = call_module[target=seq_blocks.5](args = (%seq_blocks_4,), kwargs = {})
    %seq_blocks_6 : [num_users=1] = call_module[target=seq_blocks.6](args = (%seq_blocks_5,), kwargs = {})
    %seq_blocks_7 : [num_users=1] = call_module[target=seq_blocks.7](args = (%seq_blocks_6,), kwargs = {})
    return seq_blocks_7Network overview:
{'placeholder': 1, 'get_attr': 0, 'call_function': 0, 'call_method': 0, 'call_module': 8, 'output': 1}
Layer types:
[BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(), Linear(in_features=16, out_features=128, bias=True), ReLU(), Linear(in_features=128, out_features=1024, bias=True), ReLU(), Linear(in_features=1024, out_features=5, bias=True), ReLU(inplace=True)]
accuracy
tensor(0.1825)
graph():
    %x : [num_users=1] = placeholder[target=x]
    %seq_blocks_0 : [num_users=1] = call_module[target=seq_blocks.0](args = (%x,), kwargs = {})
    %seq_blocks_1 : [num_users=1] = call_module[target=seq_blocks.1](args = (%seq_blocks_0,), kwargs = {})
    %seq_blocks_2 : [num_users=1] = call_module[target=seq_blocks.2](args = (%seq_blocks_1,), kwargs = {})
    %seq_blocks_3 : [num_users=1] = call_module[target=seq_blocks.3](args = (%seq_blocks_2,), kwargs = {})
    %seq_blocks_4 : [num_users=1] = call_module[target=seq_blocks.4](args = (%seq_blocks_3,), kwargs = {})
    %seq_blocks_5 : [num_users=1] = call_module[target=seq_blocks.5](args = (%seq_blocks_4,), kwargs = {})
    %seq_blocks_6 : [num_users=1] = call_module[target=seq_blocks.6](args = (%seq_blocks_5,), kwargs = {})
    %seq_blocks_7 : [num_users=1] = call_module[target=seq_blocks.7](args = (%seq_blocks_6,), kwargs = {})
    return seq_blocks_7Network overview:
{'placeholder': 1, 'get_attr': 0, 'call_function': 0, 'call_method': 0, 'call_module': 8, 'output': 1}
Layer types:
[BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(), Linear(in_features=16, out_features=512, bias=True), ReLU(), Linear(in_features=512, out_features=1024, bias=True), ReLU(), Linear(in_features=1024, out_features=5, bias=True), ReLU(inplace=True)]
accuracy
tensor(0.2347)
graph():
    %x : [num_users=1] = placeholder[target=x]
    %seq_blocks_0 : [num_users=1] = call_module[target=seq_blocks.0](args = (%x,), kwargs = {})
    %seq_blocks_1 : [num_users=1] = call_module[target=seq_blocks.1](args = (%seq_blocks_0,), kwargs = {})
    %seq_blocks_2 : [num_users=1] = call_module[target=seq_blocks.2](args = (%seq_blocks_1,), kwargs = {})
    %seq_blocks_3 : [num_users=1] = call_module[target=seq_blocks.3](args = (%seq_blocks_2,), kwargs = {})
    %seq_blocks_4 : [num_users=1] = call_module[target=seq_blocks.4](args = (%seq_blocks_3,), kwargs = {})
    %seq_blocks_5 : [num_users=1] = call_module[target=seq_blocks.5](args = (%seq_blocks_4,), kwargs = {})
    %seq_blocks_6 : [num_users=1] = call_module[target=seq_blocks.6](args = (%seq_blocks_5,), kwargs = {})
    %seq_blocks_7 : [num_users=1] = call_module[target=seq_blocks.7](args = (%seq_blocks_6,), kwargs = {})
    return seq_blocks_7Network overview:
{'placeholder': 1, 'get_attr': 0, 'call_function': 0, 'call_method': 0, 'call_module': 8, 'output': 1}
Layer types:
[BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(), Linear(in_features=16, out_features=2048, bias=True), ReLU(), Linear(in_features=2048, out_features=2048, bias=True), ReLU(), Linear(in_features=2048, out_features=5, bias=True), ReLU(inplace=True)]
accuracy
tensor(0.0839)
graph():
    %x : [num_users=1] = placeholder[target=x]
    %seq_blocks_0 : [num_users=1] = call_module[target=seq_blocks.0](args = (%x,), kwargs = {})
    %seq_blocks_1 : [num_users=1] = call_module[target=seq_blocks.1](args = (%seq_blocks_0,), kwargs = {})
    %seq_blocks_2 : [num_users=1] = call_module[target=seq_blocks.2](args = (%seq_blocks_1,), kwargs = {})
    %seq_blocks_3 : [num_users=1] = call_module[target=seq_blocks.3](args = (%seq_blocks_2,), kwargs = {})
    %seq_blocks_4 : [num_users=1] = call_module[target=seq_blocks.4](args = (%seq_blocks_3,), kwargs = {})
    %seq_blocks_5 : [num_users=1] = call_module[target=seq_blocks.5](args = (%seq_blocks_4,), kwargs = {})
    %seq_blocks_6 : [num_users=1] = call_module[target=seq_blocks.6](args = (%seq_blocks_5,), kwargs = {})
    %seq_blocks_7 : [num_users=1] = call_module[target=seq_blocks.7](args = (%seq_blocks_6,), kwargs = {})
    return seq_blocks_7Network overview:
{'placeholder': 1, 'get_attr': 0, 'call_function': 0, 'call_method': 0, 'call_module': 8, 'output': 1}
Layer types:
[BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(), Linear(in_features=16, out_features=8192, bias=True), ReLU(), Linear(in_features=8192, out_features=8192, bias=True), ReLU(), Linear(in_features=8192, out_features=5, bias=True), ReLU(inplace=True)]
accuracy
tensor(0.2571)
```

From the result we can see the model of the highest accuracy (27%) is:

```python
GraphModule(
  (seq_blocks): Module(
    (0): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): ReLU()
    (2): Linear(in_features=16, out_features=16, bias=True)
    (3): ReLU()
    (4): Linear(in_features=16, out_features=128, bias=True)
    (5): ReLU()
    (6): Linear(in_features=128, out_features=5, bias=True)
    (7): ReLU(inplace=True)
  )
)
```

#### 4.

To implement it in chop flow and use command to run it. We need to:

* Add a new search space to `machop/chop/actions/search/search_space`. 
  I creat in`search/search_space/LAB4/graph_lab4.py`. In this file, the transform pass function `redefine_linear_transform_pass` is created inside the  file.

* Modified `.toml file` and creat a new one in `machop/configs/examples`. Mine is `.../mase/machop/configs/examples/lab4_type.toml `. The following is the `lab4_type.toml` file.

```python
# basics
model = "jsc-three-linear-layers"
dataset = "jsc"
task = "cls"

max_epochs = 5
batch_size = 512
learning_rate = 1e-2
accelerator = "gpu"
project = "jsc-three-linear-layers"
seed = 30
log_every_n_steps = 5
load_name = "/home/lch121600/ADLSlab/mase/mase_output/jsc-three-linear-layers_classification_jsc_2024-02-10/software/training_ckpts/best.ckpt"
load_type = "pl"

[passes.quantize]
by = "type"
[passes.quantize.default.config]
name = "NA"
[passes.quantize.linear.config]
name = "integer"
"data_in_width" = 8
"data_in_frac_width" = 4
"weight_width" = 8
"weight_frac_width" = 4
"bias_width" = 8
"bias_frac_width" = 4

[search.search_space]
name = "LAB4/graph_lab4"

[search.search_space.setup]
by = "name"

[search.search_space.seed.default.config]
# the only choice "NA" is used to indicate that layers are not quantized by default
name = ["NA"]

[search.search_space.seed.linear.config]
# if search.search_space.setup.by = "type", this seed will be used to quantize all torch.nn.Linear/ F.linear
name = ["integer"]
data_in_width = [4, 8]
data_in_frac_width = ["NA"] # "NA" means data_in_frac_width = data_in_width // 2
weight_width = [2, 4, 8]
weight_frac_width = ["NA"]
bias_width = [2, 4, 8]
bias_frac_width = ["NA"]

[search.search_space.seed.seq_blocks_2.config]
#  the mase graph node with name "seq_blocks_2"
name = ["output_only"]
channel_multiplier = [2,4]

[search.search_space.seed.seq_blocks_4.config]
#  the mase graph node with name "seq_blocks_2"
name = ["output_only"]
channel_multiplier = [2,4,8]

[search.search_space.seed.seq_blocks_6.config]
#  the mase graph node with name "seq_blocks_2"
name = ["input_only"]
#channel_multiplier = [2]



[search.strategy]
name = "optuna"
eval_mode = true

[search.strategy.sw_runner.basic_evaluation]
data_loader = "val_dataloader"
num_samples = 512

[search.strategy.hw_runner.average_bitwidth]
compare_to = 32 # compare to FP32

[search.strategy.setup]
n_jobs = 1
n_trials = 10
timeout = 20000
#sampler = "tpe"
sampler = "brute-force"
# sum_scaled_metrics = true # single objective
# direction = "maximize"
sum_scaled_metrics = false # multi objective

[search.strategy.metrics]
# loss.scale = 1.0
# loss.direction = "minimize"
accuracy.scale = 1.0
accuracy.direction = "maximize"
average_bitwidth.scale = 0.2
average_bitwidth.direction = "minimize"
```
The output is:

```python
 60%|█████████████████████████████████████▊                         | 6/10 [00:06<00:04,  1.08s/it, 6.46/20000 seconds]
INFO     Best trial(s):
Best trial(s):
|    |   number | software_metrics                  | hardware_metrics                                | scaled_metrics                              |
|----+----------+-----------------------------------+-------------------------------------------------+---------------------------------------------|
|  0 |        1 | {'loss': 1.609, 'accuracy': 0.21} | {'average_bitwidth': 32, 'memory_density': 1.0} | {'accuracy': 0.21, 'average_bitwidth': 6.4} |
INFO     Searching is completed

```







